import contextlib
import io

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def counter_candidate_model(candidate, input_dim):
    """Build the revision's positive feed-forward network from explicit widths.

    The legacy network halves its hidden width per layer (floor, never below 16),
    so a candidate's explicit widths must agree with that rule.
    """
    widths = list(candidate["widths"])
    expected = [max(widths[0] // (2 ** i), 16) for i in range(len(widths))]
    if expected != widths:
        raise ValueError(f"Architecture {widths} does not match the halving rule {expected}")
    with contextlib.redirect_stdout(io.StringIO()):
        model = PositiveFeedForwardAbundanceModel(
            input_dim, widths[0], 1, len(widths), candidate["dropout"], device="cpu"
        )
        return ZINBPositiveFF(model) if candidate.get("loss") == "ZINB" else model


def predict_counter_rates(model, x):
    """Evaluation-mode weekly-rate prediction with explicit validity checks."""
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(x, dtype=torch.float32)).view(-1).cpu().numpy()
    if not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Invalid model prediction")
    return prediction


class PositiveFeedForwardAbundanceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_hidden_layers=1, dropout=0.0, use_skip_connections=False,
                 device='cpu', dtype=torch.float32):
        super(PositiveFeedForwardAbundanceModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.use_skip_connections = use_skip_connections if num_hidden_layers > 0 else False
        self.hidden_dim = hidden_dim

        layers = []
        current_dim = input_dim

        # Build hidden layers
        for i in range(num_hidden_layers):
            in_features = current_dim
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            print(f"Layer {i}: in_features={in_features}, out_features={hidden_dim}, dropout={dropout}")
            current_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 16)  # Reduce, but not below 16

        self.model = nn.Sequential(*layers)
        # Final classification/regression layer
        self.fc = nn.Linear(current_dim, output_dim)
        
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {output_dim}")
        print(f"Num hidden layers: {num_hidden_layers}, Dropout: {dropout}")
        print(f"Use skip connections: {self.use_skip_connections}")
        
        # Create a dedicated skip projection if needed
        if self.use_skip_connections and input_dim != current_dim:
            self.skip_connection = nn.Linear(input_dim, output_dim)
        else:
            self.skip_connection = None

    def forward(self, x):
        x = x.to(self.device, self.dtype)
        if x.ndim == 3:
            x = x.view(x.size(0), -1)
        last_hidden = self.model(x)
        if self.use_skip_connections:
            if self.skip_connection is not None:
                skip = self.skip_connection(x)
            else:
                skip = x
            last_hidden += skip
        out = F.softplus(self.fc(last_hidden))
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ZINBPositiveFF(nn.Module):
    """
    Wraps your PositiveFeedForwardAbundanceModel to add a π-head for ZINB.
    - Uses base.model (backbone) and base.fc (for μ_raw).
    - Adds head_pi (hidden -> 1) for logit(π).
    - forward_training(x) -> {"mu_raw", "logit_pi"} for the ZINB loss
    - forward(x) -> single number: E[Y|x] = (1-π)*μ  (so it matches MSE/NB interface)
    """
    def __init__(self, base_model: PositiveFeedForwardAbundanceModel):
        super().__init__()
        self.base = base_model

        # Feature dimension is exactly the input dim of base.fc
        feat_dim = self.base.fc.in_features
        self.head_pi = nn.Linear(feat_dim, 1)   # new tiny head: logit(π)

        # small epsilon for positivity
        self.eps = 1e-8

    def _preprocess(self, x):
        # mimic base.forward's first lines (device/dtype + optional flatten)
        x = x.to(self.base.device, self.base.dtype)
        if x.ndim == 3:
            x = x.view(x.size(0), -1)
        return x

    def _hidden(self, x):
        """
        Reproduce the hidden representation BEFORE 'fc', including
        optional skip connection exactly like the base model.
        """
        x = self._preprocess(x)
        h = self.base.model(x)  # backbone output

        if self.base.use_skip_connections:
            skip = None
            if getattr(self.base, "skip_connection", None) is not None:
                skip = self.base.skip_connection(x)
            else:
                skip = x
            # Only add if shapes match (your code assumes they do when enabled)
            if skip.shape == h.shape:
                h = h + skip
            else:
                # shape mismatch: skip adding to stay safe
                pass
        return h

    def forward_training(self, x):
        """
        Return raw params for ZINB loss: μ_raw (any real), logit_π (any real).
        """
        h = self._hidden(x)
        mu_raw   = self.base.fc(h)        # DO NOT softplus here; let loss handle it
        logit_pi = self.head_pi(h)
        return {"mu_raw": mu_raw.squeeze(-1), "logit_pi": logit_pi.squeeze(-1)}

    def forward(self, x):
        out = self.forward_training(x)
        mu  = F.softplus(out["mu_raw"]) + self.eps     # (B,1) or (B,) depending on your forward_training
        pi  = torch.sigmoid(out["logit_pi"])
        expected = (1.0 - pi) * mu                     # keep as (B,1)
        return expected   # <-- remove .squeeze(-1)
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class PositiveLSTMAbundanceModel(nn.Module):
    """
    An LSTM-based model for abundance prediction with non-negative outputs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, 
                 num_layers=1, dropout=0.0, use_skip_connections=False,
                 device='cpu', dtype=torch.float32):
        super(PositiveLSTMAbundanceModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.use_skip_connections = use_skip_connections
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.to(self.device, self.dtype)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.batch_norm(last_hidden)
        if self.use_skip_connections:
            skip = x[:, -1, :]
            last_hidden += skip
        # Apply softplus to guarantee non-negative outputs
        out = torch.nn.functional.softplus(self.fc(last_hidden))
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PositiveMultiHeadAttentionModel(nn.Module):
    """
    A model with multi-head self-attention that predicts abundance
    and uses a softplus activation to enforce non-negative outputs.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4, num_layers=1,
                 dropout=0.0, device='cpu', dtype=torch.float32):
        super(PositiveMultiHeadAttentionModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        # Final layer maps from the (aggregated) input dimension to output dimension.
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.to(self.device, self.dtype)
        for attn in self.attention_layers:
            x, _ = attn(x, x, x)
            x = self.dropout(x)
        x = x.mean(dim=1)
        out = torch.nn.functional.softplus(self.fc(x))
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    

class PositiveHurdleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_hidden_layers=2, dropout=0.3, device='cuda'):
        super().__init__()
        self.device = device
        
        # Shared features extractor
        layers = []
        current_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 32)  # Gradually reduce hidden dim but not below 32
        
        self.features = nn.Sequential(*layers)
        
        # Binary classification head (zero vs non-zero)
        self.zero_predictor = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Linear(current_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Regression head for non-zero values
        self.count_predictor = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Linear(current_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """Standard forward pass - returns just the predictions"""
        features = self.features(x)
        zero_prob = self.zero_predictor(features)
        count_pred = F.softplus(self.count_predictor(features))
        # Combine predictions: use count only when zero_prob < 0.5
        final_pred = torch.where(zero_prob < 0.5, count_pred, torch.zeros_like(count_pred))
        return final_pred
    
    def forward_training(self, x):
        """Training forward pass - returns both zero probabilities and counts"""
        features = self.features(x)
        zero_prob = self.zero_predictor(features)
        count_pred = F.softplus(self.count_predictor(features))
        return {"zero_prob": zero_prob, "count_pred": count_pred}
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
