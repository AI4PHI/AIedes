import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_bin_weights(targets, bins):
    # targets: 1D numpy array or PyTorch tensor
    # bins: list or 1D array of bin edges
    # Output: weights, np.array, len = len(bins) + 1 (inverse freq)
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    class_labels = np.digitize(targets, bins, right=False)  # see np.digitize docs
    bincounts = np.bincount(class_labels, minlength=len(bins) + 1)
    # Avoid division by zero, assign zero weight to empty bins
    weights = 1.0 / (bincounts + 1e-8)
    weights = weights / np.mean(weights)
    return weights

def estimate_theta_from_data(targets):
    """
    Estimate the negative binomial dispersion parameter theta from data.
    targets: 1D numpy array or torch tensor
    Returns: float (theta)
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    mean = np.mean(targets)
    var = np.var(targets)
    # Avoid division by zero and negative theta
    if var > mean:
        theta = mean ** 2 / (var - mean)
        theta = max(theta, 1e-2)  # avoid too small values
    else:
        theta = 1.0  # fallback if data is not overdispersed
    return float(theta)

# losses

class HurdleLoss(nn.Module):
    def __init__(self, zero_weight=1.0, nonzero_weight=1.0):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        self.bce_loss = nn.BCELoss(reduction='none')
        
    def forward(self, predictions, targets, model=None, x=None):
        if model is not None and x is not None:
            # Training mode - get detailed predictions
            predictions = model.forward_training(x)
            zero_prob = predictions["zero_prob"]
            count_pred = predictions["count_pred"]
        else:
            # Evaluation mode - use standard predictions
            count_pred = predictions
            zero_prob = (count_pred < 0.5).float()
            
        if len(targets.shape) == 1:
            targets = targets.unsqueeze(1)
            
        zero_mask = (targets == 0).float()
        
        if len(zero_prob.shape) != len(zero_mask.shape):
            zero_prob = zero_prob.view(zero_mask.shape)
            
        # Apply weighted BCE loss for zero classification
        zero_loss_raw = self.bce_loss(zero_prob, zero_mask)
        zero_loss = (zero_mask * self.zero_weight * zero_loss_raw).sum() / (zero_mask.sum() + 1e-8)
        
        # Apply weighted regression loss for non-zero values
        non_zero_mask = (1 - zero_mask)
        if non_zero_mask.sum() > 0:
            if len(count_pred.shape) != len(targets.shape):
                count_pred = count_pred.view(targets.shape)
            count_loss_raw = (torch.log1p(count_pred) - torch.log1p(targets)) ** 2
            count_loss = (non_zero_mask * self.nonzero_weight * count_loss_raw).sum() / (non_zero_mask.sum() + 1e-8)
        else:
            count_loss = torch.tensor(0.0, device=targets.device)
        
        return zero_loss + count_loss
    
class WeightedMSELoss(nn.Module):
    def __init__(self, zero_weight=1.0, nonzero_weight=3.0):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight

    def forward(self, pred, target):
        weight = torch.where(
            target > 0,
            torch.tensor(self.nonzero_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.zero_weight, device=target.device, dtype=target.dtype)
        )
        return (weight * (pred - target) ** 2).mean()

class WeightedMSLELoss(nn.Module):
    def __init__(self, zero_weight=1.0, nonzero_weight=3.0):
        super().__init__()
        self.zero_weight = zero_weight
        self.nonzero_weight = nonzero_weight
        
    def forward(self, pred, target):
        # Create weight tensor based on whether target is zero or not
        weight = torch.where(
            target > 0,
            torch.tensor(self.nonzero_weight, device=target.device, dtype=target.dtype),
            torch.tensor(self.zero_weight, device=target.device, dtype=target.dtype)
        )
        # Apply weighted mean squared logarithmic error
        return torch.mean(weight * (torch.log1p(pred) - torch.log1p(target)) ** 2)

class NegativeBinomialLoss(nn.Module):
    def __init__(self, theta=1.0):
        super().__init__()
        self.theta = theta

    def forward(self, pred, target):
        theta = torch.tensor(self.theta, device=target.device, dtype=target.dtype)
        eps = 1e-8
        loss = (torch.lgamma(target + theta)
                - torch.lgamma(theta)
                - torch.lgamma(target + 1)
                + theta * torch.log(theta + eps)
                + target * torch.log(pred + eps)
                - (target + theta) * torch.log(pred + theta + eps))
        return -torch.mean(loss)


class FrequencyWeightedLoss(nn.Module):
    def __init__(self, bins, weights, device, log_loss=False):
        """
        bins: List/tensor of bin edges. Should match np.digitize convention.
        weights: List/tensor of weights, length == len(bins) + 1.
        device: torch device.
        log_loss: If True, use log1p transform.
        """
        super().__init__()
        self.bins = torch.tensor(bins, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float, device=device)
        assert len(self.weights) == len(self.bins) + 1, \
            f"weights must have length len(bins)+1 ({len(self.bins)+1}), got {len(self.weights)}"
        self.device = device
        self.log_loss = log_loss

    def forward(self, pred, target):
        target_flat = target.view(-1)
        pred_flat = pred.view(-1)
        # This matches np.digitize(right=False)
        bin_indices = torch.bucketize(target_flat, self.bins, right=False)
        sample_weights = self.weights[bin_indices]

        if self.log_loss:
            pred_flat = torch.log1p(pred_flat)
            target_flat = torch.log1p(target_flat)

        loss = sample_weights * (pred_flat - target_flat) ** 2
        return loss.mean()

class FrequencyWeightedNegativeBinomialLoss(nn.Module):
    def __init__(self, bins, weights, device, theta=1.0):
        """
        bins: List/tensor of bin edges. Should match np.digitize convention.
        weights: List/tensor of weights, length == len(bins) + 1.
        device: torch device.
        theta: Dispersion parameter for Negative Binomial.
        """
        super().__init__()
        self.bins = torch.tensor(bins, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float, device=device)
        assert len(self.weights) == len(self.bins) + 1, \
            f"weights must have length len(bins)+1 ({len(self.bins)+1}), got {len(self.weights)}"
        self.device = device
        self.theta = theta

    def forward(self, pred, target):
        theta = torch.tensor(self.theta, device=target.device, dtype=target.dtype)
        eps = 1e-8
        target_flat = target.view(-1)
        pred_flat = pred.view(-1)
        bin_indices = torch.bucketize(target_flat, self.bins, right=False)
        sample_weights = self.weights[bin_indices]

        # Negative binomial log-likelihood (per sample)
        loss = (torch.lgamma(target_flat + theta)
                - torch.lgamma(theta)
                - torch.lgamma(target_flat + 1)
                + theta * torch.log(theta + eps)
                + target_flat * torch.log(pred_flat + eps)
                - (target_flat + theta) * torch.log(pred_flat + theta + eps))
        weighted_loss = -sample_weights * loss
        return weighted_loss.mean()

class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial (NB2).
    Learns π(x) from the model and θ as a global trainable parameter.
    """
    def __init__(self, theta_init: float = 0.3, eps: float = 1e-8,
                 learn_theta: bool = True, theta_min: float = 1e-3, theta_max: float = 1e3):
        super().__init__()
        self.eps = eps
        self.learn_theta = learn_theta
        self.theta_min = theta_min
        self.theta_max = theta_max

        if learn_theta:
            # raw param -> softplus -> clamp => θ > 0 and bounded
            self.theta_raw = nn.Parameter(torch.tensor(float(theta_init)))
        else:
            self.register_buffer("theta_const", torch.tensor(float(theta_init)))

    @property
    def theta(self):
        if self.learn_theta:
            th = F.softplus(self.theta_raw) + self.eps
            return th.clamp(self.theta_min, self.theta_max)
        else:
            return self.theta_const

    @staticmethod
    def _unpack_pred(pred):
        if isinstance(pred, dict):
            return pred["mu_raw"], pred["logit_pi"]
        if isinstance(pred, (tuple, list)) and len(pred) == 2:
            return pred[0], pred[1]
        assert pred.size(-1) == 2, "pred must have 2 columns: [mu_raw, logit_pi]"
        return pred[..., 0], pred[..., 1]

    def forward(self, pred=None, target=None, model=None, x=None, predictions=None):
        if pred is None and predictions is not None:
            pred = predictions
        if pred is None:
            if (model is None) or (x is None):
                raise ValueError("Provide pred OR (model and x).")
            pred = model.forward_training(x)

        mu_raw, logit_pi = self._unpack_pred(pred)

        y = target.to(dtype=mu_raw.dtype).view(-1)
        mu = F.softplus(mu_raw).to(y.dtype).view(-1) + self.eps
        logit_pi = logit_pi.to(y.dtype).view(-1)

        theta = self.theta.to(y.device).to(y.dtype)

        # NB log pmf (NB2)
        log_nb  = (torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
                   + theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))
                   + y * (torch.log(mu + self.eps) - torch.log(theta + mu + self.eps)))
        log_nb0 = theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))

        # Zero-inflation
        log_pi   = -F.softplus(-logit_pi)   # log(sigmoid)
        log1m_pi = -F.softplus(logit_pi)    # log(1 - sigmoid)

        is_zero      = (y <= 0.5)
        loglik_zero  = torch.logaddexp(log_pi, log1m_pi + log_nb0)
        loglik_pos   = log1m_pi + log_nb
        nll          = -torch.where(is_zero, loglik_zero, loglik_pos)

        return nll.mean()
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrequencyWeightedZINBLoss(nn.Module):
    def __init__(self, bins, weights, device, theta=1.0, eps: float = 1e-8):
        """
        Frequency-weighted Zero-Inflated Negative Binomial (NB2: mean=mu, size=theta).

        Args:
            bins: list/tensor of bin edges for targets (np.digitize convention).
            weights: list/tensor of inverse-frequency weights; length == len(bins)+1.
            device: torch device.
            theta: fixed dispersion (size) parameter of NB.
            eps: numerical stability constant.
        """
        super().__init__()
        self.bins = torch.as_tensor(bins, device=device)
        self.weights = torch.as_tensor(weights, dtype=torch.float, device=device)
        assert self.weights.numel() == self.bins.numel() + 1, \
            f"weights must have length len(bins)+1 ({self.bins.numel()+1}), got {self.weights.numel()}"
        self.device = device
        self.theta = float(theta)
        self.eps = float(eps)

    @staticmethod
    def _unpack_pred(pred):
        """
        Accepts:
          - dict with keys {"mu_raw", "logit_pi"}  (preferred)
          - tuple/list (mu_raw, logit_pi)
          - tensor [..., 2] with columns [mu_raw, logit_pi]
        Returns (mu_raw, logit_pi)
        """
        if isinstance(pred, dict):
            return pred["mu_raw"], pred["logit_pi"]
        if isinstance(pred, (tuple, list)) and len(pred) == 2:
            return pred[0], pred[1]
        # assume tensor with last dim = 2
        assert pred.size(-1) == 2, "pred must have 2 columns: [mu_raw, logit_pi]"
        return pred[..., 0], pred[..., 1]

    def forward(self, pred, target):
        theta = torch.as_tensor(self.theta, device=target.device, dtype=target.dtype)
        mu_raw, logit_pi = self._unpack_pred(pred)

        # Flatten & type/shape align
        y = target.to(dtype=mu_raw.dtype).view(-1)
        mu = F.softplus(mu_raw).to(y.dtype).view(-1) + self.eps          # μ > 0
        logit_pi = logit_pi.to(y.dtype).view(-1)

        # Frequency weights by target bins (same as your FWNB)
        bin_idx = torch.bucketize(y, self.bins, right=False)
        sample_w = self.weights[bin_idx]

        # NB log pmf (NB2)
        log_nb = (
            torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
            + theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))
            + y * (torch.log(mu + self.eps) - torch.log(theta + mu + self.eps))
        )
        log_nb0 = theta * (torch.log(theta + self.eps) - torch.log(theta + mu + self.eps))

        # Zero-inflation terms
        log_pi   = -F.softplus(-logit_pi)   # log(sigmoid)
        log1m_pi = -F.softplus(logit_pi)    # log(1 - sigmoid)

        is_zero = (y <= 0.5)
        loglik_zero = torch.logaddexp(log_pi, log1m_pi + log_nb0)   # log(π + (1-π) NB(0))
        loglik_pos  = log1m_pi + log_nb                              # log(1-π) + log NB(y)

        nll = -torch.where(is_zero, loglik_zero, loglik_pos)         # per-sample NLL
        return (sample_w * nll).mean()


def get_criterion(loss_type, params, train_loader, device):
    """
    Initialize and return the appropriate loss criterion based on specified parameters.
    
    Args:
        loss_type (str): Type of loss function to use
        params (dict): Dictionary of parameters including weights, bins, etc.
        train_loader (DataLoader): DataLoader containing training data
        device (torch.device): Device to place tensors on
    
    Returns:
        tuple: (criterion_class, criterion_params)
    """
    bins = params.get("bins", None)
    
    if loss_type == "MSE":
        criterion_class, criterion_params = nn.MSELoss, {}
    elif loss_type == "Huber":
        criterion_class, criterion_params = nn.SmoothL1Loss, {}
    elif loss_type == "WeightedMSE":
        criterion_class, criterion_params = WeightedMSELoss, {"nonzero_weight": params["nonzero_weight"], "zero_weight": params["zero_weight"]}
    elif loss_type == "WeightedMSLE":
        criterion_class, criterion_params = WeightedMSLELoss,  {"nonzero_weight": params["nonzero_weight"], "zero_weight": params["zero_weight"]}
    elif loss_type == "NegativeBinomial":
        criterion_class, criterion_params = NegativeBinomialLoss, {"theta": params.get("theta", 1.0)}
    elif loss_type == "Hurdle":
        criterion_class, criterion_params = HurdleLoss,  {"nonzero_weight": params["nonzero_weight"], "zero_weight": params["zero_weight"]}
    elif loss_type == "FrequencyWeightedMSE":
        assert bins is not None, "bins must be provided for FrequencyWeighted loss"
        # Compute weights from train targets
        train_targets = train_loader.dataset.tensors[1].detach().cpu().numpy().ravel()
        weights = compute_bin_weights(train_targets, bins)
        print("  bins =", bins)
        print("  raw weights =", weights)
        print("  min, max, mean =", weights.min(), weights.max(), weights.mean())

        criterion_class = FrequencyWeightedLoss
        criterion_params = {
            "bins": bins,
            "weights": weights,
            "device": device,
            "log_loss": False
        }
    elif loss_type == "FrequencyWeightedMSLE":
        assert bins is not None, "bins must be provided for FrequencyWeighted loss"
        # Compute weights from train targets
        train_targets = train_loader.dataset.tensors[1].detach().cpu().numpy().ravel()
        weights = compute_bin_weights(train_targets, bins)
        print("  bins =", bins)
        print("  raw weights =", weights)
        print("  min, max, mean =", weights.min(), weights.max(), weights.mean())

        criterion_class = FrequencyWeightedLoss
        criterion_params = {
            "bins": bins,
            "weights": weights,
            "device": device,
            "log_loss": True
        }
    elif loss_type == "FrequencyWeightedNegativeBinomial":
        assert bins is not None, "bins must be provided for FrequencyWeightedNegativeBinomial loss"
        train_targets = train_loader.dataset.tensors[1].detach().cpu().numpy().ravel()
        weights = compute_bin_weights(train_targets, bins)
        # Estimate theta if not provided
        if "theta" in params and params["theta"] is not None:
            theta = params["theta"]
        else:
            theta = estimate_theta_from_data(train_targets)
            print(f"  [auto] estimated theta = {theta:.4f}")
        print("  bins =", bins)
        print("  raw weights =", weights)
        print("  min, max, mean =", weights.min(), weights.max(), weights.mean())
        criterion_class = FrequencyWeightedNegativeBinomialLoss
        criterion_params = {
            "bins": bins,
            "weights": weights,
            "device": device,
            "theta": theta
        }
    elif loss_type == "ZINBLoss":
        # θ: provided or estimated from TRAIN targets
        train_targets = train_loader.dataset.tensors[1].detach().cpu().numpy().ravel()
        theta_init = params.get("theta", None)
        if theta_init is None:
            # a reasonable starting point; training will update it
            theta_init = estimate_theta_from_data(train_targets)
            print(f"  [init] theta_init (ZINB learnable) = {theta_init:.4f}")

        learn_theta = params.get("learn_theta", True)  # default: learn it
        criterion_class  = ZINBLoss
        criterion_params = {
            "theta_init": float(theta_init),
            "learn_theta": bool(learn_theta)
        }
    elif loss_type == "FrequencyWeightedZeroInflatedNegativeBinomial":
        assert bins is not None, "bins must be provided for FrequencyWeightedZeroInflatedNegativeBinomial"
        # Compute inverse-frequency bin weights from TRAIN targets (same as FWNB)
        train_targets = train_loader.dataset.tensors[1].detach().cpu().numpy().ravel()
        weights = compute_bin_weights(train_targets, bins)

        # Theta: provided or auto-estimated (same rule as FWNB)
        theta = params.get("theta", None)
        if theta is None:
            theta = estimate_theta_from_data(train_targets)
            print(f"  [auto] estimated theta (FW-ZINB) = {theta:.4f}")

        print("  bins =", bins)
        print("  raw weights =", weights)
        print("  min, max, mean =", weights.min(), weights.max(), weights.mean())

        # Use the ZINB loss that expects (mu_raw, logit_pi)
        criterion_class = FrequencyWeightedZINBLoss
        criterion_params = {
            "bins": bins,
            "weights": weights,
            "device": device,
            "theta": theta
        }
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    
    return criterion_class, criterion_params





