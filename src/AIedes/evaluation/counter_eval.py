import numpy as np
import torch
from scipy.stats import linregress



def compute_metrics(model, loader, device):
    """
    Compute comprehensive performance metrics for model evaluation.
    
    Args:
        model: The PyTorch model to evaluate
        loader: DataLoader containing evaluation data
        device: Device to run computation on (CPU or GPU)
        
    Returns:
        dict: Dictionary of metrics including R², RMSE, and specialized metrics
    """
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds_list.append(model(x))
            targets_list.append(y)
            
    # Gather predictions and targets
    preds   = torch.cat(preds_list).view(-1)
    targets = torch.cat(targets_list).view(-1)
    
    # Convert to numpy for additional calculations
    preds_np = preds.cpu().numpy().flatten()
    targets_np = targets.cpu().numpy().flatten()
    
    # Standard metrics (PyTorch calculation)
    rmse = torch.sqrt(torch.mean((targets - preds) ** 2)).item()
    r2 = 1 - torch.sum((targets - preds) ** 2).item() / torch.sum((targets - torch.mean(targets)) ** 2).item()
    
    # Pearson correlation coefficient (using scipy)
    slope, intercept, r_value, _, _ = linregress(targets_np, preds_np)
    pearson_r2 = r_value**2
    
    # Log-transformed metrics
    log_preds = np.log1p(preds_np)
    log_targets = np.log1p(targets_np)
    mean_log_target = np.mean(log_targets)
    ss_tot_log = np.sum((log_targets - mean_log_target)**2)
    ss_res_log = np.sum((log_targets - log_preds)**2)
    log_r2 = 1 - (ss_res_log / ss_tot_log) if ss_tot_log != 0 else 0
    log_slope, log_intercept, _, _, _ = linregress(log_targets, log_preds)
    
    # Non-zero analysis
    non_zero_idx = targets_np > 1./9
    targets_nonzero = targets_np[non_zero_idx]
    preds_nonzero = preds_np[non_zero_idx]
    
    # Compute R² for non-zero data
    if len(targets_nonzero) > 1:
        mean_nonzero = np.mean(targets_nonzero)
        ss_tot_nonzero = np.sum((targets_nonzero - mean_nonzero)**2)
        ss_res_nonzero = np.sum((targets_nonzero - preds_nonzero)**2)
        nonzero_r2 = 1 - ss_res_nonzero / ss_tot_nonzero if ss_tot_nonzero != 0 else 0
        nonzero_slope, nonzero_intercept, _, _, _ = linregress(targets_nonzero, preds_nonzero)
    else:
        nonzero_r2 = 0
        nonzero_slope, nonzero_intercept = 0, 0
    
    # Classification accuracy (zero vs. non-zero)
    actual_binary = (targets_np > 0).astype(int)
    pred_binary = (preds_np >= 1).astype(int) # less that a eggs per week means no eggs
    binary_accuracy = np.mean(actual_binary == pred_binary)
    
    # Calculate counts for confusion matrix
    true_positives = np.sum((actual_binary == 1) & (pred_binary == 1))
    true_negatives = np.sum((actual_binary == 0) & (pred_binary == 0))
    false_positives = np.sum((actual_binary == 0) & (pred_binary == 1))
    false_negatives = np.sum((actual_binary == 1) & (pred_binary == 0))
    
    # Calculate rates with proper handling of division by zero
    FPR = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    FNR = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
    TPR = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    TNR = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # Return comprehensive metrics dictionary
    metrics = {
        # Standard metrics
        'r2': r2,
        'rmse': rmse,
        'pearson_r2': pearson_r2,
        'slope': slope,
        'intercept': intercept,
        
        # Log-transformed metrics
        'log_r2': log_r2,
        'log_slope': log_slope,
        'log_intercept': log_intercept,
        
        # Non-zero metrics
        'nonzero_r2': nonzero_r2,
        'nonzero_slope': nonzero_slope,
        'nonzero_intercept': nonzero_intercept,
        'nonzero_count': len(targets_nonzero),
        
        # Classification metrics
        'binary_accuracy': binary_accuracy,
        'FPR': FPR,
        'FNR': FNR,
        'TPR': TPR,
        'TNR': TNR,
        
        # Raw data for plotting
        'predictions': preds,
        'targets': targets,
        'predictions_np': preds_np,
        'targets_np': targets_np
    }
    
    return metrics

