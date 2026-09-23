import numpy as np
import torch
from scipy.stats import linregress


def compute_metrics_from_arrays(targets_np, preds_np):
    """Canonical metrics; undefined small/constant-subset measures are None."""
    y = np.asarray(targets_np, dtype=float).reshape(-1)
    p = np.asarray(preds_np, dtype=float).reshape(-1)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Expected aligned finite targets and predictions")
    if (y < 0).any() or (p < 0).any():
        raise ValueError("Egg rates must be nonnegative")

    def r2(a, b):
        total = np.sum((a-a.mean())**2) if len(a) else 0.
        return float(1-np.sum((a-b)**2)/total) if total > 0 else None

    def regression(a, b):
        if len(a) < 2 or np.ptp(a) == 0:
            return None, None, None
        slope, intercept, correlation, _, _ = linregress(a, b)
        return float(slope), float(intercept), float(correlation**2) if np.isfinite(correlation) else None

    slope, intercept, pearson = regression(y, p)
    ly, lp = np.log1p(y), np.log1p(p)
    log_slope, log_intercept, _ = regression(ly, lp)
    nonzero = y > 1./9.
    nz_slope, nz_intercept, _ = regression(y[nonzero], p[nonzero])
    actual, predicted = y > 0, p >= 1
    tp, tn = int(np.sum(actual & predicted)), int(np.sum(~actual & ~predicted))
    fp, fn = int(np.sum(~actual & predicted)), int(np.sum(actual & ~predicted))
    divide = lambda a, b: float(a/b) if b else None
    accuracy = float(np.mean(actual == predicted)) if len(y) else None
    return {
        "n": len(y), "r2": r2(y, p), "log_r2": r2(ly, lp),
        "rmse": float(np.sqrt(np.mean((y-p)**2))) if len(y) else None,
        "log_mse": float(np.mean((ly-lp)**2)) if len(y) else None,
        "pearson_r2": pearson, "slope": slope, "intercept": intercept,
        "log_slope": log_slope, "log_intercept": log_intercept,
        "nonzero_r2": r2(y[nonzero], p[nonzero]), "nonzero_slope": nz_slope,
        "nonzero_intercept": nz_intercept, "nonzero_count": int(nonzero.sum()),
        "binary_accuracy": accuracy, "accuracy": accuracy,
        "TPR": divide(tp, tp+fn), "TNR": divide(tn, tn+fp),
        "FPR": divide(fp, fp+tn), "FNR": divide(fn, fn+tp),
        "sensitivity": divide(tp, tp+fn), "specificity": divide(tn, tn+fp),
        "absence_baseline": float((~actual).mean()) if len(y) else None,
        "positive_fraction": float(actual.mean()) if len(y) else None,
    }


def compute_metrics(model, loader, device):
    """Collect ordered predictions and use the common array metric calculation."""
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.append(model(x.to(device)).view(-1))
            targets.append(y.to(device).view(-1))
    pred = torch.cat(predictions) if predictions else torch.empty(0, device=device)
    target = torch.cat(targets) if targets else torch.empty(0, device=device)
    p, y = pred.cpu().numpy(), target.cpu().numpy()
    result = compute_metrics_from_arrays(y, p)
    result.update(predictions=pred, targets=target, predictions_np=p, targets_np=y)
    return result
