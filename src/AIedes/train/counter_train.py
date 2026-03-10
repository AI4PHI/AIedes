import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib.gridspec as gridspec
from AIedes.evaluation.counter_eval import compute_metrics
from AIedes.utils.counter_losses import HurdleLoss, FrequencyWeightedZINBLoss, ZINBLoss
from AIedes.models.counter_models import PositiveFeedForwardAbundanceModel, ZINBPositiveFF

def _compute_and_store_metrics(model, train_loader, test_loader, device, metrics_history):
    """Computes metrics for train and test sets and stores them."""
    train_metrics = compute_metrics(model, train_loader, device)
    test_metrics = compute_metrics(model, test_loader, device)

    # Store all metrics in history
    for key in train_metrics:
        if key not in ['predictions', 'targets', 'predictions_np', 'targets_np']:
            metric_name = f'train_{key}'
            if metric_name in metrics_history:
                metrics_history[metric_name].append(train_metrics[key])
    
    for key in test_metrics:
        if key not in ['predictions', 'targets', 'predictions_np', 'targets_np']:
            metric_name = f'test_{key}'
            if metric_name in metrics_history:
                metrics_history[metric_name].append(test_metrics[key])
    
    return train_metrics, test_metrics


def _plot_training_progress(epoch, num_epochs, train_losses, test_losses, metrics_history,
                            train_metrics, test_metrics, train_loss_avg, test_loss_avg,
                            scheduler, window_size, save_loss_fig):
    """Plots the training progress including losses and various metrics."""
    clear_output(wait=True)
    # Compute running averages
    train_loss_avg_plot = np.convolve(train_losses, np.ones(window_size) / window_size, mode='valid')
    test_loss_avg_plot = np.convolve(test_losses, np.ones(window_size) / window_size, mode='valid')
    
    # Create an outer GridSpec with 6 rows.
    fig = plt.figure(figsize=(14, 30))
    outer = gridspec.GridSpec(6, 1, height_ratios=[1, 1, 1, 1, 1, 1])
    
    # ----------------- Row 0: Loss Plots (Train & Test) -----------------
    gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0])
    ax0 = plt.subplot(gs0[0])
    ax0.plot(train_losses, label='Train Loss')
    ax0.plot(range(window_size - 1, len(train_losses)), train_loss_avg_plot, label=f'Running Avg (w={window_size})')
    ax0.set_title(f"Epoch {epoch + 1} | Train Loss: {train_loss_avg:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")
    ax0.legend(); ax0.grid()
    
    ax1 = plt.subplot(gs0[1])
    ax1.plot(test_losses, label='Test Loss')
    ax1.plot(range(window_size - 1, len(test_losses)), test_loss_avg_plot, label=f'Running Avg (w={window_size})')
    ax1.set_title(f"Epoch {epoch + 1} | Test Loss: {test_loss_avg:.4f}")
    ax1.legend(); ax1.grid()
    
    # ----------------- Row 1: R² and RMSE (All Data) -----------------
    gs1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1])
    ax2 = plt.subplot(gs1[0])
    ax2.plot(metrics_history['train_r2'], label='Train R²')
    ax2.plot(metrics_history['test_r2'], label='Test R²')
    ax2.set_title(f"Epoch {epoch + 1} | Train R²: {train_metrics['r2']:.4f}, Test R²: {test_metrics['r2']:.4f}")
    ax2.legend(); ax2.grid()
    
    ax3 = plt.subplot(gs1[1])
    ax3.plot(metrics_history['train_rmse'], label='Train RMSE')
    ax3.plot(metrics_history['test_rmse'], label='Test RMSE')
    ax3.set_title(f"Epoch {epoch + 1} | Train RMSE: {train_metrics['rmse']:.4f}, Test RMSE: {test_metrics['rmse']:.4f}")
    ax3.legend(); ax3.grid()
    
    # ----------------- Row 2: Scatter Plots (All Data) -----------------
    preds_train_flat = train_metrics['predictions'].cpu().numpy().flatten()
    target_train_flat = train_metrics['targets'].cpu().numpy().flatten()
    preds_test_flat = test_metrics['predictions'].cpu().numpy().flatten()
    targets_test_flat = test_metrics['targets'].cpu().numpy().flatten()
    
    gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[2])
    ax4 = plt.subplot(gs2[0])
    slope_train, intercept_train, r_value_train, _, _ = linregress(target_train_flat, preds_train_flat)
    ax4.scatter(target_train_flat, preds_train_flat, marker='.', alpha=0.5)
    ax4.plot(target_train_flat, slope_train * target_train_flat + intercept_train, 'r-', label='Fit')
    ax4.set_title(f"Train Data (Pearson R²: {train_metrics['pearson_r2']:.4f})")
    ax4.set_xlabel("Actual"); ax4.set_ylabel("Predicted")
    ax4.legend(); ax4.grid()
    
    ax5 = plt.subplot(gs2[1])
    slope_test, intercept_test, r_value_test, _, _ = linregress(targets_test_flat, preds_test_flat)
    ax5.scatter(targets_test_flat, preds_test_flat, marker='.', alpha=0.5)
    ax5.plot(targets_test_flat, slope_test * targets_test_flat + intercept_test, 'r-', label='Fit')
    ax5.set_title(f"Test Data (Pearson R²: {test_metrics['pearson_r2']:.4f})")
    ax5.set_xlabel("Actual"); ax5.set_ylabel("Predicted")
    ax5.legend(); ax5.grid()
    
    # ----------------- Row 3: Hexbin Density Plots -----------------
    gs3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[3])
    ax6 = plt.subplot(gs3[0])
    hb_train = ax6.hexbin(target_train_flat, preds_train_flat, gridsize=50, cmap='viridis', bins='log')
    ax6.plot(target_train_flat, slope_train * target_train_flat + intercept_train, 'r-', label='Fit')
    ax6.set_title("Train Data Density")
    ax6.set_xlabel("Actual"); ax6.set_ylabel("Predicted")
    ax6.grid(True, which="both", ls="-", alpha=0.2)
    fig.colorbar(hb_train, ax=ax6, label='log10(count)')
    
    ax7 = plt.subplot(gs3[1])
    hb_test = ax7.hexbin(targets_test_flat, preds_test_flat, gridsize=50, cmap='viridis', bins='log')
    ax7.plot(targets_test_flat, slope_test * targets_test_flat + intercept_test, 'r-', label='Fit')
    ax7.set_title("Test Data Density")
    ax7.set_xlabel("Actual"); ax7.set_ylabel("Predicted")
    ax7.grid(True, which="both", ls="-", alpha=0.2)
    fig.colorbar(hb_test, ax=ax7, label='log10(count)')
    
    # ----------------- Row 4: Log1p Scatter Plots -----------------
    gs4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[4])
    log_preds_train = np.log1p(preds_train_flat)
    log_target_train = np.log1p(target_train_flat)
    
    ax8 = plt.subplot(gs4[0])
    ax8.scatter(log_target_train, log_preds_train, marker='.', alpha=0.5)
    ax8.plot(log_target_train, train_metrics['log_slope'] * log_target_train + train_metrics['log_intercept'], 'r-', label='Fit')
    ax8.set_title(f"Log1p Train Data (True R²: {train_metrics['log_r2']:.4f})")
    ax8.set_xlabel("Log1p(Actual)"); ax8.set_ylabel("Log1p(Predicted)")
    ax8.legend(); ax8.grid()
    
    log_preds_test = np.log1p(preds_test_flat)
    log_target_test = np.log1p(targets_test_flat)
    
    ax9 = plt.subplot(gs4[1])
    ax9.scatter(log_target_test, log_preds_test, marker='.', alpha=0.5)
    ax9.plot(log_target_test, test_metrics['log_slope'] * log_target_test + test_metrics['log_intercept'], 'r-', label='Fit')
    ax9.set_title(f"Log1p Test Data (True R²: {test_metrics['log_r2']:.4f})")
    ax9.set_xlabel("Log1p(Actual)"); ax9.set_ylabel("Log1p(Predicted)")
    ax9.legend(); ax9.grid()
    
    # ----------------- Row 5: Non-Zero Analysis -----------------
    gs5 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[5])
    
    non_zero_train_idx = target_train_flat > 1./9
    target_train_nonzero = target_train_flat[non_zero_train_idx]
    preds_train_nonzero = preds_train_flat[non_zero_train_idx]
    
    non_zero_test_idx = targets_test_flat > 1./9
    targets_test_nonzero = targets_test_flat[non_zero_test_idx]
    preds_test_nonzero = preds_test_flat[non_zero_test_idx]
    
    ax10 = plt.subplot(gs5[0])
    ax10.scatter(target_train_nonzero, preds_train_nonzero, marker='.', alpha=0.5)
    ax10.plot(target_train_nonzero, train_metrics['nonzero_slope'] * target_train_nonzero + train_metrics['nonzero_intercept'], 'r-', label='Fit')
    ax10.set_title(f"Train Non-Zero (True R²: {train_metrics['nonzero_r2']:.4f})")
    ax10.set_xlabel("Actual"); ax10.set_ylabel("Predicted")
    ax10.legend(); ax10.grid()
    
    ax11 = plt.subplot(gs5[1])
    ax11.scatter(targets_test_nonzero, preds_test_nonzero, marker='.', alpha=0.5)
    ax11.plot(targets_test_nonzero, test_metrics['nonzero_slope'] * targets_test_nonzero + test_metrics['nonzero_intercept'], 'r-', label='Fit')
    ax11.set_title(f"Test Non-Zero (True R²: {test_metrics['nonzero_r2']:.4f})")
    ax11.set_xlabel("Actual"); ax11.set_ylabel("Predicted")
    ax11.legend(); ax11.grid()
    
    ax12 = plt.subplot(gs5[2])
    metrics_bar = ['Accuracy', 'TPR', 'TNR'] # Renamed to avoid conflict
    train_values = [train_metrics['binary_accuracy'], train_metrics['TPR'], train_metrics['TNR']]
    test_values = [test_metrics['binary_accuracy'], test_metrics['TPR'], test_metrics['TNR']]
    
    x_bar = np.arange(len(metrics_bar)) # Renamed to avoid conflict
    width = 0.35
    
    ax12.bar(x_bar - width/2, train_values, width, label='Train', color='blue')
    ax12.bar(x_bar + width/2, test_values, width, label='Test', color='green')
    
    ax12.set_ylim(0, 1)
    ax12.set_ylabel("Rate")
    ax12.set_title("Classification Metrics")
    ax12.set_xticks(x_bar)
    ax12.set_xticklabels(metrics_bar)
    ax12.legend()
    
    plt.tight_layout()
    if save_loss_fig is not None:
        plt.savefig(save_loss_fig)
    plt.show()
    plt.close()


def train_mosquito_net(model, train_loader, test_loader, device, num_epochs=2500, lr=0.001,
                      dtype=torch.float32, criterion_class=nn.MSELoss,
                      criterion_params={}, optimizer_class=torch.optim.Adam, optimizer_params={},
                      window_size=10, plot_loss=False, save_interval=10, save_loss_fig=None):
    """
    Trains the mosquito net model.

    Behavior changes:
    - If the criterion is ZINB (learned π) -> ensure model has forward_training via wrapper.
    - During train/eval, ZINB losses call forward_training; others use standard forward.
    """
    # ------------- instantiate criterion -------------
    criterion = criterion_class(**criterion_params)

    # ------------- auto-wrap model for ZINB if needed -------------
    zinb_losses = (FrequencyWeightedZINBLoss, HurdleLoss, ZINBLoss)
    is_zinb_loss = isinstance(criterion, zinb_losses)
    
    if is_zinb_loss and not hasattr(model, "forward_training"):
        if isinstance(model, PositiveFeedForwardAbundanceModel):
            model = ZINBPositiveFF(model)
        else:
            raise RuntimeError(
                "Selected loss requires model.forward_training(x). "
                "Provide a model with that method or wrap your base model."
            )

    # ------------- optimizer/scheduler -------------
    # Include learnable loss params (e.g., theta) if any
    learnable_loss_params = [p for p in criterion.parameters() if p.requires_grad] \
                            if hasattr(criterion, "parameters") else []

    if learnable_loss_params:
        params = list(model.parameters()) + learnable_loss_params
    else:
        params = model.parameters()

    optimizer = optimizer_class(params, lr=lr, **optimizer_params)

    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.9,
        total_iters=num_epochs
    )

    # ------------- training bookkeeping -------------
    train_losses, test_losses = [], []
    metrics_history = {
        'train_r2': [], 'test_r2': [], 'train_rmse': [], 'test_rmse': [],
        'train_pearson_r2': [], 'test_pearson_r2': [], 'train_slope': [], 'test_slope': [],
        'train_intercept': [], 'test_intercept': [], 'train_log_r2': [], 'test_log_r2': [],
        'train_log_slope': [], 'test_log_slope': [], 'train_log_intercept': [], 'test_log_intercept': [],
        'train_nonzero_r2': [], 'test_nonzero_r2': [], 'train_nonzero_slope': [], 'test_nonzero_slope': [],
        'train_nonzero_intercept': [], 'test_nonzero_intercept': [], 'train_nonzero_count': [], 'test_nonzero_count': [],
        'train_binary_accuracy': [], 'test_binary_accuracy': [], 'train_FPR': [], 'test_FPR': [],
        'train_FNR': [], 'test_FNR': [], 'train_TPR': [], 'test_TPR': [], 'train_TNR': [], 'test_TNR': []
    }

    def compute_loss(x, y, is_training=True):
        """Compute loss based on criterion type"""
        if isinstance(criterion, HurdleLoss):
            return criterion(predictions=None, targets=y, model=model, x=x)
        elif is_zinb_loss:
            pred = model.forward_training(x)
            return criterion(pred, y)
        else:
            preds = model(x)
            return criterion(preds, y)

    # ------------- epochs -------------
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            loss = compute_loss(x, y, is_training=True)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        current_train_loss_avg = epoch_train_loss / len(train_loader)
        train_losses.append(current_train_loss_avg)

        # ------------- eval -------------
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                loss_val = compute_loss(x, y, is_training=False)
                epoch_test_loss += loss_val.item() if torch.is_tensor(loss_val) else loss_val

        current_test_loss_avg = epoch_test_loss / len(test_loader)
        test_losses.append(current_test_loss_avg)
        scheduler.step()

        # ------------- logging / plots -------------
        if ((epoch + 1) % save_interval == 0 or epoch == num_epochs - 1):
            train_metrics, test_metrics = _compute_and_store_metrics(
                model, train_loader, test_loader, device, metrics_history
            )

            if plot_loss or save_loss_fig is not None:
                _plot_training_progress(
                    epoch, num_epochs, train_losses, test_losses, metrics_history,
                    train_metrics, test_metrics, current_train_loss_avg, current_test_loss_avg,
                    scheduler, window_size, save_loss_fig
                )
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} | "
                      f"Train Loss: {current_train_loss_avg:.4f}, R²: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f} | "
                      f"Test Loss: {current_test_loss_avg:.4f}, R²: {test_metrics['r2']:.4f}, RMSE: {test_metrics['rmse']:.4f}",
                      end="\r")
        if isinstance(criterion, ZINBLoss) and (epoch % 50 == 0):
            with torch.no_grad():
                raw = model.forward_training(x[:1024])  # small batch
                mu  = F.softplus(raw["mu_raw"]) + 1e-8
                pi  = torch.sigmoid(raw["logit_pi"])
                th  = criterion.theta.to(mu.device)
                nb0 = torch.pow(th/(th+mu), th)
                p0_pred = (pi + (1-pi)*nb0).mean().item()
                p0_obs  = (y[:1024]==0).float().mean().item()
                print(f"theta={th.item():.4f}  pi~{pi.mean().item():.3f}  E[Y]={((1-pi)*mu).mean().item():.3f}  zeros obs={p0_obs:.3f} pred≈{p0_pred:.3f}")

    metrics_history['train_losses'] = train_losses
    metrics_history['test_losses'] = test_losses
    return model, metrics_history
