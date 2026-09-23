import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import linregress
from AIedes.data_loader.counter_data_loader import (
    counter_feature_names,
    fit_counter_preprocessing,
    make_counter_loader,
    transform_counter_features,
)
from AIedes.evaluation.counter_eval import compute_metrics
from AIedes.utils.counter_losses import (
    FrequencyWeightedLoss,
    FrequencyWeightedZINBLoss,
    HurdleLoss,
    ZINBLoss,
    compute_bin_weights,
)
from AIedes.models.counter_models import (
    PositiveFeedForwardAbundanceModel,
    ZINBPositiveFF,
    counter_candidate_model,
    predict_counter_rates,
)

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
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from IPython.display import clear_output

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


def _strip_tensor_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "targets", "predictions_np", "targets_np"}
    }


def _compute_counter_loss(model, criterion, x, y, is_zinb_loss):
    if isinstance(criterion, HurdleLoss):
        return criterion(predictions=None, targets=y, model=model, x=x)
    if is_zinb_loss:
        return criterion(model.forward_training(x), y)
    return criterion(model(x), y)


def _average_counter_loss(model, criterion, loader, device, is_zinb_loss):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += _compute_counter_loss(model, criterion, x, y, is_zinb_loss).item()
    return total_loss / len(loader)


def _average_log_error(model, loader, device):
    """Observation-pooled unweighted log1p error for optional checkpoint selection."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            prediction = model(x.to(device)).view(-1).double()
            target = y.to(device).view(-1).double()
            total += ((torch.log1p(prediction) - torch.log1p(target)) ** 2).sum().item()
            count += target.numel()
    if not count:
        raise ValueError("Empty validation partition")
    return total / count


def _save_validation_loss_curve(train_losses, val_losses, best_epoch, output_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="validation")
    plt.axvline(best_epoch - 1, color="black", linestyle="--", label="best validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_mosquito_net_with_validation(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=2500,
    lr=0.001,
    dtype=torch.float32,
    criterion_class=nn.MSELoss,
    criterion_params=None,
    optimizer_class=torch.optim.Adam,
    optimizer_params=None,
    save_interval=10,
    save_loss_fig=None,
    early_stopping=True,
    early_stopping_patience=200,
    early_stopping_min_delta=0.0,
    selection_metric="training_loss",
    scheduler_horizon=None,
    scheduler_end_factor=0.9,
    record_metrics=True,
):
    """
    Train with validation-based early stopping.

    The validation split is used for checkpoint selection. A separate held-out
    test split should be evaluated by the caller after this function restores
    the best validation checkpoint.
    """
    if selection_metric not in ("training_loss", "log_mse"):
        raise ValueError("Unknown checkpoint selection metric")
    criterion_params = criterion_params or {}
    optimizer_params = optimizer_params or {}
    criterion = criterion_class(**criterion_params)

    zinb_losses = (FrequencyWeightedZINBLoss, HurdleLoss, ZINBLoss)
    is_zinb_loss = isinstance(criterion, zinb_losses)
    if is_zinb_loss and not hasattr(model, "forward_training"):
        if isinstance(model, PositiveFeedForwardAbundanceModel):
            model = ZINBPositiveFF(model).to(device)
        else:
            raise RuntimeError(
                "Selected loss requires model.forward_training(x). "
                "Provide a model with that method or wrap your base model."
            )

    learnable_loss_params = [p for p in criterion.parameters() if p.requires_grad] \
        if hasattr(criterion, "parameters") else []
    trainable_params = list(model.parameters()) + learnable_loss_params
    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_params)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=scheduler_end_factor,
        total_iters=scheduler_horizon or num_epochs,
    )

    metrics_history = {
        "train_losses": [],
        "val_losses": [],
        "epochs_recorded": [],
        "train_metrics": [],
        "val_metrics": [],
        "best_epoch": 0,
        "best_val_loss": float("inf"),
        "epochs_trained": 0,
        "epoch_records": [],
    }
    best = {
        "epoch": 0,
        "step": 0,
        "val_loss": float("inf"),
        "model_state": copy.deepcopy(model.state_dict()),
        "criterion_state": copy.deepcopy(criterion.state_dict()),
    }
    epochs_without_improvement = 0
    optimizer_steps = 0

    for epoch in range(num_epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = _compute_counter_loss(model, criterion, x, y, is_zinb_loss)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer_steps += 1
            epoch_train_loss += loss.item()

        train_loss = epoch_train_loss / len(train_loader)
        val_loss = (_average_log_error(model, val_loader, device) if selection_metric == "log_mse"
                    else _average_counter_loss(model, criterion, val_loader, device, is_zinb_loss))
        metrics_history["train_losses"].append(train_loss)
        metrics_history["val_losses"].append(val_loss)
        scheduler.step()
        metrics_history["epoch_records"].append(dict(epoch=epoch+1, optimizer_steps=optimizer_steps,
            weighted_train_loss=train_loss, validation_log_mse=val_loss if selection_metric == "log_mse" else None,
            learning_rate=epoch_lr))

        improved = val_loss < (best["val_loss"] - early_stopping_min_delta)
        if improved:
            best = {
                "epoch": epoch + 1,
                "step": epoch + 1,
                "val_loss": val_loss,
                "model_state": copy.deepcopy(model.state_dict()),
                "criterion_state": copy.deepcopy(criterion.state_dict()),
                "optimizer_step": optimizer_steps,
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        should_record = ((epoch + 1) % save_interval == 0) or (epoch == 0)
        if should_record and record_metrics:
            train_metrics = compute_metrics(model, train_loader, device)
            val_metrics = compute_metrics(model, val_loader, device)
            metrics_history["epochs_recorded"].append(epoch + 1)
            metrics_history["train_metrics"].append(_strip_tensor_metrics(train_metrics))
            metrics_history["val_metrics"].append(_strip_tensor_metrics(val_metrics))
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, R2: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f} | "
                f"Val Loss: {val_loss:.4f}, R2: {val_metrics['r2']:.4f}, RMSE: {val_metrics['rmse']:.4f} | "
                f"Best epoch: {best['epoch']}"
            )

        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            print(
                "Early stopping at epoch "
                f"{epoch + 1}; best validation epoch was {best['epoch']}."
            )
            break

    model.load_state_dict(best["model_state"])
    criterion.load_state_dict(best["criterion_state"])
    metrics_history["best_epoch"] = best["epoch"]
    metrics_history["best_step"] = best["step"]
    metrics_history["best_val_loss"] = best["val_loss"]
    metrics_history["epochs_trained"] = len(metrics_history["train_losses"])
    metrics_history["optimizer_steps_trained"] = optimizer_steps
    metrics_history["best_optimizer_step"] = best.get("optimizer_step", 0)

    if save_loss_fig is not None:
        _save_validation_loss_curve(
            metrics_history["train_losses"],
            metrics_history["val_losses"],
            metrics_history["best_epoch"],
            save_loss_fig,
        )

    return model, criterion, metrics_history


def train_mosquito_net_fixed_epochs(
    model,
    train_loader,
    device,
    num_epochs,
    lr=0.001,
    criterion_class=nn.MSELoss,
    criterion_params=None,
    optimizer_class=torch.optim.Adam,
    optimizer_params=None,
    save_interval=10,
    scheduler_horizon=None,
    scheduler_end_factor=0.9,
    record_metrics=True,
    epoch_callback=None,
):
    """Refit on all outer-development data for an already selected epoch count."""
    criterion_params = criterion_params or {}
    optimizer_params = optimizer_params or {}
    criterion = criterion_class(**criterion_params)
    zinb_losses = (FrequencyWeightedZINBLoss, HurdleLoss, ZINBLoss)
    is_zinb_loss = isinstance(criterion, zinb_losses)
    if is_zinb_loss and not hasattr(model, "forward_training"):
        if isinstance(model, PositiveFeedForwardAbundanceModel):
            model = ZINBPositiveFF(model).to(device)
        else:
            raise RuntimeError("Selected loss requires model.forward_training(x).")

    loss_params = [p for p in criterion.parameters() if p.requires_grad] \
        if hasattr(criterion, "parameters") else []
    optimizer = optimizer_class(list(model.parameters()) + loss_params, lr=lr, **optimizer_params)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=scheduler_end_factor,
        total_iters=scheduler_horizon or num_epochs
    )
    history = {
        "train_losses": [], "val_losses": [], "epochs_recorded": [],
        "train_metrics": [], "val_metrics": [], "best_epoch": num_epochs,
        "best_step": num_epochs, "best_val_loss": np.nan, "epochs_trained": num_epochs,
        "epoch_records": [],
    }
    optimizer_steps = 0
    for epoch in range(num_epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = _compute_counter_loss(model, criterion, x, y, is_zinb_loss)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer_steps += 1
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        history["train_losses"].append(train_loss)
        scheduler.step()
        if epoch_callback is not None:
            epoch_callback(model, epoch + 1)
        history["epoch_records"].append(dict(epoch=epoch+1, optimizer_steps=optimizer_steps,
            weighted_train_loss=train_loss, validation_log_mse=None, learning_rate=epoch_lr))
        if record_metrics and ((epoch + 1) % save_interval == 0 or epoch in (0, num_epochs - 1)):
            train_metrics = compute_metrics(model, train_loader, device)
            history["epochs_recorded"].append(epoch + 1)
            history["train_metrics"].append(_strip_tensor_metrics(train_metrics))
            print(
                f"Refit epoch {epoch + 1}/{num_epochs} | Train loss: {train_loss:.4f}, "
                f"R2: {train_metrics['r2']:.4f}, RMSE: {train_metrics['rmse']:.4f}"
            )
    history["optimizer_steps_trained"] = optimizer_steps
    history["best_optimizer_step"] = optimizer_steps
    return model, criterion, history


def check_counter_batches(n_rows, batch_size):
    """BatchNorm cannot train on a single row, so refuse singleton final batches."""
    if n_rows < 2:
        raise ValueError("Training requires at least two rows")
    if batch_size < 2:
        raise ValueError("Batch size must exceed one row")
    if n_rows > batch_size and n_rows % batch_size == 1:
        raise ValueError(
            f"{n_rows} rows at batch size {batch_size} leave a singleton final batch"
        )


def fit_counter_candidate(frame, train_indices, validation_indices, candidate, config, seed, epochs=None,
                          epoch_callback=None):
    """Fit one revision candidate with frequency-weighted WMSLE on given rows.

    Normalization and loss weights are fitted on the training rows only. With
    validation rows the checkpoint is selected by unweighted validation log1p
    error; without them the model is refitted for a fixed epoch count while
    keeping the screening learning-rate horizon.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    statistics = fit_counter_preprocessing(frame, train_indices, candidate)
    features = transform_counter_features(frame, candidate, statistics)
    targets = frame["weeklyRates"].to_numpy(dtype=np.float32).reshape(-1, 1)
    loader_params = {"device": "cpu", "batch_size": config["batch_size"], "random_state": seed}
    check_counter_batches(len(train_indices), config["batch_size"])
    train_loader, _, _ = make_counter_loader(features, targets, train_indices, loader_params, shuffle=True)

    weights = compute_bin_weights(targets[train_indices].ravel(), config["weight_bins"], empty_bins_zero=True)
    shared = {
        "lr": candidate.get("learning_rate", config["learning_rate"]),
        "criterion_class": FrequencyWeightedLoss,
        "criterion_params": {"bins": config["weight_bins"], "weights": weights.tolist(),
                             "device": "cpu", "log_loss": candidate.get("loss", "WMSLE") != "WMSE", "right": True},
        "save_interval": config["log_interval"],
        "scheduler_horizon": config["scheduler_horizon"],
        "scheduler_end_factor": config["scheduler_end_factor"],
        "record_metrics": False,
    }
    if candidate.get("loss") == "ZINB":
        shared.update(criterion_class=ZINBLoss, criterion_params={"zero_threshold": 0.0})
    model = counter_candidate_model(candidate, features.shape[1])

    if validation_indices is None:
        if not epochs:
            raise ValueError("Refits require a selected epoch count")
        model, criterion, history = train_mosquito_net_fixed_epochs(
            model, train_loader, "cpu", num_epochs=int(epochs), epoch_callback=epoch_callback, **shared
        )
        predictions = None
    else:
        validation_loader, _, _ = make_counter_loader(
            features, targets, validation_indices, loader_params, shuffle=False
        )
        model, criterion, history = train_mosquito_net_with_validation(
            model, train_loader, validation_loader, "cpu",
            num_epochs=int(epochs or config["max_epochs"]),
            early_stopping=True,
            early_stopping_patience=config["patience"],
            early_stopping_min_delta=config["min_delta"],
            selection_metric="log_mse",
            **shared,
        )
        predictions = predict_counter_rates(model, features[validation_indices])

    package = {
        "schema_version": 1,
        "candidate": candidate,
        "stats": statistics,
        "feature_names": counter_feature_names(candidate),
        "state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "input_dim": int(features.shape[1]),
        "seed": seed,
        "best_epoch": history["best_epoch"],
        "epochs_trained": history["epochs_trained"],
        "optimizer_steps_trained": history["optimizer_steps_trained"],
        "selection_metric": "unweighted_validation_log1p_mse" if validation_indices is not None else "fixed_epochs",
        "weight_bins": config["weight_bins"],
        "loss_weights": weights.tolist(),
        "normalization_scope": "training_rows_and_requested_window_only",
        "autonomous_lag_days": config["autonomous_lag_days"],
        "climate_endpoint": config["climate_endpoint"],
        "startup": config["autonomous_startup"],
        "target_units": "eggs_per_week",
        "climate_alignment_evidence": config["climate_alignment_evidence"],
    }
    return package, history["epoch_records"], predictions
