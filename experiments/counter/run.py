import os
import sys
import subprocess
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project folder is in the path (adjust as needed)
sys.path.append(os.path.abspath(os.path.join('../../')))

from AIedes.models.counter_models import (
    PositiveFeedForwardAbundanceModel,
    PositiveLSTMAbundanceModel,
    PositiveMultiHeadAttentionModel,
    PositiveHurdleModel
)
from AIedes.train.counter_train import train_mosquito_net

from AIedes.data_loader.counter_data_loader import (
    load_and_process_data,
    prepare_data,
)

from AIedes.utils.counter_losses import get_criterion

# Resolve key directories relative to this file (stable regardless of cwd)
_COUNTER_DIR = Path(__file__).resolve().parent
_DATA_DIR = _COUNTER_DIR / "data"
_CREATOR_SCRIPT = _COUNTER_DIR / "data" / "src" / "create_eggs_dataset.py"
_DEFAULT_INPUT_FILE = str(_DATA_DIR / "eggs_y_norm.pkl")
_DEFAULT_SOURCE_PICKLE = str(_DATA_DIR / "AIMSurv_albopictus_2020_era5_land.pkl")

# -----------------------
# Data Loading and Preprocessing
# -----------------------
def initialize_model(input_dim, params, device):
    model_type = params.get("model_type", "positive_ff")
    if model_type == "positive_ff":
        model = PositiveFeedForwardAbundanceModel(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            num_hidden_layers=params["num_layers"] - 1,
            dropout=params["dropout"],
            device=device
        )
    elif model_type == "positive_lstm":
        model = PositiveLSTMAbundanceModel(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            device=device
        )
    elif model_type == "positive_mha":
        model = PositiveMultiHeadAttentionModel(
            input_dim=input_dim,
            hidden_dim=params["hidden_dim"],
            output_dim=1,
            num_heads=params.get("num_heads", 4),
            num_layers=params.get("num_layers", 1),
            dropout=params["dropout"],
            device=device
        )
    elif model_type == "hurdle":
        model = PositiveHurdleModel(
        input_dim=input_dim,
        hidden_dim=params["hidden_dim"],
        output_dim=1,
        num_hidden_layers=params["num_layers"] - 1,
        dropout=params["dropout"],
        device=device
    )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    print(f"num of params of the model: {model.count_parameters()}")
    return model.to(device)


def train_model(model, train_loader, test_loader, params, device):
    # Calculate the fraction of zeros (f0) in the target tensor.
    f0 = train_loader.dataset.tensors[1].eq(0).float().mean().item()
    f1 = 1.0 - f0  # Fraction of nonzero entries
    
    epsilon = 1e-8
    params["zero_weight"] = f1
    params["nonzero_weight"] = f0
   
    print(f"Zero weight: {params['zero_weight']}, Non-zero weight: {params['nonzero_weight']}")
        
    loss_type = params.get("loss_type", "MSE")

    # In train_model function, replace $SELECTION_PLACEHOLDER$ with:
    criterion_class, criterion_params = get_criterion(loss_type, params, train_loader, device)

    return train_mosquito_net(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=params["num_epochs"],
        lr=params["lr"],
        dtype=torch.float32,
        criterion_class=criterion_class,
        criterion_params=criterion_params,
        optimizer_class=torch.optim.Adam,
        optimizer_params={},
        plot_loss=params.get("plot_loss", False),
        save_interval=params.get("save_interval", 1),
        save_loss_fig=params.get("save_loss_fig", False)
    )

import os

def file_name_save(params, variable_info):
    """
    Builds a concise, human-readable filename like:
      modelResNet50_lossmse_hid128_lay4_do0.2_lr1e-03_bs32_sd42_temp+soil
    """
    # 1) ensure output folder exists
    folder = params['folder_res']
    os.makedirs(folder, exist_ok=True)

    # 2) define abbreviations for each param
    mapping = {
        'model_type':   'model',
        'loss_type':    'loss',
        'hidden_dim':   'hid',
        'num_layers':   'lay',
        'dropout':      'do',
        'lr':           'lr',
        'batch_size':   'bs',
        'random_state': 'sd',
    }

    parts = []
    for key, abbr in mapping.items():
        v = params[key]
        # compact float formatting
        if isinstance(v, float):
            # scientific if very small, else up to 2 significant digits
            v_str = f"{v:.0e}" if v < 0.01 else f"{v:.2g}"
        else:
            v_str = str(v)
        parts.append(f"{abbr}{v_str}")

    # 3) join your variable_info list with '+'
    var_str = "+".join(variable_info)

    # 4) combine everything
    filename = "_".join(parts + [var_str])
    return filename

def save_results(trained_model, metrics_history, params, variable_info,
                 train_data, test_data, train_targets, test_targets):
    folder_res = params['folder_res']
    filename_base = file_name_save(params, variable_info)
    
    # Save the model
    model_path = os.path.join(folder_res, f"{filename_base}.pth")
    torch.save(trained_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Build the results DataFrame. For each performance metric, the full history is stored
    # with a column name with the suffix '_history' and the final value is stored with
    # the plain metric name.
    results = pd.DataFrame({
        "model_type": [params["model_type"]],
        "loss_type": [params["loss_type"]],
        "hidden_dim": [params["hidden_dim"]],
        "num_layers": [params["num_layers"]],
        "dropout": [params["dropout"]],
        "num_epochs": [params["num_epochs"]],
        "learning_rate": [params["lr"]],
        "batch_size": [params["batch_size"]],
        "random_state": [params["random_state"]],
        "nonzero_weight": [params.get("nonzero_weight", np.nan)],
        "num_heads": [params.get("num_heads", np.nan)],
        "command_line_params": [str(params)],
        "num_params": [trained_model.count_parameters()],
        "input_variables": ["; ".join(variable_info)],
        # Save full history arrays with the _history suffix
        "train_loss_history": [metrics_history["train_losses"]],
        "test_loss_history": [metrics_history["test_losses"]],
        "train_r2_history": [metrics_history["train_r2"]],
        "test_r2_history": [metrics_history["test_r2"]],
        "train_rmse_history": [metrics_history["train_rmse"]],
        "test_rmse_history": [metrics_history["test_rmse"]],
        "train_pearson_r2_history": [metrics_history["train_pearson_r2"]],
        "test_pearson_r2_history": [metrics_history["test_pearson_r2"]],
        "train_slope_history": [metrics_history["train_slope"]],
        "test_slope_history": [metrics_history["test_slope"]],
        "train_intercept_history": [metrics_history["train_intercept"]],
        "test_intercept_history": [metrics_history["test_intercept"]],
        "train_log_r2_history": [metrics_history["train_log_r2"]],
        "test_log_r2_history": [metrics_history["test_log_r2"]],
        "train_log_slope_history": [metrics_history["train_log_slope"]],
        "test_log_slope_history": [metrics_history["test_log_slope"]],
        "train_log_intercept_history": [metrics_history["train_log_intercept"]],
        "test_log_intercept_history": [metrics_history["test_log_intercept"]],
        "train_nonzero_r2_history": [metrics_history["train_nonzero_r2"]],
        "test_nonzero_r2_history": [metrics_history["test_nonzero_r2"]],
        "train_nonzero_slope_history": [metrics_history["train_nonzero_slope"]],
        "test_nonzero_slope_history": [metrics_history["test_nonzero_slope"]],
        "train_nonzero_intercept_history": [metrics_history["train_nonzero_intercept"]],
        "test_nonzero_intercept_history": [metrics_history["test_nonzero_intercept"]],
        "train_binary_accuracy_history": [metrics_history["train_binary_accuracy"]],
        "test_binary_accuracy_history": [metrics_history["test_binary_accuracy"]],
        "train_FPR_history": [metrics_history["train_FPR"]],
        "test_FPR_history": [metrics_history["test_FPR"]],
        "train_FNR_history": [metrics_history["train_FNR"]],
        "test_FNR_history": [metrics_history["test_FNR"]],
        "train_TPR_history": [metrics_history["train_TPR"]],
        "test_TPR_history": [metrics_history["test_TPR"]],
        "train_TNR_history": [metrics_history["train_TNR"]],
        "test_TNR_history": [metrics_history["test_TNR"]],
        # Save final results (last epoch) without the suffix
        "train_loss": [metrics_history["train_losses"][-1]],
        "test_loss": [metrics_history["test_losses"][-1]],
        "train_r2": [metrics_history["train_r2"][-1]],
        "test_r2": [metrics_history["test_r2"][-1]],
        "train_rmse": [metrics_history["train_rmse"][-1]],
        "test_rmse": [metrics_history["test_rmse"][-1]],
        "train_pearson_r2": [metrics_history["train_pearson_r2"][-1]],
        "test_pearson_r2": [metrics_history["test_pearson_r2"][-1]],
        "train_slope": [metrics_history["train_slope"][-1]],
        "test_slope": [metrics_history["test_slope"][-1]],
        "train_intercept": [metrics_history["train_intercept"][-1]],
        "test_intercept": [metrics_history["test_intercept"][-1]],
        "train_log_r2": [metrics_history["train_log_r2"][-1]],
        "test_log_r2": [metrics_history["test_log_r2"][-1]],
        "train_log_slope": [metrics_history["train_log_slope"][-1]],
        "test_log_slope": [metrics_history["test_log_slope"][-1]],
        "train_log_intercept": [metrics_history["train_log_intercept"][-1]],
        "test_log_intercept": [metrics_history["test_log_intercept"][-1]],
        "train_nonzero_r2": [metrics_history["train_nonzero_r2"][-1]],
        "test_nonzero_r2": [metrics_history["test_nonzero_r2"][-1]],
        "train_nonzero_slope": [metrics_history["train_nonzero_slope"][-1]],
        "test_nonzero_slope": [metrics_history["test_nonzero_slope"][-1]],
        "train_nonzero_intercept": [metrics_history["train_nonzero_intercept"][-1]],
        "test_nonzero_intercept": [metrics_history["test_nonzero_intercept"][-1]],
        "train_binary_accuracy": [metrics_history["train_binary_accuracy"][-1]],
        "test_binary_accuracy": [metrics_history["test_binary_accuracy"][-1]],
        "train_FPR": [metrics_history["train_FPR"][-1]],
        "test_FPR": [metrics_history["test_FPR"][-1]],
        "train_FNR": [metrics_history["train_FNR"][-1]],
        "test_FNR": [metrics_history["test_FNR"][-1]],
        "train_TPR": [metrics_history["train_TPR"][-1]],
        "test_TPR": [metrics_history["test_TPR"][-1]],
        "train_TNR": [metrics_history["train_TNR"][-1]],
        "test_TNR": [metrics_history["test_TNR"][-1]],
        "train_data": [train_data.cpu().numpy()],
        "test_data": [test_data.cpu().numpy()],
        "train_targets": [train_targets.cpu().numpy()],
        "test_targets": [test_targets.cpu().numpy()]
    })
    
    results_path = os.path.join(folder_res, f"{filename_base}_results.pkl")
    results.to_pickle(results_path)
    print(f"Results saved to {results_path}")

def run_training(params):
    climate_tensor, targets, class_labels, variable_info = load_and_process_data(params)
    (train_loader, test_loader, train_climate, test_climate,
     train_targets, test_targets, device) = prepare_data(
         climate_tensor, targets, class_labels, params
    )
    model = initialize_model(input_dim=climate_tensor.shape[1], params=params, device=device)
    filename = file_name_save(params, variable_info)
    if params["save_loss_fig"] == "yes":
        params["save_loss_fig"] = os.path.join(params["folder_res"], f"{filename}_loss.png")
    else:
        params["save_loss_fig"] = None
    
    # Now train_model returns a dictionary with all the history
    trained_model, metrics_history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        params=params,
        device=device
    )
    
    save_results(trained_model, metrics_history, params, variable_info,
                 train_climate, test_climate, train_targets, test_targets)
    return (trained_model, metrics_history, train_climate, test_climate, train_targets, test_targets)

def ensure_eggs_dataset_exists(params):
    """
    Check if the dataset file exists. If not, run create_eggs_dataset.py
    automatically to generate it, then verify the output was created.
    """
    input_file = Path(params["input_file"])

    # Resolve relative paths relative to counter_dir
    if not input_file.is_absolute():
        input_file = _COUNTER_DIR / input_file

    input_file = input_file.resolve()
    params["input_file"] = str(input_file)  # normalise back

    if input_file.exists():
        print(f"Dataset found: {input_file}")
        return

    print(f"Dataset not found: {input_file}")
    print("Attempting to generate dataset automatically...")

    source_pickle = params.get("dataset_source_pickle", _DEFAULT_SOURCE_PICKLE)
    source_pickle = Path(source_pickle)
    if not source_pickle.is_absolute():
        source_pickle = _COUNTER_DIR / source_pickle
    source_pickle = source_pickle.resolve()

    if not source_pickle.exists():
        raise FileNotFoundError(
            f"Source pickle not found: {source_pickle}\n"
            "Please provide the climate-enriched source file via --dataset_source_pickle."
        )

    _DATA_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(_CREATOR_SCRIPT),
        "--input", str(source_pickle),
        "--output-dir", str(_DATA_DIR),
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(_COUNTER_DIR))

    if not input_file.exists():
        raise RuntimeError(
            f"Dataset generation completed but expected file not found: {input_file}"
        )
    print(f"Dataset generated successfully: {input_file}")

# -----------------------
# Command-Line Execution
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train mosquito abundance prediction model.")
    parser.add_argument("--model_type", type=str, default="hurdle", choices=["positive_ff", "positive_lstm", "positive_mha", "hurdle"])
    parser.add_argument("--hidden_dim", type=int, default=30)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--variables", nargs="+", default=["t2m:mean,max", "tp:mean,max"], help="Each spec `var:stat1,stat2:days` (days optional)")
    parser.add_argument("--loss_type", type=str, default="Hurdle", 
                    choices=["MSE", "Huber", "WeightedMSE", "WeightedMSLE", "NegativeBinomial", "Hurdle", "FrequencyWeightedMSE", "FrequencyWeightedMSLE", "FrequencyWeightedNegativeBinomial", "ZINBLoss", "FrequencyWeightedZeroInflatedNegativeBinomial"])
    #parser.add_argument("--nonzero_weight", type=float, default=3.0)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--folder_res", type=str, default="./results/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--input_file", type=str, default=_DEFAULT_INPUT_FILE)
    parser.add_argument(
        "--dataset_source_pickle",
        type=str,
        default=_DEFAULT_SOURCE_PICKLE,
        help="Path to the climate-enriched source pickle used to generate the dataset if missing.",
    )
    parser.add_argument("--save_loss_fig", type=str, default="yes", choices=["yes", "no"])
    parser.add_argument("--plot_loss", type=bool, default=False)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument(
        "--bins",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional list of bin edges for target classes. "
            "E.g. `--bins 0 10 50 200` creates 5 classes: "
            "<0, [0–10), [10–50), [50–200), ≥200."
        )
    )
    args = parser.parse_args()
    params = vars(args)

    torch.set_num_threads(params["num_threads"])
    torch.set_num_interop_threads(params["num_threads"])
    
    print(args)
    ensure_eggs_dataset_exists(params)
    run_training(params)
