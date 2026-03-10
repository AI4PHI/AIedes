import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from AIedes.data_loader.counter_normalization_stats import normalize_dataframe, create_flagged_weekly_rates


def compute_weekly_stats(data, include_mean=True, include_max=True, include_min=True):
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array.")
    n_samples, n_days = data.shape
    usable_days = (n_days // 7) * 7
    if usable_days < n_days:
        print(f"Using only the first {usable_days} days (ignoring {n_days - usable_days} extra days).")
    truncated = data[:, :usable_days].reshape(n_samples, -1, 7)
    stats = []
    if include_mean:
        stats.append(truncated.mean(axis=2))
    if include_max:
        stats.append(truncated.max(axis=2))
    if include_min:
        stats.append(truncated.min(axis=2))
    return np.stack(stats, axis=2)

def parse_stats_options(variables):
    """
    From specs like:
      ["t2m:mean,max:30",  "tp:30",  "land_cover"]
    produce:
      {
        't2m':  {'stats': ['mean','max'], 'days': 30},
        'tp':   {'stats': [],           'days': 30},
        'land_cover': {'stats': [],     'days': None}
      }
    """
    stats_config = {}
    for spec in variables:
        for token in spec.split():
            parts = token.split(":")
            var = parts[0]
            stats = []
            days = None

            if len(parts) == 2:
                # two-part: could be stats OR days
                if parts[1].isdigit():
                    days = int(parts[1])
                else:
                    stats = parts[1].split(",")
            elif len(parts) == 3:
                # three-part: var:stats:days
                stats = parts[1].split(",") if parts[1] else []
                days = int(parts[2])

            stats_config[var] = {"stats": stats, "days": days}
    return stats_config

def load_and_process_data(params, normalize=True):
    """
    Load and process the climate data from a pickle file, applying specified statistics and normalization.
    Parameters
    ----------
    params : dict
        Dictionary containing:
        - "input_file": Path to the input pickle file containing climate data.
        - "variables": List of variable specifications, e.g. ["t2m:mean,max:30", "tp:30", "land_cover"]
        - "bins": Optional list of bin edges for classification.
    normalize : bool
        Whether to apply normalization to the data.
    Returns
    -------
    climate_tensor : np.ndarray
        Processed climate tensor with shape (n_samples, n_days, n_features).
    targets : np.ndarray
        Target values reshaped to (n_samples, 1).
    class_labels : np.ndarray or None
        Class labels if bins are provided, otherwise None.
    variable_info : list
        List of variable names with applied statistics, e.g. ["t2m(mean,max)", "tp(30)"].
    """

    data_egg = pd.read_pickle(params["input_file"])
    if normalize:
        # Normalize the data using the provided normalization stats
        print("Normalizing data...")
        data_egg = normalize_dataframe(data_egg)
    
    # make pair with flag in prev1_rates and prev2_rates
    data_egg = create_flagged_weekly_rates(data_egg)
    
    stats_config = parse_stats_options(params["variables"])
    climate, variable_info = [], []

    for var, cfg in stats_config.items():
        if var not in data_egg:
            print(f"Warning: Variable {var} not found. Skipping.")
            continue

        # stack into shape (n_samples, n_days)
        var_data = np.vstack(data_egg[var])

        # 1) truncate to last cfg['days'] if requested
        if cfg["days"] is not None:
            d = cfg["days"]
            if var_data.shape[1] >= d:
                var_data = var_data[:, -d:]
            else:
                # pad at front with NaNs so every row is length d
                pad = np.full((var_data.shape[0], d - var_data.shape[1]), np.nan)
                var_data = np.hstack([pad, var_data])

        # 2) compute stats only if cfg["stats"] is non-empty
        stats = cfg["stats"]
        if stats:
            var_data = compute_weekly_stats(
                var_data,
                include_mean="mean" in stats,
                include_max="max"   in stats,
                include_min="min"   in stats
            )
            variable_info.append(f"{var}({','.join(stats)})")
        else:
            variable_info.append(var)

        climate.append(var_data)

    if not climate:
        raise ValueError("No valid variables selected for the climate tensor.")

    climate_tensor = np.concatenate(
        [v.reshape(v.shape[0], -1) for v in climate],
        axis=-1
    )
    if np.isnan(climate_tensor).any():
        # optional: figure out which columns have NaNs
        nan_cols = np.unique(np.where(np.isnan(climate_tensor))[1])
        bad_vars = [variable_info[i] for i in nan_cols]
        raise ValueError(f"NaN detected in climate_tensor for variables: {bad_vars}")

    targets = np.asarray(data_egg["weeklyRates"]).reshape(-1, 1)

    # — compute class labels if bins were passed —
    bins = params.get("bins")
    class_labels = None
    if bins is not None:
        # flatten to 1D, digitize returns 1..len(bins), 0 for < bins[0], len(bins)+1 for ≥ bins[-1]
        flat = targets.ravel()
        class_labels = np.digitize(flat, bins, right=False)

    return climate_tensor, targets, class_labels, variable_info


def prepare_data(climate_tensor, targets, class_labels, params):
    train_climate, test_climate, train_targets, test_targets = train_test_split(
        climate_tensor,
        targets,
        test_size=0.2,
        random_state=params["random_state"],
        stratify=class_labels if class_labels is not None else None
    )
    device = params["device"]  # Get selected device
    train_climate = torch.tensor(train_climate, device=device, dtype=torch.float32)
    test_climate  = torch.tensor(test_climate, device=device, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, device=device, dtype=torch.float32)
    test_targets  = torch.tensor(test_targets, device=device, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(train_climate, train_targets),
                              batch_size=params["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_climate, test_targets),
                             batch_size=params["batch_size"], shuffle=False)
    return train_loader, test_loader, train_climate, test_climate, train_targets, test_targets, device

