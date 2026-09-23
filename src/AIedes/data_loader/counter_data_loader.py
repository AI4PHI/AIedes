import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from AIedes.data_loader.counter_normalization_stats import (
    create_flagged_weekly_rates,
    fit_normalization_stats,
    normalize_dataframe,
)


def compute_weekly_stats(data, include_mean=True, include_max=True, include_min=True,
                         keep_partial_week=False, recent_weeks=False):
    """Weekly aggregation of daily windows.

    keep_partial_week=True keeps every day by letting the oldest block be short,
    so a 30/60/90-day window remains exactly that long. Otherwise the window is
    truncated to whole weeks: the legacy default keeps the oldest weeks
    (reproducing previous counter experiments), and recent_weeks=True keeps the
    most recent ones.
    """
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D numpy array.")
    n_samples, n_days = data.shape
    if keep_partial_week:
        ends = list(range(n_days, 0, -7))[::-1]
        blocks = [data[:, a:b] for a, b in zip([0] + ends[:-1], ends)]
        operations = [(include_mean, np.mean), (include_max, np.max), (include_min, np.min)]
        return np.stack([np.stack([fn(block, axis=1) for block in blocks], axis=1)
                         for include, fn in operations if include], axis=2)
    usable_days = (n_days // 7) * 7
    if usable_days < n_days:
        ignored = "oldest" if recent_weeks else "extra"
        print(f"Using {usable_days} days (ignoring {n_days - usable_days} {ignored} days).")
    window = data[:, -usable_days:] if recent_weeks else data[:, :usable_days]
    truncated = window.reshape(n_samples, -1, 7)
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

def load_and_process_data(params, normalize=True, normalization_indices=None, data_frame=None,
                          saved_normalization=None, require_targets=True):
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

    data_egg = pd.read_pickle(params["input_file"]) if data_frame is None else data_frame.copy()
    verbose = params.get("verbose", True)
    if normalization_indices is not None and saved_normalization is not None:
        raise ValueError("Choose training-fitted or saved normalization, not both")
    if normalize:
        if verbose:
            print("Normalizing data...")
        requested = parse_stats_options(params["variables"])
        normalized = [name for name in requested if name.endswith("_znorm")]
        raw_fields = [name.removesuffix("_znorm") for name in normalized]
        # p1zn/p2zn are the flagged aliases produced below from these raw columns.
        for alias, raw in (("p1zn", "prev1_rates"), ("p2zn", "prev2_rates")):
            if alias in requested:
                raw_fields.append(raw)
        raw_fields = list(dict.fromkeys(raw_fields))
        if normalization_indices is not None or saved_normalization is not None:
            windows = {name.removesuffix("_znorm"): spec["days"] for name, spec in requested.items()
                       if name.endswith("_znorm")} if params.get("normalize_selected_window") else None
            fitted = saved_normalization if saved_normalization is not None else fit_normalization_stats(
                data_egg, normalization_indices, raw_fields, field_windows=windows,
                allow_constant=params.get("allow_constant_normalization", False))
            data_egg = normalize_dataframe(
                data_egg, normalization_stats=fitted,
                fields_to_normalize=raw_fields, verbose=verbose,
            )
            params["normalization_scope"] = "saved_training_stats" if saved_normalization is not None else "training_rows_only"
            params["normalization_stats"] = fitted
        else:
            data_egg = normalize_dataframe(data_egg)
            params["normalization_scope"] = "legacy_global_constants"
    
    # make pair with flag in prev1_rates and prev2_rates
    data_egg = create_flagged_weekly_rates(data_egg)
    
    stats_config = parse_stats_options(params["variables"])
    climate, variable_info = [], []
    feature_slices = {}
    feature_cursor = 0

    for var, cfg in stats_config.items():
        if var not in data_egg:
            if params.get("strict_features"):
                raise ValueError(f"Required variable missing: {var}")
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
                include_min="min"   in stats,
                keep_partial_week=params.get("keep_partial_week", False),
                recent_weeks=params.get("recent_weeks", False),
            )
            variable_info.append(f"{var}({','.join(stats)})")
        else:
            variable_info.append(var)

        flattened = var_data.reshape(var_data.shape[0], -1)
        feature_slices[var] = [feature_cursor, feature_cursor + flattened.shape[1]]
        feature_cursor += flattened.shape[1]
        climate.append(flattened)

    if not climate:
        raise ValueError("No valid variables selected for the climate tensor.")

    climate_tensor = np.concatenate(
        climate,
        axis=-1
    )
    params["feature_slices"] = feature_slices
    if not np.isfinite(climate_tensor).all():
        bad_vars = [name for name, (a, b) in feature_slices.items()
                    if not np.isfinite(climate_tensor[:, a:b]).all()]
        raise ValueError(f"NaN detected in climate_tensor for variables: {bad_vars}")

    targets = np.asarray(data_egg["weeklyRates"]).reshape(-1, 1) if "weeklyRates" in data_egg else None
    if targets is None and require_targets:
        raise ValueError("Targets required for training/evaluation")

    # — compute class labels if bins were passed —
    bins = params.get("bins")
    class_labels = None
    if bins is not None and targets is not None:
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
    generator = torch.Generator()
    generator.manual_seed(params["random_state"])
    train_loader = DataLoader(TensorDataset(train_climate, train_targets),
                              batch_size=params["batch_size"], shuffle=True,
                              generator=generator)
    test_loader = DataLoader(TensorDataset(test_climate, test_targets),
                             batch_size=params["batch_size"], shuffle=False)
    return train_loader, test_loader, train_climate, test_climate, train_targets, test_targets, device


def can_stratify(labels):
    if labels is None:
        return False
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return False
    _, counts = np.unique(labels, return_counts=True)
    return np.all(counts >= 2)


def split_train_val_test_indices(n_samples, class_labels, params):
    test_size = params.get("test_size", 0.2)
    val_size = params.get("val_size", 0.2)
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be > 0 and sum to < 1.")

    indices = np.arange(n_samples)
    stratify = class_labels if can_stratify(class_labels) else None

    try:
        train_val_idx, test_idx, train_val_labels, _ = train_test_split(
            indices,
            class_labels if class_labels is not None else indices,
            test_size=test_size,
            random_state=params["random_state"],
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"Warning: stratified test split failed ({exc}); using unstratified split.")
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=params["random_state"],
            stratify=None,
        )
        train_val_labels = class_labels[train_val_idx] if class_labels is not None else None

    relative_val_size = val_size / (1.0 - test_size)
    val_stratify = train_val_labels if can_stratify(train_val_labels) else None

    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=params["random_state"],
            stratify=val_stratify,
        )
    except ValueError as exc:
        print(f"Warning: stratified validation split failed ({exc}); using unstratified split.")
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=params["random_state"],
            stratify=None,
        )

    return {
        "train": np.sort(train_idx),
        "val": np.sort(val_idx),
        "test": np.sort(test_idx),
    }


def make_counter_loader(features, targets, indices, params, shuffle):
    device = params["device"]
    data = torch.tensor(features[indices], device=device, dtype=torch.float32)
    target = torch.tensor(targets[indices], device=device, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(params["random_state"])
    loader = DataLoader(
        TensorDataset(data, target),
        batch_size=params["batch_size"],
        shuffle=shuffle,
        generator=generator if shuffle or params.get("independent_loader_rng", False) else None,
    )
    return loader, data, target


def prepare_train_val_test_data(climate_tensor, targets, class_labels, params, indices=None):
    if indices is None:
        if params.get("split_type", "random") != "random":
            raise ValueError("Grouped training requires explicit frozen split indices.")
        indices = split_train_val_test_indices(len(targets), class_labels, params)
    combined = np.concatenate([indices[key] for key in ("train", "val", "test")])
    if (any(len(indices[key]) == 0 for key in ("train", "val", "test"))
            or not np.array_equal(np.sort(combined), np.arange(len(targets)))):
        raise ValueError("Split indices must be nonempty, disjoint and cover every data row once.")
    train_loader, train_data, train_targets = make_counter_loader(
        climate_tensor, targets, indices["train"], params, shuffle=True
    )
    val_loader, val_data, val_targets = make_counter_loader(
        climate_tensor, targets, indices["val"], params, shuffle=False
    )
    test_loader, test_data, test_targets = make_counter_loader(
        climate_tensor, targets, indices["test"], params, shuffle=False
    )
    print(
        "Split sizes: "
        f"train={len(indices['train'])}, val={len(indices['val'])}, test={len(indices['test'])}"
    )
    return {
        "indices": indices,
        "loaders": {"train": train_loader, "val": val_loader, "test": test_loader},
        "tensors": {
            "train_data": train_data,
            "val_data": val_data,
            "test_data": test_data,
            "train_targets": train_targets,
            "val_targets": val_targets,
            "test_targets": test_targets,
        },
        "device": params["device"],
    }


def counter_feature_options(candidate):
    """Translate a revision candidate into the existing variable-spec interface."""
    variables = [f"{field}_znorm:{'mean:' if candidate['weekly'] else ''}{candidate['window']}"
                 for field in candidate['fields']]
    if candidate['lags']:
        variables += [f'p{k}zn' for k in range(1, candidate.get('lag_order', 2) + 1)]
    return dict(variables=variables, verbose=False, strict_features=True,
                keep_partial_week=True, normalize_selected_window=True,
                allow_constant_normalization=True)


def fit_counter_preprocessing(frame, indices, candidate):
    fields = candidate['fields'] + ([f'prev{k}_rates' for k in range(1, candidate.get('lag_order', 2) + 1)] if candidate['lags'] else [])
    windows = {field: candidate['window'] for field in candidate['fields']}
    return fit_normalization_stats(frame, indices, fields, field_windows=windows, allow_constant=True)


def transform_counter_features(frame, candidate, statistics):
    values, _, _, _ = load_and_process_data(counter_feature_options(candidate),
        data_frame=frame, saved_normalization=statistics, require_targets=False)
    return values.astype(np.float32)


def counter_feature_names(candidate):
    names = []
    for field in candidate["fields"]:
        n = candidate["window"]
        if candidate["weekly"]:
            ends = list(range(n, 0, -7))[::-1]
            for a, b in zip([0] + ends[:-1], ends):
                names.append(f"{field}:mean:offsets{a-n+1}..{b-n}")
        else:
            names += [f"{field}:offset{d}" for d in range(-n+1, 1)]
    if candidate["lags"]:
        for k in range(1, candidate.get('lag_order', 2) + 1):
            names += [f"p{k}:value", f"p{k}:available"]
    return names
