"""Closed-loop training utilities for lagged Counter models.

The ordinary Counter fit uses observed egg-rate lags.  This module trains on a
model-generated trajectory instead: after every parameter update the trajectory
is rebuilt, so a prediction made at time *t* is used as the lag input at later
times.  The implementation deliberately keeps the climate features and the
normalisation package identical to the ordinary fit.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _ordered_rows(frame, row_indices):
    """Return rows grouped by trap and ordered by observation date."""
    selected = frame.iloc[np.asarray(row_indices, dtype=int)]
    return sorted(
        (int(index) for index in selected.index),
        key=lambda index: (str(frame.iloc[index].id_trap),
                           pd.Timestamp(frame.iloc[index].end_date), index),
    )


def recursive_rollout(
    model,
    features,
    frame,
    row_indices,
    normalization_stats,
    feature_slices,
    device="cpu",
    detach_feedback=True,
    return_features=False,
    initial_rate=0.0,
):
    """Create differentiable predictions using only earlier predictions.

    ``detach_feedback=True`` is the stable truncated-BPTT variant: generated
    values are still used as inputs, but gradients are not propagated through
    the whole trap history.  Set it to False for full backpropagation through
    the rollout.  Both variants rerun the complete trajectory after every
    optimiser update.
    """
    if not any(name in feature_slices for name in ("p1zn", "p2zn")):
        x = torch.as_tensor(features[np.asarray(row_indices)], dtype=torch.float32,
                            device=device)
        prediction = model(x).reshape(-1)
        if return_features:
            return prediction, np.asarray(row_indices, dtype=int), x
        return prediction, np.asarray(row_indices, dtype=int)

    frame = frame.reset_index(drop=True)
    row_indices = np.asarray(row_indices, dtype=int)
    ordered = _ordered_rows(frame, row_indices)
    end_dates = pd.to_datetime(frame["end_date"]).astype("int64").to_numpy()
    effort = pd.to_numeric(frame["samplingEffort Days"]).to_numpy(dtype=float)
    traps = frame["id_trap"].to_numpy()
    feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    predictions = {}
    # Each trap is processed chronologically, so maintain a small sorted
    # history per trap instead of scanning all selected rows for every lag.
    history_dates = {}
    history_values = {}
    updated_features = {}
    current_features = {}

    # BatchNorm cannot estimate statistics from a one-row recurrent step.  Eval
    # mode still preserves autograd, and uses the same running statistics as
    # deployment.  Dropout is therefore also disabled during the rollout.
    was_training = model.training
    model.eval()
    try:
        grouped = {}
        for index in ordered:
            grouped.setdefault(traps[index], []).append(index)

        def make_current(index):
            current = feature_tensor[index].clone()
            for alias, raw_name, periods in (("p1zn", "prev1_rates", 1),
                                              ("p2zn", "prev2_rates", 2)):
                if alias not in feature_slices:
                    continue
                start, stop = feature_slices[alias]
                width = stop - start
                if width not in (1, 2):
                    raise ValueError(f"Expected one lag value or a value/flag pair for {alias}")
                center = end_dates[index] - int(periods * effort[index] * 86400e9)
                lower, upper = center - int(3 * 86400e9), center + int(3 * 86400e9)
                trap = traps[index]
                dates = history_dates.get(trap, [])
                values = history_values.get(trap, [])
                stats = normalization_stats[raw_name]
                left = bisect.bisect_right(dates, lower)
                right = bisect.bisect_left(dates, upper)
                prior_values = values[left:right]
                if prior_values:
                    value = torch.stack(prior_values).mean()
                    if detach_feedback:
                        value = value.detach()
                    normalized = (value - float(stats["mean"])) / float(stats["std"])
                    current[start] = normalized
                    if width == 2:
                        current[start + 1] = value.new_tensor(1.0)
                else:
                    normalized = (float(initial_rate) - float(stats["mean"])) / float(stats["std"])
                    current[start] = current.new_tensor(normalized)
                    if width == 2:
                        current[start + 1] = current.new_tensor(0.0)
            current_features[index] = current.detach()
            return current

        # One forward pass per chronological position across all traps.  This
        # retains exact per-trap recurrence while avoiding thousands of
        # singleton PyTorch calls on the normal 5k-row dataset.
        for position in range(max(map(len, grouped.values()))):
            active = [rows[position] for rows in grouped.values()
                      if position < len(rows)]
            batch = torch.stack([make_current(index) for index in active])
            batch_predictions = model(batch).reshape(-1)
            for index, prediction in zip(active, batch_predictions):
                predictions[index] = prediction
                updated_features[index] = current_features[index]
                trap = traps[index]
                history_dates.setdefault(trap, []).append(end_dates[index])
                history_values.setdefault(trap, []).append(prediction)
    finally:
        model.train(was_training)

    # Restore the caller's row order, which is important for metric reporting.
    prediction = torch.stack([predictions[int(index)] for index in row_indices])
    if return_features:
        rollout_features = torch.stack([updated_features[int(index)] for index in row_indices])
        return prediction, row_indices, rollout_features
    return prediction, row_indices


def train_counter_autoregressive(
    model,
    features,
    targets,
    frame,
    train_indices,
    normalization_stats,
    feature_slices,
    criterion,
    epochs,
    learning_rate,
    device="cpu",
    scheduler_end_factor=0.9,
    detach_feedback=True,
    clip_norm=10.0,
    callback=None,
    initial_rate=0.0,
):
    """Train a Counter model against its own recursive predictions.

    One optimiser step is performed per epoch over the complete training
    trajectory.  This is intentionally a test-oriented implementation: it
    makes the feedback semantics explicit and reproducible before introducing
    more elaborate batching or multi-step sampling.
    """
    model = model.to(device)
    target = torch.as_tensor(np.asarray(targets).reshape(-1), dtype=torch.float32,
                             device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=scheduler_end_factor,
        total_iters=max(1, int(epochs)),
    )
    history = []
    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, ordered = recursive_rollout(
            model, features, frame, train_indices, normalization_stats,
            feature_slices, device=device, detach_feedback=detach_feedback,
            initial_rate=initial_rate,
        )
        loss = criterion(prediction, target[torch.as_tensor(ordered, device=device)])
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite autoregressive training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        scheduler.step()
        value = float(loss.detach().cpu())
        history.append({"epoch": epoch + 1, "loss": value,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if callback is not None:
            callback(model, epoch + 1, value)

    # Match the main Counter protocol: the fixed-budget final state is the
    # checkpoint. No training/validation/test score selects an earlier epoch.
    return model, history


def train_counter_autoregressive_minibatch(
    model, features, targets, frame, train_indices, normalization_stats,
    feature_slices, criterion, epochs, learning_rate, batch_size=64,
    device="cpu", scheduler_end_factor=0.9, clip_norm=10.0, callback=None,
    initial_rate=0.0,
):
    """Train by regenerating detached recursive inputs once per epoch.

    The rollout is generated with the weights at the beginning of an epoch.
    The generated feature table is then optimized with ordinary minibatches;
    the next epoch regenerates it using the updated weights.  This keeps the
    requested closed-loop training semantics without backpropagating through
    every previous observation in every trap.
    """
    model = model.to(device)
    target = torch.as_tensor(np.asarray(targets).reshape(-1), dtype=torch.float32,
                             device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=scheduler_end_factor,
        total_iters=max(1, int(epochs)),
    )
    history = []
    for epoch in range(int(epochs)):
        model.eval()
        with torch.no_grad():
            rollout_prediction, ordered, rollout_x = recursive_rollout(
                model, features, frame, train_indices, normalization_stats,
                feature_slices, device=device, detach_feedback=True,
                return_features=True,
                initial_rate=initial_rate,
            )
        ordered_tensor = torch.as_tensor(ordered, dtype=torch.long, device=device)
        rollout_y = target[ordered_tensor]
        rollout_loss = float(criterion(rollout_prediction, rollout_y).cpu())

        loader = DataLoader(TensorDataset(rollout_x, rollout_y),
                            batch_size=batch_size, shuffle=True)
        model.train()
        minibatch_loss = 0.0
        for batch_x, batch_y in loader:
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite autoregressive minibatch loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            minibatch_loss += float(loss.detach().cpu())
        scheduler.step()
        history.append({
            "epoch": epoch + 1,
            "rollout_loss": rollout_loss,
            "minibatch_loss": minibatch_loss / max(1, len(loader)),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        if callback is not None:
            callback(model, epoch + 1, rollout_loss)

    # Match the main Counter protocol: use the final fixed-budget state.
    return model, history
