"""Parallel five-fold runner for value-only autoregressive Counter fits.

The recurrent time axis of one trap is sequential by definition.  The safe
parallel boundary is therefore the outer fold (and, later, independent seeds):
each worker owns one complete model and uses a bounded number of Torch threads.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from AIedes.data_loader.counter_data_loader import (
    counter_feature_names,
    fit_counter_preprocessing,
    transform_counter_features,
)
from AIedes.evaluation.counter_eval import compute_metrics_from_arrays
from AIedes.models.counter_models import counter_candidate_model
from AIedes.train.counter_autoregressive import (
    recursive_rollout,
    train_counter_autoregressive_minibatch,
)
from AIedes.utils.counter_losses import FrequencyWeightedLoss, compute_bin_weights


DEFAULT_CANDIDATE = {
    "fields": ["t2m_mean", "d2m_mean", "tp_sum"],
    "weekly": False,
    "lags": True,
    "lag_order": 2,
    "window": 90,
    "widths": [64, 32],
    "dropout": 0.2,
    "loss": "WMSLE",
    "learning_rate": 0.001,
}


def value_only_features(full_features, candidate):
    """Remove lag availability flags and return value-only feature metadata."""
    names = counter_feature_names(candidate)
    keep = [index for index, name in enumerate(names) if not name.endswith(":available")]
    features = np.asarray(full_features, dtype=np.float32)[:, keep]
    output_names = [names[index] for index in keep]
    slices = {}
    for lag, alias in ((1, "p1zn"), (2, "p2zn")):
        name = f"p{lag}:value"
        if name in output_names:
            index = output_names.index(name)
            slices[alias] = (index, index + 1)
    return features, output_names, slices


def _load_fold_data(payload):
    frame = pd.read_pickle(payload["input_path"]).reset_index(drop=True)
    manifest = json.loads(Path(payload["split_manifest"]).read_text())
    groups = np.asarray(manifest["outer_test_fold"], dtype=int)
    if len(groups) != len(frame):
        raise ValueError("Split manifest length does not match the Counter dataset")
    frame["outer_group"] = groups
    fold = int(payload["fold"])
    train = np.flatnonzero(groups != fold)
    test = np.flatnonzero(groups == fold)
    if not len(train) or not len(test):
        raise ValueError(f"Fold {fold} has an empty train or test partition")
    candidate = payload["candidate"]
    stats = fit_counter_preprocessing(frame, train, candidate)
    transformed = transform_counter_features(frame, candidate, stats)
    features, feature_names, feature_slices = value_only_features(transformed, candidate)
    return frame, train, test, stats, features, feature_names, feature_slices


def _run_fold(payload):
    """Train and persist one outer fold; called in an independent process."""
    fold = int(payload["fold"])
    output = Path(payload["output"])
    fold_output = output / f"fold{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, int(payload["threads"])))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A forked worker may inherit an already-initialized Torch runtime.
        # In that case the inter-op pool is already fixed; intra-op threads
        # remain explicitly bounded above.
        pass
    torch.manual_seed(int(payload["seed"]))
    np.random.seed(int(payload["seed"]))

    frame, train, test, stats, features, feature_names, feature_slices = _load_fold_data(payload)
    candidate = payload["candidate"]
    targets = frame.weeklyRates.to_numpy(dtype=np.float32)
    weights = compute_bin_weights(targets[train], [1.0], empty_bins_zero=True)
    criterion = FrequencyWeightedLoss(
        bins=[1.0], weights=weights.tolist(), device="cpu", log_loss=True, right=True
    )
    model = counter_candidate_model(candidate, features.shape[1])
    progress = fold_output / "progress.csv"
    progress.write_text("epoch,rollout_loss,r2,log_r2,rmse,rmsle,seconds\n")
    rows = []
    started = time.perf_counter()
    report_every = max(1, int(payload["report_every"]))

    def report(current_model, epoch, rollout_loss):
        if epoch % report_every and epoch != int(payload["epochs"]):
            return
        with torch.no_grad():
            prediction, _ = recursive_rollout(
                current_model, features, frame, test, stats, feature_slices,
                detach_feedback=True, initial_rate=0.0,
            )
        values = compute_metrics_from_arrays(targets[test], prediction.cpu().numpy())
        row = {
            "epoch": epoch,
            "rollout_loss": float(rollout_loss),
            "r2": float(values["r2"]),
            "log_r2": float(values["log_r2"]),
            "rmse": float(values["rmse"]),
            "rmsle": float(np.sqrt(values["log_mse"])),
            "seconds": float(time.perf_counter() - started),
        }
        rows.append(row)
        with progress.open("a") as stream:
            stream.write(",".join(str(row[key]) for key in row) + "\n")
        print(
            f"fold={fold} epoch={epoch:4d} R2={row['r2']:.4f} "
            f"logR2={row['log_r2']:.4f} RMSE={row['rmse']:.3f} "
            f"RMSLE={row['rmsle']:.4f} elapsed={row['seconds'] / 60:.1f} min",
            flush=True,
        )

    model, history = train_counter_autoregressive_minibatch(
        model, features, targets, frame, train, stats, feature_slices,
        criterion, epochs=int(payload["epochs"]),
        learning_rate=float(candidate["learning_rate"]),
        batch_size=int(payload["batch_size"]), scheduler_end_factor=0.9,
        initial_rate=0.0, callback=report,
    )
    with torch.no_grad():
        prediction, _ = recursive_rollout(
            model, features, frame, test, stats, feature_slices,
            detach_feedback=True, initial_rate=0.0,
        )
    prediction = prediction.cpu().numpy()
    metrics = compute_metrics_from_arrays(targets[test], prediction)
    summary = {
        "fold": fold,
        "seed": int(payload["seed"]),
        "candidate": candidate,
        "mode": "value_only_recursive_zero_start",
        "epochs": int(payload["epochs"]),
        "threads": int(payload["threads"]),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "seconds": float(time.perf_counter() - started),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "best_epoch_by_rollout_loss": int(min(history, key=lambda row: row["rollout_loss"])["epoch"]),
    }
    torch.save({
        "state_dict": model.state_dict(),
        "candidate": candidate,
        "stats": stats,
        "feature_names": feature_names,
        "feature_slices": feature_slices,
        "mode": summary["mode"],
        "fold": fold,
    }, fold_output / "model.pt")
    pd.DataFrame({
        "row_id": frame.iloc[test].row_id.to_numpy() if "row_id" in frame else test,
        "target": targets[test],
        "prediction": prediction,
    }).to_csv(fold_output / "test_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(fold_output / "training_history.csv", index=False)
    (fold_output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_autoregressive_experiment(
    input_path,
    split_manifest,
    output,
    candidate=None,
    folds=range(1, 6),
    workers=5,
    threads_per_worker=1,
    epochs=3000,
    batch_size=64,
    seed=1,
    report_every=20,
):
    """Run independent outer folds in parallel and return fold summaries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    candidate = dict(DEFAULT_CANDIDATE if candidate is None else candidate)
    folds = [int(fold) for fold in folds]
    workers = min(max(1, int(workers)), len(folds))
    payloads = [{
        "input_path": str(Path(input_path).resolve()),
        "split_manifest": str(Path(split_manifest).resolve()),
        "output": str(output.resolve()),
        "candidate": candidate,
        "fold": fold,
        "threads": int(threads_per_worker),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "report_every": int(report_every),
    } for fold in folds]
    def completed(payload):
        summary_path = output / f"fold{payload['fold']}" / "summary.json"
        if not summary_path.exists():
            return False
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return (
            summary.get("candidate") == payload["candidate"]
            and summary.get("epochs") == payload["epochs"]
            and summary.get("seed") == payload["seed"]
        )

    pending = [payload for payload in payloads if not completed(payload)]
    print(f"Autoregressive folds: {len(folds)} total, {len(pending)} pending, {workers} processes x {threads_per_worker} Torch threads", flush=True)
    summaries = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_fold, payload): payload for payload in pending}
        for future in as_completed(futures):
            summaries.append(future.result())
    summaries.extend(
        json.loads((output / f"fold{fold}" / "summary.json").read_text())
        for fold in folds if fold not in {item["fold"] for item in summaries}
    )
    summaries.sort(key=lambda item: item["fold"])
    (output / "fold_summaries.json").write_text(json.dumps(summaries, indent=2, sort_keys=True))
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("experiments/counter/data/eggs_y_norm_7days.pkl"))
    parser.add_argument("--split-manifest", type=Path, default=Path("experiments/counter/data/balanced_seed42.json"))
    parser.add_argument("--output", type=Path, default=Path("experiments/counter/results/autoregressive_zero"))
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1))
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--fold", type=int, choices=range(1, 6),
                        help="Run only this outer fold; omit to run all five")
    parser.add_argument(
        "--widths", default="64,32",
        help="Comma-separated hidden widths, e.g. 64 or 64,32",
    )
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1,
                        help="Training seed for every outer fold in this run")
    parser.add_argument("--report-every", type=int, default=20)
    args = parser.parse_args(argv)
    widths = [int(value.strip()) for value in args.widths.split(",") if value.strip()]
    if not widths or any(value < 1 for value in widths):
        parser.error("--widths must contain positive integers")
    candidate = dict(DEFAULT_CANDIDATE, widths=widths)
    run_autoregressive_experiment(
        args.input, args.split_manifest, args.output,
        candidate=candidate, workers=args.workers,
        threads_per_worker=args.threads_per_worker, seed=args.seed,
        folds=[args.fold] if args.fold is not None else range(1, 6),
        epochs=args.epochs, batch_size=args.batch_size, report_every=args.report_every,
    )


if __name__ == "__main__":
    main()
