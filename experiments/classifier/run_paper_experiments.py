#!/usr/bin/env python3
"""Reproducible paper experiments for the AIedes Classifier.

This driver deliberately reuses the project classifier model, data loader,
training loop, and evaluation functions.  It adds explicit split bookkeeping,
training-only scaling, reproducible seeds, and plain result tables required for
the revised manuscript.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

# These must be defined before NumPy/PyTorch are imported.  Every parallel
# worker represents one independent model fit and is intentionally restricted
# to one CPU thread.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from AIedes.data_loader.classificator_data_loader import get_data_loaders_from_indices
from AIedes.evaluation.classificator_evaluate import classification_metrics, predict_probabilities
from AIedes.train.classificator_train import build_model, train_model


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "experiments/classifier/paper_results/input/copernicus_classifier_prepared.csv.gz"
DEFAULT_OUTPUT = ROOT / "experiments/classifier/paper_results"
TEMPERATURE_FEATURES = [f"temp_{month}_C" for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
PRECIPITATION_FEATURES = [f"precip_{month}_mm" for month in ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")]
INPUT_SETS = {
    "temp": TEMPERATURE_FEATURES,
    "temp_tp": TEMPERATURE_FEATURES + PRECIPITATION_FEATURES,
}
ARCHITECTURES = [(), (10,), (10, 10), (10, 10, 10), (20,), (20, 20), (20, 20, 20), (30,), (30, 30), (30, 30, 30), (40, 40), (40, 40, 40), (50, 50), (50, 50, 50)]
SELECTION_CONTEXT: dict = {}
PREDICTION_CONTEXT: dict = {}


def architecture_name(layers: tuple[int, ...]) -> str:
    return json.dumps(list(layers), separators=(",", ","))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def configure_worker_threads() -> None:
    """Configure a fresh worker before it performs any autograd work."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits inter-op configuration only before its parallel
        # runtime has been used.  The environment variables above still keep
        # the underlying numerical libraries single-threaded.
        pass


def initialize_selection_worker(X_by_input: dict[str, np.ndarray], target: np.ndarray) -> None:
    """Initialize selection data in a spawned worker process."""
    global SELECTION_CONTEXT
    configure_worker_threads()
    SELECTION_CONTEXT = {"X": X_by_input, "target": target}


def initialize_prediction_worker(X: np.ndarray, target: np.ndarray) -> None:
    """Initialize prediction data in a spawned worker process."""
    global PREDICTION_CONTEXT
    configure_worker_threads()
    PREDICTION_CONTEXT = {"X": X, "target": target}


def write_json(path: Path, contents: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contents, indent=2, sort_keys=True) + "\n")


def assert_disjoint(*index_sets: np.ndarray) -> None:
    for position, left in enumerate(index_sets):
        for right in index_sets[position + 1:]:
            if np.intersect1d(left, right).size:
                raise RuntimeError("Data partitions overlap.")


def fit_and_predict(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    predict_idx: np.ndarray,
    layers: tuple[int, ...],
    seed: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    validation_idx: np.ndarray | None = None,
    return_artifacts: bool = False,
) -> tuple[np.ndarray, dict, dict, dict, int]:
    """Fit the existing model implementation and predict an explicit partition."""
    assert_disjoint(train_idx, predict_idx)
    if validation_idx is not None:
        assert_disjoint(train_idx, validation_idx)
        if not np.array_equal(np.asarray(validation_idx), np.asarray(predict_idx)):
            assert_disjoint(validation_idx, predict_idx)
    set_seed(seed)
    train_loader, val_loader, predict_loader, scaler = get_data_loaders_from_indices(
        X,
        y,
        train_idx=train_idx,
        val_idx=validation_idx,
        test_idx=predict_idx,
        batch_size=batch_size,
        num_workers=0,
    )
    model, criterion, optimizer = build_model(
        X.shape[1], learning_rate, hidden_layers=list(layers), device="cpu", verbose=False
    )
    history: dict | None = {} if return_artifacts or validation_idx is not None else None
    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        None,
        epochs,
        device="cpu",
        metrics_dict=history,
        show_progress=False,
    )
    labels, probabilities = predict_probabilities(model, predict_loader, device="cpu")
    if not np.array_equal(labels.astype(int), y[predict_idx].astype(int)):
        raise RuntimeError("Prediction labels no longer match the requested rows.")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    # NumPy arrays are used for process-safe checkpoint transport.  PyTorch
    # tensor transport otherwise requests file descriptors, which is not
    # available in the execution sandbox.
    state = (
        {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
        if return_artifacts else None
    )
    return probabilities, history or {}, scaler, state, parameters


def selection_worker(task: dict) -> dict:
    """Run one independent architecture/fold fit in a single CPU worker."""
    X = SELECTION_CONTEXT["X"][task["input_set"]]
    target = SELECTION_CONTEXT["target"]
    probabilities, _, _, _, parameters = fit_and_predict(
        X,
        target,
        task["train_idx"],
        task["validation_idx"],
        task["layers"],
        seed=task["seed"],
        epochs=task["epochs"],
        learning_rate=task["learning_rate"],
        batch_size=task["batch_size"],
        validation_idx=task["validation_idx"],
    )
    metric = classification_metrics(target[task["validation_idx"]], probabilities)
    record = {
        "input_set": task["input_set"],
        "architecture": architecture_name(task["layers"]),
        "n_hidden_layers": len(task["layers"]),
        "n_parameters": parameters,
        "fold": task["fold"],
        **metric,
    }
    return record


def prediction_worker(task: dict) -> dict:
    """Run one independent final, map, or spatial-validation model fit."""
    X = PREDICTION_CONTEXT["X"]
    target = PREDICTION_CONTEXT["target"]
    probabilities, history, scaler, state, parameters = fit_and_predict(
        X,
        target,
        task["train_idx"],
        task["predict_idx"],
        task["layers"],
        seed=task["seed"],
        epochs=task["epochs"],
        learning_rate=task["learning_rate"],
        batch_size=task["batch_size"],
        return_artifacts=task.get("return_artifacts", False),
    )
    return {
        **{key: value for key, value in task.items() if key not in {"train_idx", "predict_idx"}},
        "probabilities": probabilities,
        "metrics": classification_metrics(target[task["predict_idx"]], probabilities),
        "scaler": scaler,
        "state": state,
        "parameters": parameters,
        "history": history,
    }


def checkpoint_path(checkpoint_dir: Path, task: dict) -> Path:
    """Return the stable checkpoint path for one independent prediction fit."""
    return checkpoint_dir / f"{task['checkpoint_key']}.npz"


def load_prediction_checkpoint(path: Path, task: dict) -> dict:
    """Load the compact outputs needed to resume an interrupted comparison."""
    with np.load(path, allow_pickle=False) as checkpoint:
        probabilities = checkpoint["probabilities"]
        metrics = json.loads(str(checkpoint["metrics_json"].item()))
        parameters = int(checkpoint["parameters"].item())
    return {
        **{key: value for key, value in task.items() if key not in {"train_idx", "predict_idx"}},
        "probabilities": probabilities,
        "metrics": metrics,
        "scaler": None,
        "state": None,
        "parameters": parameters,
        "history": {},
    }


def save_prediction_checkpoint(path: Path, result: dict) -> None:
    """Atomically save compact prediction outputs for one completed fit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as destination:
        np.savez_compressed(
            destination,
            probabilities=result["probabilities"],
            metrics_json=np.asarray(json.dumps(result["metrics"], sort_keys=True)),
            parameters=np.asarray(result["parameters"]),
        )
    temporary.replace(path)


def run_parallel_prediction_tasks(
    X: np.ndarray,
    target: np.ndarray,
    tasks: list[dict],
    jobs: int,
    checkpoint_dir: Path | None = None,
) -> list[dict]:
    """Execute independent fits in spawned, single-threaded workers.

    A forked process is unsafe for PyTorch autograd after the parent has
    initialized the Torch runtime.  Spawned workers receive their data through
    the initializer and start with a clean autograd runtime.
    """
    results: list[dict] = []
    remaining_tasks: list[dict] = []
    if checkpoint_dir is not None:
        for task in tasks:
            path = checkpoint_path(checkpoint_dir, task)
            if path.exists():
                results.append(load_prediction_checkpoint(path, task))
            else:
                remaining_tasks.append(task)
        print(
            f"Prediction fits already checkpointed: {len(results)}; "
            f"remaining: {len(remaining_tasks)}",
            flush=True,
        )
    else:
        remaining_tasks = tasks

    if not remaining_tasks:
        return results

    with ProcessPoolExecutor(
        max_workers=min(jobs, len(remaining_tasks)),
        mp_context=get_context("spawn"),
        initializer=initialize_prediction_worker,
        initargs=(X, target),
    ) as executor:
        futures = {
            executor.submit(prediction_worker, task): task for task in remaining_tasks
        }
        for complete, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if checkpoint_dir is not None:
                save_prediction_checkpoint(
                    checkpoint_path(checkpoint_dir, futures[future]), result
                )
            if complete % 5 == 0 or complete == len(remaining_tasks):
                print(
                    f"Completed independent fit {complete}/{len(remaining_tasks)}",
                    flush=True,
                )
    return results


def load_data(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    required = {"row_id", "target_presence", "NUTS3_Code", "CountryCode", "latitude", "longitude"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Prepared input is missing: {', '.join(sorted(missing))}")
    return dataframe.sort_values("row_id").reset_index(drop=True)


def outer_random_split(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    return train_test_split(all_idx, test_size=0.20, stratify=y, random_state=seed)


def run_selection(dataframe: pd.DataFrame, output: Path, arguments) -> dict:
    """Choose input set and architecture without using the outer test partition."""
    target = dataframe["target_presence"].to_numpy(dtype=int)
    outer_train, outer_test = outer_random_split(target, arguments.seed)
    assert_disjoint(outer_train, outer_test)
    split_table = dataframe.loc[:, ["row_id", "NUTS3_Code", "CountryCode"]].copy()
    split_table["outer_partition"] = "train"
    split_table.loc[outer_test, "outer_partition"] = "test"
    split_table.to_csv(output / "main_random_split.csv", index=False)

    folds = StratifiedKFold(n_splits=arguments.selection_folds, shuffle=True, random_state=arguments.seed)
    global SELECTION_CONTEXT
    SELECTION_CONTEXT = {
        "X": {name: dataframe[features].to_numpy(dtype=np.float32) for name, features in INPUT_SETS.items()},
        "target": target,
    }
    partial_metrics_path = output / "inner_validation_metrics.partial.csv"
    existing_records = (
        pd.read_csv(partial_metrics_path).to_dict("records")
        if partial_metrics_path.exists() else []
    )
    completed_keys = {
        (record["input_set"], record["architecture"], int(record["fold"]))
        for record in existing_records
    }
    tasks: list[dict] = []
    for input_name, features in INPUT_SETS.items():
        for layers in ARCHITECTURES:
            for fold, (relative_train, relative_validation) in enumerate(
                folds.split(outer_train, target[outer_train]), start=1
            ):
                train_idx = outer_train[relative_train]
                validation_idx = outer_train[relative_validation]
                task = {
                    "input_set": input_name,
                    "layers": layers,
                    "fold": fold,
                    "train_idx": train_idx,
                    "validation_idx": validation_idx,
                    "seed": arguments.seed + fold,
                    "epochs": arguments.epochs,
                    "learning_rate": arguments.learning_rate,
                    "batch_size": arguments.batch_size,
                }
                task_key = (input_name, architecture_name(layers), fold)
                if task_key not in completed_keys:
                    tasks.append(task)
    records: list[dict] = list(existing_records)
    print(f"Selection fits already checkpointed: {len(existing_records)}; remaining: {len(tasks)}", flush=True)
    if tasks:
        with ProcessPoolExecutor(
            max_workers=min(arguments.jobs, len(tasks)),
            mp_context=get_context("spawn"),
            initializer=initialize_selection_worker,
            initargs=(SELECTION_CONTEXT["X"], target),
        ) as executor:
            futures = [executor.submit(selection_worker, task) for task in tasks]
            for complete, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                pd.DataFrame([record]).to_csv(
                    partial_metrics_path,
                    mode="a",
                    header=not partial_metrics_path.exists(),
                    index=False,
                )
                if complete % 5 == 0 or complete == len(tasks):
                    print(f"Completed selection fit {complete}/{len(tasks)}", flush=True)
    results = pd.DataFrame(records).sort_values(["input_set", "architecture", "fold"])
    results.to_csv(output / "inner_validation_metrics.csv", index=False)
    summary = results.groupby(["input_set", "architecture", "n_hidden_layers", "n_parameters"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"), sd_accuracy=("accuracy", "std"),
        mean_sensitivity=("sensitivity", "mean"), mean_specificity=("specificity", "mean"),
        mean_csi=("csi", "mean"),
    )
    summary = summary.sort_values(["mean_accuracy", "n_parameters"], ascending=[False, True])
    summary.to_csv(output / "inner_validation_summary.csv", index=False)
    selected = summary.iloc[0].to_dict()
    selected["outer_train_rows"] = int(len(outer_train))
    selected["outer_test_rows"] = int(len(outer_test))
    selected["seed"] = arguments.seed
    write_json(output / "selected_configuration.json", selected)
    # Only one representative loss curve is needed for the paper figure.  It is
    # deliberately rerun after configuration selection, avoiding fragile large
    # loss-history checkpoints during the parallel architecture sweep.
    selected_input = selected["input_set"]
    selected_architecture = tuple(json.loads(selected["architecture"]))
    first_train, first_validation = next(folds.split(outer_train, target[outer_train]))
    _, history, _, _, _ = fit_and_predict(
        SELECTION_CONTEXT["X"][selected_input],
        target,
        outer_train[first_train],
        outer_train[first_validation],
        selected_architecture,
        seed=arguments.seed + 1,
        epochs=arguments.epochs,
        learning_rate=arguments.learning_rate,
        batch_size=arguments.batch_size,
        validation_idx=outer_train[first_validation],
    )
    pd.DataFrame({
        "epoch": range(1, len(history["train_losses"]) + 1),
        "train_loss": history["train_losses"],
        "validation_loss": history["val_losses"],
    }).to_csv(output / "selected_configuration_loss_history.csv", index=False)
    return selected


def selected_layers(selected: dict) -> tuple[int, ...]:
    return tuple(json.loads(selected["architecture"]))


def run_main_outer_test(dataframe: pd.DataFrame, output: Path, selected: dict, arguments) -> None:
    target = dataframe["target_presence"].to_numpy(dtype=int)
    outer_train, outer_test = outer_random_split(target, arguments.seed)
    layers = selected_layers(selected)
    features = INPUT_SETS[selected["input_set"]]
    X = dataframe[features].to_numpy(dtype=np.float32)
    prediction_table = dataframe.loc[outer_test, ["row_id", "latitude", "longitude", "NUTS3_Code", "CountryCode", "target_presence"]].copy()
    tasks = []
    for seed in arguments.ensemble_seeds:
        tasks.append({
            "seed": seed, "train_idx": outer_train, "predict_idx": outer_test,
            "layers": layers, "epochs": arguments.epochs,
            "learning_rate": arguments.learning_rate, "batch_size": arguments.batch_size,
            "return_artifacts": True,
        })
    fit_results = run_parallel_prediction_tasks(X, target, tasks, arguments.jobs)
    seed_probabilities = []
    metric_records = []
    models_dir = output / "models/main_outer_test"
    models_dir.mkdir(parents=True, exist_ok=True)
    for result in sorted(fit_results, key=lambda result: result["seed"]):
        seed = result["seed"]
        probabilities = result["probabilities"]
        prediction_table[f"probability_seed_{seed}"] = probabilities
        seed_probabilities.append(probabilities)
        metric_records.append({"seed": seed, **result["metrics"]})
        torch.save({"state_dict": result["state"], "scaler": result["scaler"], "features": features, "layers": list(layers)}, models_dir / f"seed_{seed}.pt")
    ensemble_probability = np.mean(np.vstack(seed_probabilities), axis=0)
    prediction_table["probability_ensemble"] = ensemble_probability
    prediction_table["predicted_presence"] = (ensemble_probability >= 0.5).astype(int)
    prediction_table.to_csv(output / "main_outer_test_predictions.csv", index=False)
    metrics = classification_metrics(target[outer_test], ensemble_probability)
    write_json(output / "main_outer_test_ensemble_metrics.json", metrics)
    pd.DataFrame(metric_records).to_csv(output / "main_outer_test_seed_metrics.csv", index=False)


def comparison_folds(
    X: np.ndarray,
    target: np.ndarray,
    dataframe: pd.DataFrame,
    mode: str,
    arguments,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict]]:
    """Build and validate random or NUTS3-grouped comparison folds."""
    if mode == "random":
        splitter = StratifiedKFold(
            n_splits=arguments.map_folds,
            shuffle=True,
            random_state=arguments.seed,
        )
        folds = list(splitter.split(X, target))
        return folds, []

    groups = dataframe["NUTS3_Code"].to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=arguments.nuts3_folds,
        shuffle=True,
        random_state=arguments.seed,
    )
    folds = list(splitter.split(X, target, groups))
    split_records: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        train_regions = set(groups[train_idx])
        test_regions = set(groups[test_idx])
        if train_regions.intersection(test_regions):
            raise RuntimeError("A NUTS3 region occurs in both training and test partitions.")
        split_records.extend(
            {"fold": fold, "partition": "train", "NUTS3_Code": code}
            for code in sorted(train_regions)
        )
        split_records.extend(
            {"fold": fold, "partition": "test", "NUTS3_Code": code}
            for code in sorted(test_regions)
        )
    return folds, split_records


def metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize ensemble metrics across folds separately for each model."""
    records: list[dict] = []
    ensemble = metrics[metrics["aggregation"] == "ensemble"]
    for model, group in ensemble.groupby("model"):
        for metric in ("accuracy", "sensitivity", "specificity", "csi"):
            records.append({
                "model": model,
                "metric": metric,
                "mean": group[metric].mean(),
                "sd": group[metric].std(),
            })
    return pd.DataFrame(records)


def run_oof_comparison(
    dataframe: pd.DataFrame,
    output: Path,
    selected: dict,
    arguments,
    mode: str,
) -> None:
    """Compare logistic and selected neural models on matched held-out rows."""
    if mode not in {"random", "nuts3"}:
        raise ValueError(f"Unsupported comparison mode: {mode}")
    target = dataframe["target_presence"].to_numpy(dtype=int)
    layers = selected_layers(selected)
    features = INPUT_SETS[selected["input_set"]]
    X = dataframe[features].to_numpy(dtype=np.float32)
    folds, split_records = comparison_folds(X, target, dataframe, mode, arguments)
    fold_indices = {fold: test_idx for fold, (_, test_idx) in enumerate(folds, start=1)}

    tasks: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(folds, start=1):
        for model_name, model_layers in (("logistic", ()), ("aiedes", layers)):
            for seed in arguments.ensemble_seeds:
                tasks.append({
                    "checkpoint_key": f"fold_{fold}_{model_name}_seed_{seed}",
                    "fold": fold,
                    "model": model_name,
                    "seed": seed,
                    "train_idx": train_idx,
                    "predict_idx": test_idx,
                    "layers": model_layers,
                    "epochs": arguments.epochs,
                    "learning_rate": arguments.learning_rate,
                    "batch_size": arguments.batch_size,
                })

    fit_results = run_parallel_prediction_tasks(
        X,
        target,
        tasks,
        arguments.jobs,
        checkpoint_dir=output / "checkpoints" / f"{mode}_oof",
    )

    prediction_table = dataframe.loc[:, [
        "row_id", "latitude", "longitude", "NUTS3_Code", "CountryCode",
        "target_presence",
    ]].copy()
    prediction_table["fold"] = -1
    prediction_table["logistic_probability"] = np.nan
    prediction_table["nn_probability"] = np.nan
    metric_records: list[dict] = []

    for fold, test_idx in fold_indices.items():
        prediction_table.loc[test_idx, "fold"] = fold
        for model_name, column in (
            ("logistic", "logistic_probability"),
            ("aiedes", "nn_probability"),
        ):
            model_results = sorted(
                (
                    result for result in fit_results
                    if result["fold"] == fold and result["model"] == model_name
                ),
                key=lambda result: result["seed"],
            )
            if len(model_results) != len(arguments.ensemble_seeds):
                raise RuntimeError(
                    f"Expected {len(arguments.ensemble_seeds)} {model_name} fits "
                    f"for fold {fold}, found {len(model_results)}."
                )
            for result in model_results:
                metric_records.append({
                    "fold": fold,
                    "seed": result["seed"],
                    "model": model_name,
                    "aggregation": "seed",
                    **result["metrics"],
                })
            probability = np.mean(
                np.vstack([result["probabilities"] for result in model_results]),
                axis=0,
            )
            prediction_table.loc[test_idx, column] = probability
            metric_records.append({
                "fold": fold,
                "seed": "ensemble",
                "model": model_name,
                "aggregation": "ensemble",
                **classification_metrics(target[test_idx], probability),
            })

    probability_columns = ["logistic_probability", "nn_probability"]
    if prediction_table[probability_columns].isna().any().any():
        raise RuntimeError(f"{mode} out-of-fold predictions do not cover every row.")
    if prediction_table["fold"].lt(1).any():
        raise RuntimeError(f"{mode} fold assignments do not cover every row.")

    metrics = pd.DataFrame(metric_records).sort_values(
        ["model", "fold", "aggregation", "seed"]
    )
    pooled = pd.DataFrame([
        {"model": model_name, **classification_metrics(target, prediction_table[column].to_numpy())}
        for model_name, column in (
            ("logistic", "logistic_probability"),
            ("aiedes", "nn_probability"),
        )
    ])

    if mode == "random":
        prediction_table.to_csv(output / "main_random_oof_map_predictions.csv", index=False)
        metrics.to_csv(output / "main_random_oof_map_metrics_by_fold.csv", index=False)
        pooled.to_csv(output / "main_random_oof_map_metrics_pooled.csv", index=False)
        metric_summary(metrics).to_csv(
            output / "main_random_oof_map_ensemble_summary.csv", index=False
        )
    else:
        prediction_table.to_csv(output / "nuts3_held_region_predictions.csv", index=False)
        metrics.to_csv(output / "nuts3_metrics_by_fold.csv", index=False)
        pooled.to_csv(output / "nuts3_metrics_pooled.csv", index=False)
        metric_summary(metrics).to_csv(output / "nuts3_ensemble_summary.csv", index=False)
        pd.DataFrame(split_records).to_csv(output / "nuts3_split_manifest.csv", index=False)


def run_random_oof_maps(dataframe: pd.DataFrame, output: Path, selected: dict, arguments) -> None:
    """Create random out-of-fold predictions for the main comparison figure."""
    run_oof_comparison(dataframe, output, selected, arguments, mode="random")


def run_nuts3_sensitivity(dataframe: pd.DataFrame, output: Path, selected: dict, arguments) -> None:
    """Create NUTS3-grouped held-region predictions for the SI comparison."""
    run_oof_comparison(dataframe, output, selected, arguments, mode="nuts3")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--stage",
        choices=("all", "selection", "outer-test", "maps", "nuts3", "comparisons"),
        default="all",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-folds", type=int, default=5)
    parser.add_argument("--map-folds", type=int, default=5)
    parser.add_argument("--nuts3-folds", type=int, default=5)
    parser.add_argument("--ensemble-seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--jobs", type=int, default=32,
                        help="Independent single-threaded training workers.")
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    dataframe = load_data(arguments.data)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    run_configuration = {
        **vars(arguments),
        "data": os.path.relpath(arguments.data, ROOT),
        "output_dir": os.path.relpath(arguments.output_dir, ROOT),
        "torch_version": torch.__version__,
    }
    write_json(arguments.output_dir / "run_configuration.json", run_configuration)
    write_json(
        arguments.output_dir / f"run_configuration_{arguments.stage}.json",
        run_configuration,
    )
    selected_path = arguments.output_dir / "selected_configuration.json"
    if arguments.stage in {"all", "selection"}:
        selected = run_selection(dataframe, arguments.output_dir, arguments)
    else:
        if not selected_path.exists():
            raise FileNotFoundError("Run the selection stage before this stage.")
        selected = json.loads(selected_path.read_text())
    if arguments.stage in {"all", "outer-test"}:
        run_main_outer_test(dataframe, arguments.output_dir, selected, arguments)
    if arguments.stage in {"all", "maps", "comparisons"}:
        run_random_oof_maps(dataframe, arguments.output_dir, selected, arguments)
    if arguments.stage in {"all", "nuts3", "comparisons"}:
        run_nuts3_sensitivity(dataframe, arguments.output_dir, selected, arguments)


if __name__ == "__main__":
    main()
