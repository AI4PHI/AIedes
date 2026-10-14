"""Evaluate the completed zero-start autoregressive ensemble on VectorNet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from AIedes.data_loader.counter_data_loader import transform_counter_features
from AIedes.models.counter_models import counter_candidate_model
from AIedes.train.counter_autoregressive import recursive_rollout
from AIedes.train.counter_autoregressive_experiment import value_only_features


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXTERNAL = ROOT / "experiments/counter/data/eggs_y_norm_7days_vectorNet.pkl"
DEFAULT_JOBS = ROOT / "experiments/counter/results/autoregressive_jobs"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def load_prediction(model_path: Path, external: pd.DataFrame) -> np.ndarray:
    package = torch.load(model_path, map_location="cpu", weights_only=False)
    candidate = package["candidate"]
    full_features = transform_counter_features(external, candidate, package["stats"])
    features, _, feature_slices = value_only_features(full_features, candidate)
    model = counter_candidate_model(candidate, features.shape[1])
    model.load_state_dict(package["state_dict"])
    model.eval()
    with torch.no_grad():
        prediction, _ = recursive_rollout(
            model,
            features,
            external,
            np.arange(len(external)),
            package["stats"],
            feature_slices,
            initial_rate=0.0,
        )
    return prediction.numpy()


def summarize_traps(frame: pd.DataFrame) -> dict:
    dates = pd.to_datetime(frame["date"])
    first_lag = frame["prev1_rates"].notna()
    second_lag = frame["prev2_rates"].notna()
    lengths = frame.groupby("id_trap").size()
    return {
        "observations": int(len(frame)),
        "traps": int(frame["id_trap"].nunique()),
        "date_first": str(dates.min().date()),
        "date_last": str(dates.max().date()),
        "lag_availability": {
            "prev1": int(first_lag.sum()),
            "prev2": int(second_lag.sum()),
            "both": int((first_lag & second_lag).sum()),
            "exactly_one": int((first_lag ^ second_lag).sum()),
            "neither": int((~first_lag & ~second_lag).sum()),
        },
        "trap_length": {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "minimum": int(lengths.min()),
            "maximum": int(lengths.max()),
            "at_least_16": int((lengths >= 16).sum()),
            "frequency": {str(int(k)): int(v) for k, v in lengths.value_counts().sort_index().items()},
        },
        "periods": {
            "before_september": {
                "observations": int((dates < pd.Timestamp("2021-09-01")).sum()),
                "traps": int(frame.loc[dates < pd.Timestamp("2021-09-01"), "id_trap"].nunique()),
            },
            "september_onward": {
                "observations": int((dates >= pd.Timestamp("2021-09-01")).sum()),
                "traps": int(frame.loc[dates >= pd.Timestamp("2021-09-01"), "id_trap"].nunique()),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    external = pd.read_pickle(args.external).reset_index(drop=True).copy()
    external["date"] = pd.to_datetime(external["end_date"])
    predictions = []
    model_paths = []
    for fold in range(1, 6):
        for seed in range(1, 6):
            path = args.jobs / f"64_32_seed{seed}_fold{fold}" / f"fold{fold}" / "model.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            model_paths.append(str(path))
            predictions.append(load_prediction(path, external))

    external["target"] = external["weeklyRates"].to_numpy(float)
    external["prediction"] = np.mean(predictions, axis=0)
    columns = ["id_trap", "countryCode", "date", "target", "prediction"]
    external[columns].to_csv(args.output / "vectornet_autoregressive_predictions.csv", index=False)
    report = summarize_traps(external)
    report.update({"models": len(model_paths), "model_paths": model_paths,
                   "mode": "zero-start recursive autoregressive ensemble"})
    (args.output / "vectornet_autoregressive_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
