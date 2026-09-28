"""Aggregate the five-seed, five-fold autoregressive predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(data: pd.DataFrame) -> dict[str, float]:
    y = data.target.to_numpy(float)
    p = data.prediction.to_numpy(float)
    ly, lp = np.log1p(y), np.log1p(p)
    return {
        "n": int(len(data)),
        "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)),
        "log_r2": float(1 - np.sum((ly - lp) ** 2) / np.sum((ly - ly.mean()) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "rmsle": float(np.sqrt(np.mean((ly - lp) ** 2))),
    }


def aggregate(root: Path, architecture: str, eligible: Path | None = None) -> tuple[pd.DataFrame, dict]:
    eligible_ids = None
    if eligible is not None and eligible.is_file():
        reference = pd.read_csv(eligible, usecols=["row_id", "reconstruction_eligible"])
        eligible_ids = set(reference.loc[reference.reconstruction_eligible, "row_id"])
    folds = []
    for fold in range(1, 6):
        seeds = []
        for seed in range(1, 6):
            path = root / f"{architecture}_seed{seed}_fold{fold}" / f"fold{fold}" / "test_predictions.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            data = pd.read_csv(path).set_index("row_id")
            seeds.append(data[["target", "prediction"]].rename(columns={"prediction": f"seed{seed}"}))
        fold_data = seeds[0][["target"]].copy()
        for seed_data in seeds:
            if not np.array_equal(fold_data.index, seed_data.index):
                raise ValueError(f"Seed row mismatch in fold {fold}")
            fold_data = fold_data.join(seed_data.drop(columns="target"))
        fold_data["prediction"] = fold_data.filter(like="seed").mean(axis=1)
        if eligible_ids is not None:
            fold_data = fold_data.loc[fold_data.index.intersection(eligible_ids)]
        folds.append(fold_data[["target", "prediction"]])
    pooled = pd.concat(folds).reset_index()
    return pooled, metrics(pooled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("experiments/counter/results/autoregressive_jobs"))
    parser.add_argument("--architecture", default="64_32")
    parser.add_argument(
        "--eligible", type=Path,
        default=Path("experiments/counter/results/evaluation/reconstruction_predictions.csv"),
        help="Optional reconstruction cohort used for the matched comparison.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    pooled, result = aggregate(args.root, args.architecture, args.eligible)
    output = args.output or args.root / f"comparison_{args.architecture}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    pooled.to_csv(output.with_name(f"pooled_predictions_{args.architecture}.csv"), index=False)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
