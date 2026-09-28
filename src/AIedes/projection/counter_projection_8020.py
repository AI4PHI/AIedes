#!/usr/bin/env python3
"""Project Europe with the median-performing 80/20 climate-only models.

This is a comparison pipeline for the deployment projection.  It selects one
saved climate-only model from each outer fold: the seed whose held-out
``test_loss`` is the median of the five seeds in that fold.  The selected
models keep their original 80/20-fold normalization statistics and are not
refitted on the full data.

Run from the repository root::

    /home/biazzin/conda-envs/aiedes-env/bin/python \
        experiments/counter/project_europe_8020_median.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments" / "counter"

from AIedes.projection.counter_projection import (
    DEFAULT_CLIMATE,
    DEFAULT_END,
    DEFAULT_MONTHS,
    DEFAULT_START,
    MODEL_ID,
    load_model,
    plot_monthly_map,
    predict_grid_parallel,
    read_json,
    save_dataset,
    selected_climate_only,
    validate_grid,
)


DEFAULT_RESULTS = HERE / "results"
DEFAULT_OUTPUT = HERE / "results" / "europe_projection_80_20_median"


def median_fold_models(results: Path) -> list[dict]:
    """Return the median-test-loss model from each of the five outer folds."""
    selected = []
    for fold in range(1, 6):
        fold_dir = results / "final_models" / f"outer{fold}"
        records = []
        for result_path in sorted(fold_dir.glob(f"{MODEL_ID}_seed*/result.json")):
            result = read_json(result_path)
            test_loss = result.get("test_loss")
            if test_loss is None or not np.isfinite(float(test_loss)):
                raise ValueError(f"Missing finite test_loss in {result_path}")
            model_path = result_path.parent / "model.pt"
            if not model_path.is_file():
                raise FileNotFoundError(model_path)
            records.append(
                {
                    "fold": fold,
                    "seed": int(result_path.parent.name.rsplit("seed", 1)[1]),
                    "test_loss": float(test_loss),
                    "result_path": str(result_path),
                    "model_path": str(model_path),
                }
            )
        if len(records) != 5:
            raise ValueError(
                f"Expected five {MODEL_ID} models in {fold_dir}, found {len(records)}"
            )
        ordered = sorted(records, key=lambda item: (item["test_loss"], item["seed"]))
        chosen = ordered[len(ordered) // 2]
        chosen["ranked_test_losses"] = [item["test_loss"] for item in ordered]
        selected.append(chosen)
    return selected


def compare_prediction_ranges(current_path: Path, comparison_path: Path) -> dict:
    """Summarize the finite prediction distributions for the two ensembles."""
    summary = {}
    for label, path in (("full_data", current_path), ("8020_median", comparison_path)):
        with xr.open_dataset(path) as dataset:
            values = np.asarray(dataset["weekly_rate"].values, dtype=float)
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"No finite predictions found in {path}")
        summary[label] = {
            "path": str(path),
            "count": int(finite.size),
            "min": float(np.min(finite)),
            "p01": float(np.quantile(finite, 0.01)),
            "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99)),
            "max": float(np.max(finite)),
            "mean": float(np.mean(finite)),
        }
    return summary


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    daily_path = output / "predictions_2020_climate_only_80_20_median_daily.nc"
    map_png = output / "weekly_rates_map_climate_only_80_20_median.png"
    map_pdf = output / "weekly_rates_map_climate_only_80_20_median.pdf"
    if args.plot_only:
        if not daily_path.is_file():
            raise FileNotFoundError(f"Missing projection file for --plot-only: {daily_path}")
        with xr.open_dataset(daily_path) as source:
            daily = source.load()
        plot_monthly_map(daily, map_png, map_pdf, months=tuple(args.months))
        print(f"Saved comparison map: {map_png}", flush=True)
        return output

    results = args.results.resolve()
    selected_candidate = selected_climate_only(results)
    selected_models = median_fold_models(results)

    loaded = [load_model(Path(item["model_path"])) for item in selected_models]
    models = [model for model, _ in loaded]
    packages = [package for _, package in loaded]
    if any(package["candidate"] != selected_candidate for package in packages):
        raise ValueError("A selected 80/20 model does not match the climate-only candidate")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError("Projection end date precedes start date")
    climate_path = args.climate.resolve()
    if not climate_path.is_file():
        raise FileNotFoundError(climate_path)
    with xr.open_dataset(climate_path) as source:
        climate = source.sortby("time").load()
    validate_grid(climate, start, end)
    all_dates = pd.DatetimeIndex(climate.time.values)
    dates = pd.date_range(start, end, freq="D")
    if not dates.isin(all_dates).all():
        raise ValueError("Requested projection dates are missing from the climate grid")

    print("Selected median-test-loss 80/20 models:", flush=True)
    for item in selected_models:
        print(
            f"  outer{item['fold']} seed{item['seed']} "
            f"test_loss={item['test_loss']:.8f}",
            flush=True,
        )

    daily = predict_grid_parallel(climate, packages, models, dates, args.prediction_workers)
    weekly_path = output / "predictions_2020_climate_only_80_20_median_weekly.nc"
    monthly_path = output / "predictions_2020_climate_only_80_20_median_monthly.nc"
    save_dataset(daily, daily_path)
    weekly = daily.resample(time="7D", origin=dates[0], label="right", closed="right").mean()
    monthly = daily.resample(time="1ME").mean()
    save_dataset(weekly, weekly_path)
    save_dataset(monthly, monthly_path)

    if not args.no_plot:
        plot_monthly_map(daily, map_png, map_pdf, months=tuple(args.months))

    current_daily = HERE / "results" / "europe_projection" / "predictions_2020_climate_only_daily.nc"
    comparison = compare_prediction_ranges(current_daily, daily_path)
    metadata = {
        "model_id": MODEL_ID,
        "model_mode": "80_20_outer_fold_median_test_loss",
        "candidate": selected_candidate,
        "selected_models": selected_models,
        "climate_path": str(climate_path),
        "start": str(start.date()),
        "end": str(end.date()),
        "prediction_workers": args.prediction_workers,
        "outputs": [str(daily_path), str(weekly_path), str(monthly_path)],
        "map_outputs": [str(map_png), str(map_pdf)] if not args.no_plot else [],
        "prediction_range_comparison": comparison,
    }
    with (output / "projection_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, default=str)
    print(json.dumps(comparison, indent=2, sort_keys=True), flush=True)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--climate", type=Path, default=DEFAULT_CLIMATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--months", nargs="+", type=int, default=list(DEFAULT_MONTHS))
    parser.add_argument(
        "--prediction-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel prediction workers; defaults to CPU count minus one",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
