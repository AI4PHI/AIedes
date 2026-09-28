#!/usr/bin/env python3
"""Diagnose and mask out-of-distribution climate conditions in the projection."""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments" / "counter"

from AIedes.data_loader.counter_data_loader import transform_counter_features  # noqa: E402
from AIedes.projection.counter_projection import (
    DEFAULT_CLIMATE,
    DEFAULT_MONTHS,
    MODEL_ID,
    apply_land_mask,
    load_model,
    plot_monthly_map,
    read_json,
    save_dataset,
    valid_projection_mask,
    weekly_feature_matrix,
)
from AIedes.train.counter_protocol import load_protocol


DEFAULT_RESULTS = HERE / "results"
DEFAULT_PROJECTION = HERE / "results" / "europe_projection"
DEFAULT_OUTPUT = HERE / "results" / "europe_projection_ood"


def training_feature_bounds(frame, package, lower_quantile=0.01, upper_quantile=0.99):
    """Calculate robust feature support bounds from the full training rows."""
    features = transform_counter_features(frame, package["candidate"], package["stats"])
    if not np.isfinite(features).all():
        raise ValueError("Non-finite transformed training features")
    return (
        np.quantile(features, lower_quantile, axis=0).astype(np.float32),
        np.quantile(features, upper_quantile, axis=0).astype(np.float32),
        features.shape,
    )


def support_chunk(ds, package, dates, valid_mask, lower, upper):
    """Compute the supported grid mask for a chunk of endpoint dates."""
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    time_index = pd.DatetimeIndex(ds.time.values)
    supported = np.zeros((len(dates), nlat, nlon), dtype=np.float32)
    for position, date in enumerate(dates):
        index = int(time_index.get_loc(date))
        features = weekly_feature_matrix(ds, index, package, valid_mask)
        within = np.isfinite(features).all(axis=1)
        within &= (features >= lower).all(axis=1)
        within &= (features <= upper).all(axis=1)
        flat = np.zeros(nlat * nlon, dtype=np.float32)
        flat[valid_mask] = within.astype(np.float32)
        supported[position] = flat.reshape(nlat, nlon)
    return xr.DataArray(
        supported,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": dates.to_numpy(),
            "latitude": ds.latitude.values,
            "longitude": ds.longitude.values,
        },
    )


def support_grid_parallel(ds, package, dates, lower, upper, workers):
    workers = min(max(int(workers), 1), len(dates))
    valid_mask = valid_projection_mask(ds, dates)
    chunks = [pd.DatetimeIndex(chunk) for chunk in np.array_split(dates, workers) if len(chunk)]
    print(f"Computing OOD support mask with {len(chunks)} workers", flush=True)
    torch.set_num_threads(1)
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {
            pool.submit(support_chunk, ds, package, chunk, valid_mask, lower, upper): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return xr.concat(results, dim="time").sortby("time"), valid_mask


def land_support_summary(support, climate, valid_mask):
    """Return all-grid and land-only support percentages by month."""
    try:
        import regionmask

        land = regionmask.defined_regions.natural_earth_v5_0_0.land_50.mask(
            climate.longitude, climate.latitude
        ).notnull().values
        lakes = (
            ((climate.latitude.values[:, None] >= 59.8) & (climate.latitude.values[:, None] <= 61.8)
             & (climate.longitude.values[None, :] >= 30.5) & (climate.longitude.values[None, :] <= 33.2))
            | ((climate.latitude.values[:, None] >= 61) & (climate.latitude.values[:, None] <= 62.5)
               & (climate.longitude.values[None, :] >= 35) & (climate.longitude.values[None, :] <= 37))
        )
        land &= ~lakes
    except Exception:
        land = np.ones((climate.sizes["latitude"], climate.sizes["longitude"]), dtype=bool)

    valid_2d = valid_mask.reshape(land.shape)
    summaries = {}
    for month in range(1, 13):
        month_support = support.supported.where(support.time.dt.month == month, drop=True).values
        if not month_support.size:
            continue
        valid_all = np.broadcast_to(valid_2d, month_support.shape)
        valid_land = np.broadcast_to(valid_2d & land, month_support.shape)
        summaries[str(month)] = {
            "all_grid_valid_fraction_supported": float(month_support[valid_all > 0].mean()),
            "land_valid_fraction_supported": float(month_support[valid_land > 0].mean()),
            "all_grid_valid_points": int(valid_all.sum()),
            "land_valid_points": int(valid_land.sum()),
        }
    return summaries


def run(args):
    projection = args.projection.resolve()
    source_daily = projection / "predictions_2020_climate_only_daily.nc"
    if not source_daily.is_file():
        raise FileNotFoundError(source_daily)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    masked_daily_path = output / "predictions_2020_climate_only_supported_daily.nc"
    map_png = output / "weekly_rates_map_climate_only_supported.png"
    map_pdf = output / "weekly_rates_map_climate_only_supported.pdf"
    if args.plot_only:
        if not masked_daily_path.is_file():
            raise FileNotFoundError(masked_daily_path)
        with xr.open_dataset(masked_daily_path) as source:
            supported_daily = source.load()
        plot_monthly_map(supported_daily, map_png, map_pdf, months=tuple(args.months))
        print(f"Saved corrected supported-domain map: {map_png}", flush=True)
        return

    config = load_protocol(args.config.resolve())
    with xr.open_dataset(source_daily) as source:
        daily = source.load()
    with xr.open_dataset(args.climate.resolve()) as source:
        climate = source.sortby("time").load()

    model_path = projection / "models" / f"{MODEL_ID}_seed1.pt"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    _, package = load_model(model_path)
    frame = pd.read_pickle(config["weekly_data"]).reset_index(drop=True)
    lower, upper, training_shape = training_feature_bounds(
        frame, package, args.lower_quantile, args.upper_quantile
    )
    dates = pd.DatetimeIndex(daily.time.values)
    support, valid_mask = support_grid_parallel(
        climate, package, dates, lower, upper, args.workers
    )

    support_dataset = xr.Dataset(
        {"supported": support},
        attrs={
            "model_id": MODEL_ID,
            "support_definition": f"all 39 weekly features between training q{args.lower_quantile:g} and q{args.upper_quantile:g}",
            "training_rows": int(training_shape[0]),
            "training_features": int(training_shape[1]),
        },
    )
    support_path = output / "climate_support_mask_2020_daily.nc"
    save_dataset(support_dataset, support_path)
    support_fraction_path = output / "climate_support_fraction_2020_monthly.nc"
    save_dataset(support_dataset.resample(time="1ME").mean(), support_fraction_path)

    supported_daily = daily.where(support_dataset.supported > 0)
    masked_weekly_path = output / "predictions_2020_climate_only_supported_weekly.nc"
    masked_monthly_path = output / "predictions_2020_climate_only_supported_monthly.nc"
    save_dataset(supported_daily, masked_daily_path)
    masked_weekly = supported_daily.resample(
        time="7D", origin=dates[0], label="right", closed="right"
    ).mean()
    masked_monthly = supported_daily.resample(time="1ME").mean()
    save_dataset(masked_weekly, masked_weekly_path)
    save_dataset(masked_monthly, masked_monthly_path)

    plot_monthly_map(supported_daily, map_png, map_pdf, months=tuple(args.months))

    summary = land_support_summary(support_dataset, climate, valid_mask)
    metadata = {
        "model_id": MODEL_ID,
        "source_projection": str(source_daily),
        "support_mask": str(support_path),
        "support_fraction_monthly": str(support_fraction_path),
        "quantiles": [args.lower_quantile, args.upper_quantile],
        "training_rows": int(training_shape[0]),
        "training_features": int(training_shape[1]),
        "feature_lower_bounds": lower.tolist(),
        "feature_upper_bounds": upper.tolist(),
        "outputs": [str(masked_daily_path), str(masked_weekly_path), str(masked_monthly_path)],
        "map_outputs": [str(map_png), str(map_pdf)],
        "monthly_support_summary": summary,
    }
    with (output / "ood_summary.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--climate", type=Path, default=DEFAULT_CLIMATE)
    parser.add_argument("--projection", type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--months", nargs="+", type=int, default=list(DEFAULT_MONTHS))
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
