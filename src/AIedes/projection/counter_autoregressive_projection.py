#!/usr/bin/env python3
"""Project Europe recursively with a five-fold, five-seed model ensemble.

Dates remain sequential within each grid-cell trajectory. Independent latitude
bands may be projected in separate processes, and each ensemble member uses
its own predicted values at 7 and 14 days as lag inputs.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments" / "counter"

from AIedes.evaluation.counter_reconstruction import load_model  # noqa: E402
from AIedes.models.counter_models import (  # noqa: E402
    counter_candidate_model,
    predict_counter_rates,
)
from AIedes.utils.counter_experiment import FIELDS  # noqa: E402
from AIedes.projection.counter_projection import (
    DEFAULT_CLIMATE,
    DEFAULT_END,
    DEFAULT_MONTHS,
    DEFAULT_START,
    plot_monthly_map,
    read_json,
    save_dataset,
    valid_projection_mask,
    validate_grid,
)


MODEL_ID = "8bec14a3ec4c"
DEFAULT_RESULTS = HERE / "results"
DEFAULT_OUTPUT = HERE / "results" / "europe_projection_autoregressive_all_folds"


def load_projection_model(path: Path, jobs_root: Path | None = None):
    """Load either legacy deployment packages or new AR job packages."""
    if jobs_root is None:
        return load_model(path)
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = counter_candidate_model(package["candidate"], len(package["feature_names"]))
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model, package


def all_fold_model_paths(results: Path, jobs_root: Path | None = None,
                         architecture: str = "64_32") -> list[Path]:
    """Resolve all 5 folds x 5 seeds, without metric-based filtering."""
    paths = []
    for fold in range(1, 6):
        for seed in range(1, 6):
            if jobs_root is None:
                path = (results / "final_models" / f"outer{fold}"
                        / f"{MODEL_ID}_seed{seed}" / "model.pt")
            else:
                path = (jobs_root / f"{architecture}_seed{seed}_fold{fold}"
                        / f"fold{fold}" / "model.pt")
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    return paths


def validate_packages(packages: list[dict], results: Path,
                      jobs_root: Path | None = None,
                      architecture: str = "64_32") -> dict:
    if jobs_root is not None:
        expected_widths = [int(value) for value in architecture.split("_")]
        reference = packages[0]["candidate"]
        if reference.get("widths") != expected_widths:
            raise ValueError(f"Unexpected autoregressive architecture: {reference}")
        for package in packages:
            candidate = package["candidate"]
            if (candidate.get("window") != 90 or candidate.get("weekly")
                    or not candidate.get("lags")
                    or candidate.get("lag_order", 2) != 2):
                raise ValueError(f"Unsupported autoregressive candidate: {candidate}")
            if set(package.get("feature_slices", {})) != {"p1zn", "p2zn"}:
                raise ValueError("Expected value-only p1/p2 autoregressive features")
        return reference
    selected_path = results / "selection" / "selected.json"
    selected = read_json(selected_path)["main"]
    if selected["id"] != MODEL_ID:
        raise ValueError(f"Expected selected main model {MODEL_ID}, found {selected['id']}")
    for package in packages:
        candidate = package["candidate"]
        if candidate != selected:
            raise ValueError(f"Unexpected lagged candidate: {candidate}")
        if (
            candidate.get("window") != 90
            or candidate.get("weekly")
            or not candidate.get("lags")
            or candidate.get("lag_order", 2) != 2
            or candidate.get("widths") != [64]
        ):
            raise ValueError(f"Unsupported autoregressive candidate: {candidate}")
        if package.get("autonomous_lag_days")[:2] != [7, 14]:
            raise ValueError("Expected autonomous lags of 7 and 14 days")
    return selected


def stacked_parameters(packages: list[dict]) -> dict[str, torch.Tensor]:
    """Stack identical feed-forward networks along an ensemble dimension."""
    keys = [
        "model.0.weight",
        "model.0.bias",
        "model.1.weight",
        "model.1.bias",
        "model.1.running_mean",
        "model.1.running_var",
        "fc.weight",
        "fc.bias",
    ]
    if "model.4.weight" in packages[0]["state_dict"]:
        keys[4:4] = [
            "model.4.weight",
            "model.4.bias",
            "model.5.weight",
            "model.5.bias",
            "model.5.running_mean",
            "model.5.running_var",
        ]
    return {
        key: torch.stack([package["state_dict"][key].detach().cpu() for package in packages])
        for key in keys
    }


def normalization_arrays(packages: list[dict]):
    climate_mean = np.asarray(
        [[package["stats"][field]["mean"] for field in FIELDS] for package in packages],
        dtype=np.float32,
    )
    climate_std = np.asarray(
        [[package["stats"][field]["std"] for field in FIELDS] for package in packages],
        dtype=np.float32,
    )
    lag_mean = np.asarray(
        [
            [package["stats"][f"prev{k}_rates"]["mean"] for k in (1, 2)]
            for package in packages
        ],
        dtype=np.float32,
    )
    lag_std = np.asarray(
        [
            [package["stats"][f"prev{k}_rates"]["std"] for k in (1, 2)]
            for package in packages
        ],
        dtype=np.float32,
    )
    if not all(np.isfinite(value).all() and (value > 0).all() for value in (climate_std, lag_std)):
        raise ValueError("Invalid saved normalization statistics")
    return climate_mean, climate_std, lag_mean, lag_std


def batched_forward(parameters: dict[str, torch.Tensor], features: np.ndarray) -> np.ndarray:
    """Evaluation-mode forward pass for the stacked ensemble."""
    x = torch.from_numpy(features)
    hidden = torch.bmm(x, parameters["model.0.weight"].transpose(1, 2))
    hidden += parameters["model.0.bias"][:, None, :]
    hidden = (
        (hidden - parameters["model.1.running_mean"][:, None, :])
        / torch.sqrt(parameters["model.1.running_var"][:, None, :] + 1e-5)
    )
    hidden = hidden * parameters["model.1.weight"][:, None, :]
    hidden += parameters["model.1.bias"][:, None, :]
    hidden = torch.relu(hidden)
    if "model.4.weight" in parameters:
        hidden = torch.bmm(hidden, parameters["model.4.weight"].transpose(1, 2))
        hidden += parameters["model.4.bias"][:, None, :]
        hidden = (hidden - parameters["model.5.running_mean"][:, None, :]) / torch.sqrt(
            parameters["model.5.running_var"][:, None, :] + 1e-5
        )
        hidden = hidden * parameters["model.5.weight"][:, None, :]
        hidden += parameters["model.5.bias"][:, None, :]
        hidden = torch.relu(hidden)
    output = torch.bmm(hidden, parameters["fc.weight"].transpose(1, 2)).squeeze(-1)
    output += parameters["fc.bias"]
    return functional.softplus(output).numpy()


def raw_climate_window(ds, index: int, valid_mask: np.ndarray) -> np.ndarray:
    pieces = []
    for variable in (
        "2m_temperature_mean",
        "2m_dewpoint_temperature_mean",
        "total_precipitation_sum",
    ):
        values = ds[variable].isel(time=slice(index - 89, index + 1)).values
        if values.shape[0] != 90:
            raise ValueError(f"Expected 90 climate days at index {index}")
        pieces.append(np.asarray(values, dtype=np.float32).reshape(90, -1).T[valid_mask])
    raw = np.concatenate(pieces, axis=1)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite climate value passed to autoregressive projection")
    return raw


def verify_batched_forward(models, packages, parameters, climate_mean, climate_std):
    """Check the vectorized implementation against ordinary model inference."""
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(8, 270)).astype(np.float32)
    count = len(packages)
    features = np.empty((count, len(raw), 274), dtype=np.float32)
    for model_index in range(count):
        for field_index in range(3):
            left, right = field_index * 90, (field_index + 1) * 90
            features[model_index, :, left:right] = (
                raw[:, left:right] - climate_mean[model_index, field_index]
            ) / climate_std[model_index, field_index]
        features[model_index, :, 270:] = 0.0
    actual = batched_forward(parameters, features)
    for model_index, model in enumerate(models):
        expected = predict_counter_rates(model, features[model_index])
        if not np.allclose(actual[model_index], expected, rtol=2e-5, atol=2e-5):
            raise ValueError(f"Batched inference differs for ensemble member {model_index}")


def predict_recursive_grid(
    ds,
    models,
    packages,
    dates: pd.DatetimeIndex,
    batch_size: int,
):
    """Advance every ensemble member's independent recursive grid trajectory."""
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    npoints = nlat * nlon
    member_count = len(packages)
    valid_mask = valid_projection_mask(ds, dates)
    valid_indices = np.flatnonzero(valid_mask)
    nvalid = len(valid_indices)
    if not nvalid:
        raise ValueError("No valid European grid points")

    climate_mean, climate_std, lag_mean, lag_std = normalization_arrays(packages)
    feature_width = len(packages[0]["feature_names"])
    parameters = stacked_parameters(packages)
    if feature_width == 274:
        verify_batched_forward(models, packages, parameters, climate_mean, climate_std)

    daily_mean = np.full((len(dates), nlat, nlon), np.nan, dtype=np.float32)
    daily_sd = np.full_like(daily_mean, np.nan)
    # Only the latest 14 independent predictions per model are needed.
    history = np.empty((member_count, 14, nvalid), dtype=np.float32)
    time_index = pd.DatetimeIndex(ds.time.values)

    for date_position, date in enumerate(dates):
        climate = raw_climate_window(ds, int(time_index.get_loc(date)), valid_mask)
        member_sum = np.zeros(nvalid, dtype=np.float64)
        member_sum_sq = np.zeros(nvalid, dtype=np.float64)

        for left in range(0, nvalid, batch_size):
            right = min(left + batch_size, nvalid)
            raw = climate[left:right]
            features = np.empty((member_count, right - left, feature_width), dtype=np.float32)
            for field_index in range(3):
                source_left, source_right = field_index * 90, (field_index + 1) * 90
                features[:, :, source_left:source_right] = (
                    raw[None, :, source_left:source_right]
                    - climate_mean[:, None, field_index, None]
                ) / climate_std[:, None, field_index, None]

            features[:, :, 270:] = 0.0
            if date_position >= 7:
                lag = history[:, (date_position - 7) % 14, left:right]
                features[:, :, 270] = (lag - lag_mean[:, 0, None]) / lag_std[:, 0, None]
            if date_position >= 14:
                lag = history[:, (date_position - 14) % 14, left:right]
                if feature_width == 274:
                    features[:, :, 272] = (lag - lag_mean[:, 1, None]) / lag_std[:, 1, None]
                    features[:, :, 273] = 1.0
                else:
                    features[:, :, 271] = (lag - lag_mean[:, 1, None]) / lag_std[:, 1, None]

            predictions = batched_forward(parameters, features)
            if not np.isfinite(predictions).all() or (predictions < 0).any():
                raise ValueError(f"Invalid recursive prediction on {date.date()}")
            history[:, date_position % 14, left:right] = predictions
            member_sum[left:right] = predictions.sum(axis=0, dtype=np.float64)
            member_sum_sq[left:right] = np.square(predictions, dtype=np.float64).sum(
                axis=0, dtype=np.float64
            )

        mean = (member_sum / member_count).astype(np.float32)
        variance = np.maximum(member_sum_sq / member_count - np.square(mean), 0.0)
        sd = np.sqrt(variance).astype(np.float32)
        mean_flat = np.full(npoints, np.nan, dtype=np.float32)
        sd_flat = np.full(npoints, np.nan, dtype=np.float32)
        mean_flat[valid_indices] = mean
        sd_flat[valid_indices] = sd
        daily_mean[date_position] = mean_flat.reshape(nlat, nlon)
        daily_sd[date_position] = sd_flat.reshape(nlat, nlon)
        if date_position == 0 or (date_position + 1) % 10 == 0 or date_position + 1 == len(dates):
            print(
                f"Projected {date_position + 1}/{len(dates)} dates through {date.date()}",
                flush=True,
            )

    coords = {
        "time": dates.to_numpy(),
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values,
    }
    return xr.Dataset(
        {
            "weekly_rate": (("time", "latitude", "longitude"), daily_mean),
            "ensemble_sd": (("time", "latitude", "longitude"), daily_sd),
        },
        coords=coords,
        attrs={
            "target_units": "eggs_per_week",
            "model": MODEL_ID,
            "model_mode": "autoregressive_all_outer_folds_and_seeds",
            "ensemble_members": str(member_count),
            "ensemble_composition": "5 trap-blocked folds x 5 training seeds; no score filtering",
            "recursive_lag_days": "7,14",
            "startup": "missing lag values and availability flags for first 7/14 days",
            "prediction_semantics": "daily endpoints for a 90-day daily-climate recursive model",
        },
    )


def distribution_summary(dataset: xr.Dataset) -> dict:
    values = np.asarray(dataset.weekly_rate.values, dtype=float)
    finite = values[np.isfinite(values)]
    uncertainty = np.asarray(dataset.ensemble_sd.values, dtype=float)
    uncertainty = uncertainty[np.isfinite(uncertainty)]
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "median": float(np.median(finite)),
        "mean": float(finite.mean()),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
        "ensemble_sd_mean": float(uncertainty.mean()),
        "ensemble_sd_p95": float(np.quantile(uncertainty, 0.95)),
    }


def _spatial_worker(payload):
    """Project one contiguous latitude band in an independent process."""
    (
        climate_path,
        model_paths,
        results_path,
        jobs_root,
        architecture,
        start,
        end,
        batch_size,
        threads,
        latitude_start,
        latitude_stop,
    ) = payload
    torch.set_num_threads(int(threads))
    jobs_root_path = Path(jobs_root) if jobs_root else None
    loaded = [load_projection_model(Path(path), jobs_root_path) for path in model_paths]
    models = [item[0] for item in loaded]
    packages = [item[1] for item in loaded]
    validate_packages(packages, Path(results_path), jobs_root_path, architecture)
    with xr.open_dataset(climate_path) as source:
        climate = (
            source.sortby("time")
            .isel(latitude=slice(latitude_start, latitude_stop))
            .load()
        )
    dates = pd.date_range(start, end, freq="D")
    projected = predict_recursive_grid(climate, models, packages, dates, int(batch_size))
    return latitude_start, projected


def run(args):
    results = args.results.resolve()
    output = args.output.resolve()
    jobs_root = args.jobs_root.resolve() if args.jobs_root else None
    output.mkdir(parents=True, exist_ok=True)
    daily_path = output / "predictions_2020_autoregressive_all_folds_daily.nc"
    weekly_path = output / "predictions_2020_autoregressive_all_folds_weekly.nc"
    monthly_path = output / "predictions_2020_autoregressive_all_folds_monthly.nc"
    map_png = output / "weekly_rates_map_autoregressive_all_folds.png"
    map_pdf = output / "weekly_rates_map_autoregressive_all_folds.pdf"

    if args.plot_only:
        if not daily_path.is_file():
            raise FileNotFoundError(daily_path)
        with xr.open_dataset(daily_path) as source:
            daily = source.load()
        plot_monthly_map(daily, map_png, map_pdf, months=tuple(args.months))
        return output

    model_paths = all_fold_model_paths(results, jobs_root, args.architecture)
    # Candidate metadata are identical across the required artifacts; workers
    # perform the complete package validation before inference.
    _, reference_package = load_projection_model(model_paths[0], jobs_root)
    candidate = validate_packages([reference_package], results, jobs_root, args.architecture)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if end < start:
        raise ValueError("Projection end date precedes start date")
    dates = pd.date_range(start, end, freq="D")
    climate_path = args.climate.resolve()
    if not climate_path.is_file():
        raise FileNotFoundError(climate_path)
    with xr.open_dataset(climate_path) as source:
        climate_header = source.sortby("time")
        validate_grid(climate_header, start, end)
        latitude_count = climate_header.sizes["latitude"]

    spatial_workers = min(max(1, int(args.spatial_workers)), latitude_count)
    if spatial_workers == 1:
        torch.set_num_threads(max(1, int(args.threads)))
        loaded = [load_projection_model(path, jobs_root) for path in model_paths]
        models = [item[0] for item in loaded]
        packages = [item[1] for item in loaded]
        validate_packages(packages, results, jobs_root, args.architecture)
        with xr.open_dataset(climate_path) as source:
            climate = source.sortby("time").load()
        print(
            f"Projecting {len(dates)} dates with all {len(models)} fold/seed models "
            f"using {torch.get_num_threads()} Torch threads",
            flush=True,
        )
        daily = predict_recursive_grid(climate, models, packages, dates, args.batch_size)
    else:
        latitude_chunks = [
            chunk for chunk in np.array_split(np.arange(latitude_count), spatial_workers) if len(chunk)
        ]
        jobs = [
            (
                str(climate_path),
                [str(path) for path in model_paths],
                str(results),
                str(jobs_root) if jobs_root else "",
                args.architecture,
                str(start.date()),
                str(end.date()),
                int(args.batch_size),
                int(args.threads),
                int(chunk[0]),
                int(chunk[-1]) + 1,
            )
            for chunk in latitude_chunks
        ]
        print(
            f"Projecting {len(dates)} dates with all 25 fold/seed models in "
            f"{len(jobs)} latitude workers x {args.threads} Torch threads",
            flush=True,
        )
        bands = []
        with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(_spatial_worker, job) for job in jobs]
            for future in as_completed(futures):
                latitude_start, band = future.result()
                bands.append((latitude_start, band))
                print(f"Completed latitude band beginning at index {latitude_start}", flush=True)
        daily = xr.concat([band for _, band in sorted(bands)], dim="latitude")
    save_dataset(daily, daily_path)
    weekly = daily.resample(time="7D", origin=dates[0], label="right", closed="right").mean()
    monthly = daily.resample(time="1ME").mean()
    save_dataset(weekly, weekly_path)
    save_dataset(monthly, monthly_path)
    if not args.no_plot:
        plot_monthly_map(daily, map_png, map_pdf, months=tuple(args.months))

    metadata = {
        "model_id": MODEL_ID if jobs_root is None else f"autoregressive_{args.architecture}",
        "model_source": str(jobs_root) if jobs_root is not None else "final_models",
        "candidate": candidate,
        "model_mode": "autoregressive_all_outer_folds_and_seeds",
        "models": [str(path) for path in model_paths],
        "model_count": len(model_paths),
        "selection_filter": None,
        "climate_path": str(climate_path),
        "start": str(start.date()),
        "end": str(end.date()),
        "recursive_lag_days": [7, 14],
        "startup": "missing flags until each recursive lag becomes available",
        "distribution": distribution_summary(daily),
        "outputs": [str(daily_path), str(weekly_path), str(monthly_path)],
        "map_outputs": [str(map_png), str(map_pdf)] if not args.no_plot else [],
    }
    with (output / "projection_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    print(json.dumps(metadata["distribution"], indent=2, sort_keys=True), flush=True)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--jobs-root", type=Path, default=None,
        help="Use completed fold/seed autoregressive job outputs instead of final_models.",
    )
    parser.add_argument(
        "--architecture", default="64_32",
        help="Underscore-separated hidden widths when --jobs-root is used.",
    )
    parser.add_argument("--climate", type=Path, default=DEFAULT_CLIMATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--months", nargs="+", type=int, default=list(DEFAULT_MONTHS))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--spatial-workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
