#!/usr/bin/env python3
"""Generate the climate-only European Counter projection and paper map.

The projection model is the selected no-lag WMSLE configuration
``450f9e361ceb``.  Five models are refitted on all AIMSurv rows, each model's
training-fitted normalization statistics are applied to the European climate
grid, and predictions are averaged before the NetCDF files and map are written.

Run from the repository root, for example::

    /home/biazzin/conda-envs/aiedes-env/bin/python \
        python -m AIedes.projection.counter_projection

The large climate grid and projection outputs are local artifacts and are
excluded from Git by the repository's ``.gitignore`` rules.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments" / "counter"

from AIedes.data_loader.counter_data_loader import compute_weekly_stats  # noqa: E402
from AIedes.evaluation.counter_reconstruction import load_model  # noqa: E402
from AIedes.models.counter_models import predict_counter_rates  # noqa: E402
from AIedes.train.counter_train import fit_counter_candidate  # noqa: E402
from AIedes.utils.counter_experiment import FIELDS  # noqa: E402
from AIedes.train.counter_protocol import load_protocol


MODEL_ID = "450f9e361ceb"
DEFAULT_CLIMATE = HERE / "data" / "europe_copernicus_data_2020.nc"
DEFAULT_OUTPUT = HERE / "results" / "europe_projection"
DEFAULT_START = "2020-04-01"
DEFAULT_END = "2020-12-31"
DEFAULT_MONTHS = (6, 7, 8, 9)


def read_json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def selected_climate_only(output: Path) -> dict:
    path = output / "selection" / "selected.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected configuration: {path}")
    candidate = read_json(path)["climate_only"]
    if candidate["id"] != MODEL_ID:
        raise ValueError(
            f"Expected climate-only configuration {MODEL_ID}, found {candidate['id']}"
        )
    if candidate.get("lags"):
        raise ValueError("The European projection requires the climate-only no-lag model")
    if candidate.get("fields") != list(FIELDS):
        raise ValueError(f"Unexpected climate-only fields: {candidate.get('fields')}")
    return candidate


def load_training_frame(config: dict) -> pd.DataFrame:
    frame = pd.read_pickle(config["weekly_data"]).reset_index(drop=True)
    required = {"weeklyRates", *FIELDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training data are missing required columns: {missing}")
    for field in FIELDS:
        values = np.stack(frame[field].to_numpy()).astype(float)
        if values.shape != (len(frame), 90) or not np.isfinite(values).all():
            raise ValueError(f"Invalid training climate arrays for {field}")
    target = frame["weeklyRates"].to_numpy(float)
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError("Invalid training weekly rates")
    return frame


def deployment_model_paths(model_dir: Path, seeds: list[int]) -> list[Path]:
    return [model_dir / f"{MODEL_ID}_seed{seed}.pt" for seed in seeds]


def _train_full_data_worker(payload):
    """Train and save one seed in a separate process."""
    data_path, config, candidate, seed, output_path, threads = payload
    started = time.monotonic()
    frame = pd.read_pickle(data_path).reset_index(drop=True)
    indices = np.arange(len(frame), dtype=int)
    package, history, _ = fit_counter_candidate(
        frame,
        indices,
        None,
        candidate,
        config,
        seed,
        epochs=config["max_epochs"],
        num_threads=threads,
    )
    package.update(
        deployment="europe_climate_only",
        deployment_model_id=MODEL_ID,
        full_data_rows=len(frame),
        full_data_sha256=None,
        history_epochs=len(history),
    )
    path = Path(output_path)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(package, temporary)
    temporary.replace(path)
    return seed, str(path), time.monotonic() - started


def train_full_data_models(
    output: Path,
    config: dict,
    candidate: dict,
    seeds: list[int],
    force: bool = False,
    workers: int = 5,
    threads: int = 4,
) -> list[Path]:
    """Refit and cache the five full-data climate-only packages."""
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = deployment_model_paths(model_dir, seeds)
    if not force and all(path.is_file() for path in paths):
        return paths

    frame = load_training_frame(config)
    pending = [
        (seed, path)
        for seed, path in zip(seeds, paths)
        if force or not path.is_file()
    ]
    for seed, path in zip(seeds, paths):
        if (seed, path) not in pending:
            print(f"Using cached full-data model: {path}", flush=True)
    if not pending:
        return paths

    worker_count = min(max(int(workers), 1), len(pending))
    print(
        f"Training {len(pending)} full-data models in parallel: "
        f"{worker_count} workers × {threads} CPU threads "
        f"({len(frame)} rows, {config['max_epochs']} epochs/model)",
        flush=True,
    )
    jobs = [
        (
            config["weekly_data"],
            config,
            candidate,
            seed,
            str(path),
            int(threads),
        )
        for seed, path in pending
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        futures = {pool.submit(_train_full_data_worker, job): job[2] for job in jobs}
        for future in as_completed(futures):
            seed, path, seconds = future.result()
            print(
                f"Completed full-data climate-only seed={seed}: {path} "
                f"({seconds / 60:.1f} min)",
                flush=True,
            )
    return paths


def validate_grid(ds, start: pd.Timestamp, end: pd.Timestamp):
    required = {
        "2m_temperature_mean",
        "2m_dewpoint_temperature_mean",
        "total_precipitation_sum",
    }
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise ValueError(f"Climate grid is missing variables: {missing}")
    if set(ds.dims) != {"time", "latitude", "longitude"} and not {
        "time", "latitude", "longitude"
    }.issubset(ds.dims):
        raise ValueError(f"Unexpected climate grid dimensions: {dict(ds.sizes)}")
    times = pd.DatetimeIndex(ds.time.values)
    if len(times) < 2 or not (times[1:] - times[:-1] == pd.Timedelta(days=1)).all():
        raise ValueError("Climate grid time coordinate is not a complete daily sequence")
    if times[0] > start - pd.Timedelta(days=89) or times[-1] < end:
        raise ValueError(
            f"Climate grid does not cover the required window "
            f"{(start - pd.Timedelta(days=89)).date()} to {end.date()}"
        )
    for coordinate in ("latitude", "longitude"):
        values = np.asarray(ds[coordinate].values, dtype=float)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(f"Coordinate {coordinate} is not one-dimensional")
        spacing = np.diff(values)
        if not np.allclose(np.abs(spacing), 0.1, atol=1e-6):
            raise ValueError(f"Coordinate {coordinate} is not a 0.1-degree regular grid")
    units = {name: ds[name].attrs.get("units") for name in required}
    expected_units = {
        "2m_temperature_mean": "K",
        "2m_dewpoint_temperature_mean": "K",
        "total_precipitation_sum": "m",
    }
    if units != expected_units:
        raise ValueError(f"Unexpected climate units: {units}; expected {expected_units}")


def weekly_feature_matrix(
    ds,
    index: int,
    package: dict,
    valid_mask: np.ndarray,
    raw_values: list[np.ndarray] | None = None,
):
    """Build exactly the 39 features used by the selected weekly model."""
    candidate = package["candidate"]
    if candidate.get("window") != 90 or not candidate.get("weekly") or candidate.get("lags"):
        raise ValueError(f"Unsupported projection candidate: {candidate}")
    start = index - candidate["window"] + 1
    if raw_values is None:
        raw_values = [
            ds[variable].isel(time=slice(start, index + 1)).values
            for variable in (
                "2m_temperature_mean",
                "2m_dewpoint_temperature_mean",
                "total_precipitation_sum",
            )
        ]
    raw_fields = {}
    for field, values in zip(FIELDS, raw_values):
        if values.shape[0] != 90:
            raise ValueError(f"Expected 90 climate days, got {values.shape[0]}")
        raw_fields[field] = np.asarray(values, dtype=np.float32).reshape(90, -1).T

    pieces = []
    for field in FIELDS:
        stats = package["stats"].get(field)
        if not stats or not np.isfinite(stats["std"]) or stats["std"] == 0:
            raise ValueError(f"Missing/invalid saved normalization for {field}")
        normalized = (raw_fields[field] - stats["mean"]) / stats["std"]
        # The training loader uses keep_partial_week=True, so 90 days become
        # one 6-day oldest block followed by twelve 7-day blocks (13 values).
        weekly = compute_weekly_stats(
            normalized,
            include_mean=True,
            include_max=False,
            include_min=False,
            keep_partial_week=True,
        )[:, :, 0]
        pieces.append(weekly[valid_mask])
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def predict_grid(
    ds,
    packages: list[dict],
    models: list[torch.nn.Module],
    dates: pd.DatetimeIndex,
    global_valid_mask: np.ndarray | None = None,
    progress: bool = True,
):
    nlat, nlon = ds.sizes["latitude"], ds.sizes["longitude"]
    npoints = nlat * nlon
    daily_mean = np.full((len(dates), nlat, nlon), np.nan, dtype=np.float32)
    daily_sd = np.full_like(daily_mean, np.nan)
    time_index = pd.DatetimeIndex(ds.time.values)
    batch_size = 8192

    for date_position, date in enumerate(dates):
        index = int(time_index.get_loc(date))
        start = index - 89
        if global_valid_mask is None:
            valid_mask = np.ones(npoints, dtype=bool)
            for name in (
                "2m_temperature_mean",
                "2m_dewpoint_temperature_mean",
                "total_precipitation_sum",
            ):
                values = ds[name].isel(time=slice(start, index + 1)).values
                valid_mask &= np.isfinite(values).all(axis=0).reshape(-1)
        else:
            valid_mask = global_valid_mask
        if not valid_mask.any():
            raise ValueError(f"No valid European grid points on {date.date()}")

        ensemble = []
        for package, model in zip(packages, models):
            features = weekly_feature_matrix(ds, index, package, valid_mask)
            prediction = np.empty(features.shape[0], dtype=np.float32)
            for left in range(0, len(features), batch_size):
                right = min(left + batch_size, len(features))
                prediction[left:right] = predict_counter_rates(
                    model, features[left:right]
                )
            ensemble.append(prediction)

        ensemble_values = np.stack(ensemble, axis=0)
        mean = ensemble_values.mean(axis=0)
        sd = ensemble_values.std(axis=0, ddof=0)
        daily_flat = np.full(npoints, np.nan, dtype=np.float32)
        sd_flat = np.full(npoints, np.nan, dtype=np.float32)
        daily_flat[valid_mask] = mean
        sd_flat[valid_mask] = sd
        daily_mean[date_position] = daily_flat.reshape(nlat, nlon)
        daily_sd[date_position] = sd_flat.reshape(nlat, nlon)
        if progress and (date_position == 0 or (date_position + 1) % 10 == 0 or date_position + 1 == len(dates)):
            print(f"Projected {date_position + 1}/{len(dates)} dates through {date.date()}", flush=True)

    coords = {
        "time": dates.to_numpy(),
        "latitude": ds.latitude.values,
        "longitude": ds.longitude.values,
    }
    result = xr.Dataset(
        {
            "weekly_rate": (("time", "latitude", "longitude"), daily_mean),
            "ensemble_sd": (("time", "latitude", "longitude"), daily_sd),
        },
        coords=coords,
        attrs={
            "target_units": "eggs_per_week",
            "model": MODEL_ID,
            "model_mode": "climate_only",
            "ensemble_members": str(len(models)),
            "prediction_semantics": "daily endpoints for a 90-day weekly-climate model",
        },
    )
    return result


def valid_projection_mask(ds, dates: pd.DatetimeIndex) -> np.ndarray:
    """Find grid points with complete climate data for the entire projection."""
    time_index = pd.DatetimeIndex(ds.time.values)
    first = int(time_index.get_loc(dates[0])) - 89
    last = int(time_index.get_loc(dates[-1])) + 1
    valid = np.ones(ds.sizes["latitude"] * ds.sizes["longitude"], dtype=bool)
    for name in (
        "2m_temperature_mean",
        "2m_dewpoint_temperature_mean",
        "total_precipitation_sum",
    ):
        values = ds[name].isel(time=slice(first, last)).values
        valid &= np.isfinite(values).all(axis=0).reshape(-1)
    return valid


def predict_grid_parallel(
    ds,
    packages: list[dict],
    models: list[torch.nn.Module],
    dates: pd.DatetimeIndex,
    workers: int,
):
    """Predict independent date chunks concurrently with one Torch thread each."""
    workers = min(max(int(workers), 1), len(dates))
    if workers == 1:
        return predict_grid(ds, packages, models, dates)
    valid_mask = valid_projection_mask(ds, dates)
    chunks = [pd.DatetimeIndex(chunk) for chunk in np.array_split(dates, workers) if len(chunk)]
    print(
        f"Predicting {len(dates)} dates in parallel with {len(chunks)} workers "
        f"(one Torch CPU thread/worker; one CPU left free by default)",
        flush=True,
    )
    torch.set_num_threads(1)
    results = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
        futures = {
            pool.submit(predict_grid, ds, packages, models, chunk, valid_mask, False): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return xr.concat(results, dim="time").sortby("time")


def save_dataset(dataset, path: Path):
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in dataset.data_vars
    }
    dataset.to_netcdf(path, engine="netcdf4", format="NETCDF4", encoding=encoding)


def apply_land_mask(dataset):
    try:
        import regionmask

        # Keep the same Natural Earth land mask and lake exclusions used by
        # the original weekly-rates map notebook.
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
        mask_idx = land.mask(dataset.longitude, dataset.latitude)
        land_mask = xr.DataArray(
            mask_idx.notnull(),
            dims=("latitude", "longitude"),
            coords={"latitude": dataset.latitude, "longitude": dataset.longitude},
        )
        masked = dataset.where(land_mask)

        lakes_mask = (
            (
                (dataset.latitude >= 59.8)
                & (dataset.latitude <= 61.8)
                & (dataset.longitude >= 30.5)
                & (dataset.longitude <= 33.2)
            )
            | (
                (dataset.latitude >= 61)
                & (dataset.latitude <= 62.5)
                & (dataset.longitude >= 35)
                & (dataset.longitude <= 37)
            )
        )
        return masked.where(~lakes_mask), "natural_earth_v5_0_0_land_50_plus_lakes"
    except Exception as exc:  # pragma: no cover - depends on local GIS cache
        print(f"Warning: land mask unavailable ({exc}); plotting the full climate grid", flush=True)
        return dataset, "none"


def plot_monthly_map(dataset, path_png: Path, path_pdf: Path, months=DEFAULT_MONTHS):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import seaborn as sns  # registers the legacy ``flare`` colormap
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    monthly = dataset.resample(time="1ME").mean()
    selected = monthly.where(monthly.time.dt.month.isin(months), drop=True)
    if not selected.time.size:
        raise ValueError("No requested months are present in the projection")
    selected, mask_name = apply_land_mask(selected)
    values = selected.weekly_rate.values
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError("No finite values remain for plotting")
    vmin = 0
    vmax = float(selected.weekly_rate.max().item())
    linthresh = 40
    linscale = 0.5

    def transparent_cmap(base_cmap="Reds", N=256):
        # ``cm.get_cmap`` was removed in recent Matplotlib versions; this is
        # the equivalent call and produces the same registered Seaborn cmap.
        cmap = plt.get_cmap(base_cmap, N)
        colors = cmap(np.linspace(0, 1, N))
        colors[:, -1] = np.linspace(0, 1, N)
        return mcolors.ListedColormap(colors)

    cmap = transparent_cmap("flare")

    # Keep the raster and the basemap in the same geographic coordinate
    # system until Cartopy transforms both of them. Manually projecting the
    # raster with pyproj to EPSG:3857 and then overlaying Cartopy features can
    # introduce a latitude-dependent displacement when their ellipsoid
    # definitions differ.
    lons = selected.longitude.values
    lats = selected.latitude.values

    def to_edges(values):
        midpoints = (values[:-1] + values[1:]) / 2.0
        first = values[0] - (midpoints[0] - values[0])
        last = values[-1] + (values[-1] - midpoints[-1])
        return np.concatenate(([first], midpoints, [last]))

    lon_edges = to_edges(lons)
    lat_edges = to_edges(lats)
    web_mercator = ccrs.Mercator()
    geographic_extent = [
        float(lon_edges.min()),
        float(lon_edges.max()),
        float(lat_edges.min()),
        float(lat_edges.max()),
    ]

    if selected.time.size != 4:
        raise ValueError("The legacy map layout requires exactly four selected months")
    fig = plt.figure(figsize=(10, 10))
    axes = [fig.add_subplot(2, 2, index + 1, projection=web_mercator) for index in range(4)]
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05)

    image = None
    norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
    for axis, timestamp, values_2d in zip(
        axes, selected.time.values, selected.weekly_rate.values
    ):
        axis.set_extent(geographic_extent, crs=ccrs.PlateCarree())
        axis.set_facecolor("white")
        axis.patch.set_facecolor("white")
        axis.coastlines("50m", linewidth=0.5, color="lightgray", zorder=1)
        axis.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="lightgray", zorder=1)
        image = axis.pcolormesh(
            lon_edges,
            lat_edges,
            values_2d,
            shading="auto",
            cmap=cmap,
            norm=norm,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
        axis.set_title(pd.Timestamp(timestamp).strftime("%B"), fontsize=12)
        for spine in axis.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.3)

    # Preserve the original figure's linear colorbar, even though the map
    # itself uses SymLogNorm.
    norm_linear = Normalize(vmin=vmin, vmax=vmax)
    mappable = cm.ScalarMappable(norm=norm_linear, cmap=cmap)
    fig.colorbar(
        mappable,
        ax=axes,
        orientation="vertical",
        fraction=0.026,
        pad=0.04,
        label="Weekly Egg-laying Rate",
    )
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        daily_path = output / "predictions_2020_climate_only_daily.nc"
        if not daily_path.is_file():
            raise FileNotFoundError(f"Missing projection file for --plot-only: {daily_path}")
        with xr.open_dataset(daily_path) as source:
            daily = source.load()
        plot_monthly_map(
            daily,
            output / "weekly_rates_map_climate_only.png",
            output / "weekly_rates_map_climate_only.pdf",
            months=tuple(args.months),
        )
        print(f"Saved map: {output / 'weekly_rates_map_climate_only.png'}", flush=True)
        return output
    config = load_protocol(args.config.resolve())
    selected_output = args.results.resolve()
    candidate = selected_climate_only(selected_output)
    seeds = [int(seed) for seed in config["final_seeds"]]
    model_paths = deployment_model_paths(output / "models", seeds)
    if args.no_train:
        missing = [str(path) for path in model_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "--no-train requested, but full-data climate-only models are missing: "
                + ", ".join(missing)
            )
    else:
        model_paths = train_full_data_models(
            output,
            config,
            candidate,
            seeds,
            force=args.force_train,
            workers=args.workers,
            threads=args.threads,
        )
    loaded = [load_model(path) for path in model_paths]
    models = [model for model, _ in loaded]
    packages = [package for _, package in loaded]
    if any(package["candidate"] != candidate for package in packages):
        raise ValueError("A cached projection model does not match selected climate-only configuration")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    if end < start:
        raise ValueError("Projection end date precedes start date")
    climate_path = args.climate.resolve()
    if not climate_path.is_file():
        raise FileNotFoundError(climate_path)
    with xr.open_dataset(climate_path) as source:
        ds = source.sortby("time").load()
    validate_grid(ds, start, end)
    all_dates = pd.DatetimeIndex(ds.time.values)
    dates = pd.date_range(start, end, freq="D")
    if not dates.isin(all_dates).all():
        raise ValueError("Requested projection dates are missing from the climate grid")

    daily = predict_grid_parallel(ds, packages, models, dates, args.prediction_workers)
    daily_path = output / "predictions_2020_climate_only_daily.nc"
    weekly_path = output / "predictions_2020_climate_only_weekly.nc"
    monthly_path = output / "predictions_2020_climate_only_monthly.nc"
    save_dataset(daily, daily_path)
    weekly = daily.resample(time="7D", origin=dates[0], label="right", closed="right").mean()
    monthly = daily.resample(time="1ME").mean()
    save_dataset(weekly, weekly_path)
    save_dataset(monthly, monthly_path)
    print(f"Saved daily predictions: {daily_path}", flush=True)
    print(f"Saved weekly predictions: {weekly_path}", flush=True)
    print(f"Saved monthly predictions: {monthly_path}", flush=True)

    if not args.no_plot:
        plot_monthly_map(
            daily,
            output / "weekly_rates_map_climate_only.png",
            output / "weekly_rates_map_climate_only.pdf",
            months=tuple(args.months),
        )
        print(f"Saved map: {output / 'weekly_rates_map_climate_only.png'}", flush=True)

    write_json(
        output / "projection_metadata.json",
        {
            "model_id": MODEL_ID,
            "candidate": candidate,
            "seeds": seeds,
            "model_paths": [str(path) for path in model_paths],
            "climate_path": str(climate_path),
            "start": str(start.date()),
            "end": str(end.date()),
            "months_plotted": list(args.months),
            "mode": "climate_only_no_recursive_lags",
            "normalization": "saved full-data training statistics per ensemble member",
            "outputs": [str(path) for path in (daily_path, weekly_path, monthly_path)],
        },
    )
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--climate", type=Path, default=DEFAULT_CLIMATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--months", nargs="+", type=int, default=list(DEFAULT_MONTHS))
    parser.add_argument("--force-train", action="store_true", help="Refit the five full-data models")
    parser.add_argument("--no-train", action="store_true", help="Require cached full-data models")
    parser.add_argument("--workers", type=int, default=5, help="Parallel full-data training workers")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads per training worker")
    parser.add_argument(
        "--prediction-workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel prediction workers; defaults to CPU count minus one",
    )
    parser.add_argument("--plot-only", action="store_true", help="Plot an existing daily projection")
    parser.add_argument("--no-plot", action="store_true", help="Write NetCDF only")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
