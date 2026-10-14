"""Render the final autoregressive European projection for the manuscript.

The script reproduces the four-panel June--September layout from the original
map notebook.  It consumes already-computed predictions; it never trains a
model or performs a projection.  The geographic mask follows the notebook's
Natural Earth land mask and lake exclusions and additionally masks the Sea of
Azov, avoiding marine grid cells at the resolution of the climate raster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from AIedes.projection.counter_projection import apply_land_mask


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    ROOT / "experiments" / "counter" / "results"
    / "europe_projection_autoregressive_64_32"
    / "predictions_2020_autoregressive_all_folds_daily.nc"
)
DEFAULT_DESTINATION = ROOT / "paper" / "img"
MONTHS = (6, 7, 8, 9)

# Approximate shoreline polygon of the Sea of Azov, used only to ensure this
# water body is blank in the raster display. Coordinates are longitude, latitude.
SEA_OF_AZOV = np.array([
    (35.45, 45.30), (36.00, 45.62), (36.70, 45.72), (37.35, 45.92),
    (38.05, 46.12), (38.82, 46.38), (39.38, 46.72), (39.43, 47.13),
    (39.00, 47.37), (38.18, 47.40), (37.52, 47.22), (36.94, 46.97),
    (36.38, 46.61), (35.92, 46.08),
])


def coordinate_edges(values: np.ndarray) -> np.ndarray:
    """Return raster-cell edges from monotonically ordered cell centres."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Longitude and latitude coordinates require at least two values")
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate((
        [values[0] - (midpoints[0] - values[0])],
        midpoints,
        [values[-1] + (values[-1] - midpoints[-1])],
    ))


def sea_of_azov_mask(dataset: xr.Dataset) -> xr.DataArray:
    """Return True for grid-cell centres within the Sea of Azov polygon."""
    from matplotlib.path import Path as Polygon

    longitude, latitude = np.meshgrid(
        dataset.longitude.to_numpy(), dataset.latitude.to_numpy(),
    )
    points = np.column_stack((longitude.ravel(), latitude.ravel()))
    contained = Polygon(SEA_OF_AZOV).contains_points(points).reshape(longitude.shape)
    return xr.DataArray(
        contained,
        dims=("latitude", "longitude"),
        coords={"latitude": dataset.latitude, "longitude": dataset.longitude},
    )


def mask_water(dataset: xr.Dataset) -> xr.Dataset:
    """Apply the original land/lake mask plus an explicit Sea of Azov mask."""
    import regionmask

    land = regionmask.defined_regions.natural_earth_v5_0_0.land_50
    land_index = land.mask(dataset.longitude, dataset.latitude)
    land_mask = xr.DataArray(
        land_index.notnull(),
        dims=("latitude", "longitude"),
        coords={"latitude": dataset.latitude, "longitude": dataset.longitude},
    )
    # The two large northern lakes were explicitly excluded in the original
    # notebook because their outlines are coarse at this grid resolution.
    lakes = (
        ((dataset.latitude >= 59.8) & (dataset.latitude <= 61.8)
         & (dataset.longitude >= 30.5) & (dataset.longitude <= 33.2))
        | ((dataset.latitude >= 61.0) & (dataset.latitude <= 62.5)
           & (dataset.longitude >= 35.0) & (dataset.longitude <= 37.0))
    )
    return dataset.where(land_mask & ~lakes & ~sea_of_azov_mask(dataset))


def transparent_colormap(plt, colors, name: str = "flare", n: int = 256):
    """Match the original transparent flare map palette."""
    base = plt.get_cmap(name, n)
    rgba = base(np.linspace(0, 1, n))
    rgba[:, -1] = np.linspace(0, 1, n)
    return colors.ListedColormap(rgba)


def make_autoregressive_map(input_path: Path = DEFAULT_INPUT,
                            destination: Path = DEFAULT_DESTINATION) -> None:
    """Plot June--September means from the saved 25-model AR projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, SymLogNorm
    import seaborn  # noqa: F401  # registers the legacy ``flare`` palette
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    input_path = Path(input_path)
    destination = Path(destination)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    destination.mkdir(parents=True, exist_ok=True)

    with xr.open_dataset(input_path) as source:
        daily = source.load()
    if "weekly_rate" not in daily:
        raise ValueError("Projection must contain a 'weekly_rate' variable")

    # The committed reproducibility package keeps the compact monthly grid;
    # accept both it and the full daily projection used during generation.
    monthly = daily if len(daily.time) <= 12 else daily.resample(time="1ME").mean()
    selected = monthly.where(monthly.time.dt.month.isin(MONTHS), drop=True)
    if selected.sizes.get("time") != len(MONTHS):
        raise ValueError("Expected exactly the June--September monthly panels")
    # Reuse the projection module's land mask. The projection itself already
    # stores unsupported/ocean cells as NaN; this adds the Natural Earth/lake
    # mask when its local cache is available and otherwise preserves that
    # data-level mask without attempting a network download.
    selected, _ = apply_land_mask(selected)

    finite = selected.weekly_rate.to_numpy()
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No finite land predictions remain after masking")

    lon_edges = coordinate_edges(selected.longitude.to_numpy())
    lat_edges = coordinate_edges(selected.latitude.to_numpy())
    extent = [lon_edges.min(), lon_edges.max(), lat_edges.min(), lat_edges.max()]
    vmax = float(selected.weekly_rate.max().item())
    cmap = transparent_colormap(plt, colors)
    norm = SymLogNorm(linthresh=40, linscale=.5, vmin=0, vmax=vmax)

    figure = plt.figure(figsize=(10, 10))
    projection = ccrs.Mercator()
    axes = [figure.add_subplot(2, 2, index + 1, projection=projection)
            for index in range(4)]
    figure.subplots_adjust(
        left=.05, right=.95, top=.95, bottom=.05, wspace=.05, hspace=.05,
    )

    for axis, timestamp, values in zip(
        axes, selected.time.to_numpy(), selected.weekly_rate.to_numpy(),
    ):
        axis.set_extent(extent, crs=ccrs.PlateCarree())
        axis.set_facecolor("white")
        axis.coastlines("50m", linewidth=.5, color="lightgray", zorder=1)
        axis.add_feature(cfeature.BORDERS, linewidth=.5, edgecolor="lightgray", zorder=1)
        axis.pcolormesh(
            lon_edges, lat_edges, values, shading="auto", cmap=cmap, norm=norm,
            transform=ccrs.PlateCarree(), zorder=2,
        )
        axis.set_title(pd.Timestamp(timestamp).strftime("%B"), fontsize=12)
        for spine in axis.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(.3)

    figure.colorbar(
        cm.ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap=cmap),
        ax=axes, orientation="vertical", fraction=.026, pad=.04,
        label="Weekly Egg-laying Rate",
    )
    figure.savefig(destination / "weekly_rates_map.png", dpi=300, bbox_inches="tight")
    figure.savefig(destination / "weekly_rates_map.pdf", bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    make_autoregressive_map(arguments.input, arguments.destination)
