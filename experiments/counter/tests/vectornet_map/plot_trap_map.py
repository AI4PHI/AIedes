"""Map the VectorNet trap locations and summarize their geographic extent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
import contextily as ctx


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "experiments/counter/data/eggs_y_norm_7days_vectorNet.pkl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
ROME_LAT, ROME_LON = 41.9028, 12.4964
NATURALEARTH_PATHS = [
    # Local Natural Earth fixture available in the companion data environment.
    Path("/home/biazzin/git/Aedes-Albopictus-datasets/.conda/aiedes-data/lib/python3.14/site-packages/pyogrio/tests/fixtures/naturalearth_lowres/naturalearth_lowres.shp"),
]


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat1, lon1 = np.radians(ROME_LAT), np.radians(ROME_LON)
    lat2, lon2 = np.radians(lat), np.radians(lon)
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_pickle(args.input)
    locations = frame.groupby("id_trap").agg(
        latitude=("decimalLatitude", "first"),
        longitude=("decimalLongitude", "first"),
        observations=("id_trap", "size"),
    ).reset_index()
    locations["distance_from_rome_km"] = haversine_km(
        locations.latitude.to_numpy(float), locations.longitude.to_numpy(float)
    )

    lon_min, lon_max = locations.longitude.min() - 0.08, locations.longitude.max() + 0.08
    lat_min, lat_max = locations.latitude.min() - 0.08, locations.latitude.max() + 0.08

    # Project coordinates to Web Mercator for a detailed OpenStreetMap base.
    points = gpd.GeoDataFrame(
        locations.copy(),
        geometry=gpd.points_from_xy(locations.longitude, locations.latitude),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)
    rome = gpd.GeoDataFrame(
        {"name": ["Rome"]},
        geometry=gpd.points_from_xy([ROME_LON], [ROME_LAT]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(13, 11))
    sizes = 35 + 12 * locations.observations
    scatter = ax.scatter(
        points.geometry.x, points.geometry.y, s=sizes,
        c=locations.observations, cmap="viridis", alpha=0.88,
        edgecolor="black", linewidth=0.5, zorder=3,
    )
    for _, row in points.iterrows():
        ax.annotate(
            f"{row.id_trap:g}", (row.geometry.x, row.geometry.y),
            xytext=(4, 3), textcoords="offset points", fontsize=7, zorder=4,
        )
    ax.scatter(rome.geometry.x, rome.geometry.y, marker="*", s=240, color="#d62728",
               edgecolor="black", linewidth=0.7, label="Rome", zorder=5)
    # Expand the bounds slightly so nearby roads and settlements remain visible.
    bounds = points.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.16
    pad_y = (bounds[3] - bounds[1]) * 0.16
    ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
    ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)
    # Esri's street tiles provide roads and settlement labels without the
    # OpenStreetMap tile-server policy block encountered in this environment.
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldStreetMap, zoom=9, attribution_size=7)
    ax.set_axis_off()
    ax.set_title("VectorNet trap locations — Rome/Lazio area", fontsize=16, pad=12)
    ax.legend(loc="upper left")
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("Observations per trap")
    fig.subplots_adjust(left=0.02, right=0.91, bottom=0.02, top=0.95)
    fig.savefig(args.output / "vectornet_trap_map.png", dpi=220)
    fig.savefig(args.output / "vectornet_trap_map.pdf")
    plt.close(fig)

    report = {
        "traps": int(len(locations)),
        "observations": int(len(frame)),
        "latitude_range": [float(locations.latitude.min()), float(locations.latitude.max())],
        "longitude_range": [float(locations.longitude.min()), float(locations.longitude.max())],
        "centroid": [float(locations.latitude.mean()), float(locations.longitude.mean())],
        "distance_from_rome_km": {
            "minimum": float(locations.distance_from_rome_km.min()),
            "maximum": float(locations.distance_from_rome_km.max()),
            "mean": float(locations.distance_from_rome_km.mean()),
        },
        "locations": locations.to_dict(orient="records"),
    }
    (args.output / "vectornet_trap_geography.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "locations"}, indent=2))


if __name__ == "__main__":
    main()
