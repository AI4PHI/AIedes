#!/usr/bin/env python3
"""Render the random or NUTS3 classifier comparison from saved predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS = ROOT / "experiments/classifier/paper_results"
RESULT_LABELS = ("True positive", "True negative", "False positive", "False negative")
RESULT_COLORS = ("#2b8cbe", "#a6bddb", "#fd8d3c", "#de2d26")
MODEL_COLUMNS = (
    ("logistic_probability", "a. Simplest neural network\n(equiv. logistic regression)", "logistic"),
    ("nn_probability", "b. AIedes-classifier", "aiedes"),
)


def local_natural_earth_feature(name: str, linewidth: float):
    """Return a cached Natural Earth line feature without downloading data."""
    path = (
        Path(cartopy.config["data_dir"])
        / "shapefiles/natural_earth/cultural"
        / f"ne_50m_{name}.shp"
    )
    if not path.exists():
        raise FileNotFoundError(f"Cached Natural Earth shapefile not found: {path}")
    return cfeature.ShapelyFeature(
        shapereader.Reader(str(path)).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=linewidth,
    )


def local_coastline_feature(linewidth: float):
    """Return the cached Natural Earth coastline without downloading data."""
    path = (
        Path(cartopy.config["data_dir"])
        / "shapefiles/natural_earth/physical/ne_50m_coastline.shp"
    )
    if not path.exists():
        raise FileNotFoundError(f"Cached Natural Earth shapefile not found: {path}")
    return cfeature.ShapelyFeature(
        shapereader.Reader(str(path)).geometries(),
        ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="black",
        linewidth=linewidth,
    )


def result_categories(table: pd.DataFrame, probability_column: str) -> pd.Series:
    observed = table["target_presence"].to_numpy(dtype=int)
    predicted = (table[probability_column].to_numpy(dtype=float) >= 0.5).astype(int)
    return pd.Series(
        np.select(
            [
                (observed == 1) & (predicted == 1),
                (observed == 0) & (predicted == 0),
                (observed == 0) & (predicted == 1),
                (observed == 1) & (predicted == 0),
            ],
            [0, 1, 2, 3],
            default=-1,
        ),
        index=table.index,
    )


def plot_map_axis(
    axis,
    table: pd.DataFrame,
    probability_column: str,
    title: str,
    extent: tuple[float, float, float, float],
    title_y: float,
) -> None:
    axis.set_extent(extent, crs=ccrs.PlateCarree())
    axis.set_anchor("W")
    axis.set_facecolor("white")
    axis.add_feature(local_coastline_feature(0.5), zorder=1)
    axis.add_feature(
        local_natural_earth_feature("admin_0_boundary_lines_land", 0.4), zorder=1
    )
    categories = result_categories(table, probability_column)
    for category, color in enumerate(RESULT_COLORS):
        subset = table.loc[categories == category]
        axis.scatter(
            subset["longitude"],
            subset["latitude"],
            s=2.0,
            color=color,
            alpha=0.85,
            linewidths=0,
            transform=ccrs.PlateCarree(),
            zorder=2,
        )
    axis.text(
        0.02,
        title_y,
        title,
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black", pad=5),
        zorder=3,
    )


def plot_metrics(axis, metrics: pd.DataFrame) -> None:
    metric_columns = ["accuracy", "sensitivity", "specificity", "csi"]
    values = {
        model: metrics.set_index("model").loc[model, metric_columns].to_numpy(dtype=float) * 100
        for model in ("logistic", "aiedes")
    }
    x = np.arange(2)
    width = 0.2
    axis.bar(x - width, [values["logistic"][0], values["aiedes"][0]], width,
             label="Accuracy", color="#4daf4a")
    axis.bar(x, [values["logistic"][1], values["aiedes"][1]], width,
             label="Sensitivity", color="#2b8cbe")
    axis.bar(x + width, [values["logistic"][2], values["aiedes"][2]], width,
             label="Specificity", color="#a6bddb")
    axis.bar(x + 2 * width, [values["logistic"][3], values["aiedes"][3]], width,
             label="CSI", color="#984ea3")
    axis.yaxis.set_label_position("right")
    axis.yaxis.set_ticks_position("right")
    axis.set_xticks(x + 0.1, ["Log. Reg.", "AIedes"])
    lower = 50 if metrics[metric_columns].to_numpy().min() < 0.60 else 60
    axis.set_ylim(lower, 105)
    axis.set_yticks(np.arange(lower, 101, 10))
    axis.legend(
        loc="upper left",
        fontsize=12,
        frameon=False,
        title=r"$\bf{c. Metrics}$",
        title_fontsize=14,
    )
    axis.tick_params(axis="x", labelsize=13)
    axis.tick_params(axis="y", labelsize=12)


def input_paths(results_dir: Path, evaluation: str) -> tuple[Path, Path]:
    if evaluation == "random":
        return (
            results_dir / "main_random_oof_map_predictions.csv",
            results_dir / "main_random_oof_map_metrics_pooled.csv",
        )
    return (
        results_dir / "nuts3_held_region_predictions.csv",
        results_dir / "nuts3_metrics_pooled.csv",
    )


def default_output(evaluation: str) -> Path:
    filename = "comparison_lr_nn.png" if evaluation == "random" else "comparison_lr_nn_nuts3.png"
    return ROOT / "paper/img" / filename


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", choices=("random", "nuts3"), required=True)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    output = arguments.output or default_output(arguments.evaluation)

    prediction_path, metrics_path = input_paths(arguments.results_dir, arguments.evaluation)
    table = pd.read_csv(prediction_path)
    metrics = pd.read_csv(metrics_path)
    required = {
        "latitude", "longitude", "target_presence",
        "logistic_probability", "nn_probability",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Prediction table is missing: {', '.join(sorted(missing))}")
    if set(metrics["model"]) != {"logistic", "aiedes"}:
        raise ValueError("Pooled metric table must contain logistic and aiedes rows.")

    longitude = table["longitude"].to_numpy(dtype=float)
    latitude = table["latitude"].to_numpy(dtype=float)
    pad = 0.5
    extent = (
        longitude.min() - pad,
        longitude.max() + pad,
        latitude.min() - pad,
        latitude.max() + pad,
    )
    projection = ccrs.Mercator()
    figure = plt.figure(figsize=(14, 6), constrained_layout=False)
    grid = figure.add_gridspec(1, 3, width_ratios=(3, 3, 1), wspace=-0.20)
    axes = [figure.add_subplot(grid[0, index], projection=projection) for index in (0, 1)]
    metric_axis = figure.add_subplot(grid[0, 2])
    for index, (axis, (column, title, _)) in enumerate(zip(axes, MODEL_COLUMNS)):
        plot_map_axis(axis, table, column, title, extent, 0.90 if index == 0 else 0.94)
    plot_metrics(metric_axis, metrics)

    legend_handles = [
        Patch(color=color, label=label)
        for label, color in zip(RESULT_LABELS, RESULT_COLORS)
    ]
    figure.legend(
        handles=legend_handles,
        bbox_to_anchor=(0.025, 0.66),
        loc="center left",
        frameon=True,
        fontsize=12,
    )
    for axis in (*axes, metric_axis):
        for spine in axis.spines.values():
            spine.set_edgecolor("lightgray")
            spine.set_linewidth(0.5)
    figure.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=400, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
