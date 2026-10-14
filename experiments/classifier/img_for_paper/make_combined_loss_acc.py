#!/usr/bin/env python3
"""Create the current Classifier training and model-selection figure.

The figure is built from the reproducible outputs in
``experiments/classifier/paper_results``. It does not use archived notebook
results or the former NUTS3 exploratory result files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = ROOT / "experiments" / "classifier" / "paper_results"
DEFAULT_OUTPUT = ROOT / "paper" / "img" / "combined_loss_acc.pdf"


def load_current_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the current training history and cross-validation summary."""
    history_path = results_dir / "selected_configuration_loss_history.csv"
    summary_path = results_dir / "inner_validation_summary.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing current loss history: {history_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing current selection summary: {summary_path}")
    return pd.read_csv(history_path), pd.read_csv(summary_path)


def make_figure(history: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    """Render the two-panel current Classifier selection figure."""
    temperature_label = r"Temperature ($T_{12m}$)"
    temperature_precipitation_label = (
        r"Temperature + precipitation" "\n" r"($T_{12m}$, $P_{12m}$)"
    )
    input_labels = {"temp": temperature_label, "temp_tp": temperature_precipitation_label}
    palette = {"temp": "#1f77b4", "temp_tp": "#ff7f0e"}
    markers = {0: "o", 1: "s", 2: "D", 3: "X", 4: "P"}

    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.2,
        rc={
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
            "grid.linewidth": 0.5,
            "grid.color": "0.8",
        },
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(
        history["epoch"],
        history["train_loss"],
        label="Training loss",
        color="#2E86AB",
        linewidth=2.0,
        alpha=0.85,
    )
    axes[0].plot(
        history["epoch"],
        history["validation_loss"],
        label="Validation loss",
        color="#F24236",
        linewidth=2.0,
        linestyle="--",
        alpha=0.85,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(
        0,
        1.2 * max(history["train_loss"].max(), history["validation_loss"].max()),
    )
    axes[0].legend(frameon=True, loc="best")

    plotted = summary.copy()
    plotted["number of hidden layers"] = plotted["n_hidden_layers"]
    depth_labels = {
        depth: (
            "Logistic regression"
            if depth == 0
            else f"{depth} layer" if depth == 1 else f"{depth} layers"
        )
        for depth in sorted(plotted["number of hidden layers"].unique())
    }
    plotted["Input variables"] = plotted["input_set"].map(input_labels)
    plotted["Hidden layers"] = plotted["number of hidden layers"].map(depth_labels)

    hue_order = [temperature_precipitation_label, temperature_label]
    style_order = [depth_labels[depth] for depth in sorted(depth_labels)]
    display_palette = {input_labels[key]: colour for key, colour in palette.items()}
    display_markers = {
        depth_labels[key]: value for key, value in markers.items() if key in depth_labels
    }

    axis = axes[1]
    sns.scatterplot(
        data=plotted,
        x="n_parameters",
        y="mean_accuracy",
        hue="Input variables",
        hue_order=hue_order,
        palette=display_palette,
        style="Hidden layers",
        style_order=style_order,
        markers=display_markers,
        s=100,
        edgecolor="black",
        ax=axis,
    )
    for input_set, group in plotted.groupby("input_set"):
        group = group.sort_values("n_parameters")
        axis.plot(
            group["n_parameters"],
            group["mean_accuracy"],
            color=palette[input_set],
            linewidth=1.5,
            zorder=1,
        )
        axis.errorbar(
            group["n_parameters"],
            group["mean_accuracy"],
            yerr=group["sd_accuracy"],
            fmt="none",
            capsize=4,
            ecolor=palette[input_set],
            alpha=0.8,
            zorder=2,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Number of parameters (log scale)")
    axis.set_ylabel("Cross-validation accuracy")

    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    input_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=9,
            markerfacecolor=display_palette[label],
            markeredgecolor="black",
            label=label,
        )
        for label in hue_order
    ]
    depth_handles = [
        Line2D(
            [0],
            [0],
            marker=display_markers[label],
            linestyle="",
            markersize=9,
            markerfacecolor="none",
            markeredgecolor="black",
            label=label,
        )
        for label in style_order
    ]
    input_legend = axis.legend(
        handles=input_handles,
        title="Input variables",
        frameon=True,
        loc="upper left",
        bbox_to_anchor=(0.08, 0.98),
        borderaxespad=0,
    )
    axis.add_artist(input_legend)
    axis.legend(
        handles=depth_handles,
        title="Hidden layers",
        frameon=True,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        borderaxespad=0,
    )

    for panel, label in zip(axes, "ab"):
        sns.despine(ax=panel, trim=False)
        panel.spines["left"].set_linewidth(1.3)
        panel.spines["bottom"].set_linewidth(1.3)
        panel.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        panel.text(
            0.05,
            1.0,
            label,
            transform=panel.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="right",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    history, summary = load_current_results(args.results_dir)
    make_figure(history, summary, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
