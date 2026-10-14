"""Create observed-versus-predicted scatter plots for VectorNet validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "experiments/counter/results/reports/vectornet_predictions.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def score(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(observed)),
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "slope": float(np.polyfit(observed, predicted, 1)[0]),
    }


def make_plot(input_path: Path, output_dir: Path, label: str | None = None) -> dict[str, dict[str, float | int]]:
    frame = pd.read_csv(input_path)
    required = {"target", "date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    observed = frame["target"].to_numpy(float)
    prediction_column = "main" if "main" in frame else "prediction"
    if prediction_column not in frame:
        raise ValueError("Input must contain either 'main' or 'prediction'")
    model_label = label or ("Observed-lag ensemble" if prediction_column == "main" else "Autoregressive ensemble")
    predicted = frame[prediction_column].to_numpy(float)
    ensemble_columns = [column for column in frame.columns if column.startswith("main_fold")]
    ensemble_sd = None
    if prediction_column == "main":
        if len(ensemble_columns) != 25:
            raise ValueError(f"Expected 25 observed-lag ensemble columns, found {len(ensemble_columns)}")
        ensemble_sd = frame[ensemble_columns].std(axis=1, ddof=1).to_numpy(float)
    if not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Observed or predicted values contain non-finite values")

    dates = pd.to_datetime(frame["date"])
    period_masks = {
        "Before September": dates < pd.Timestamp("2021-09-01"),
        "September onward": dates >= pd.Timestamp("2021-09-01"),
    }
    metrics: dict[str, dict[str, float | int]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), squeeze=False)
    colors = ["#1f77b4", "#d95f02"]
    for row, (period, mask), color in zip(range(2), period_masks.items(), colors):
        x_raw, y_raw = observed[mask], predicted[mask]
        x_log, y_log = np.log1p(x_raw), np.log1p(y_raw)
        for col, (scale, x_values, y_values, axis_label, y_sd) in enumerate([
            ("linear", x_raw, y_raw, "weekly egg-laying rate", None if ensemble_sd is None else ensemble_sd[mask]),
            ("log1p", x_log, y_log, "$\\log(1 +$ weekly egg-laying rate$)$",
             None if ensemble_sd is None else np.std(np.log1p(frame.loc[mask, ensemble_columns].to_numpy(float)), axis=1, ddof=1)),
        ]):
            ax = axes[row, col]
            key = f"{period} ({scale})"
            metrics[key] = score(x_values, y_values)
            limit = max(x_values.max(), y_values.max()) * 1.05
            ax.scatter(x_values, y_values, s=24, alpha=0.6, color=color, edgecolors="none")
            if y_sd is not None:
                ax.errorbar(x_values, y_values, yerr=y_sd, fmt="none", ecolor=color,
                            alpha=0.45, elinewidth=0.8, capsize=2)
            ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="1:1 line")
            fit = np.polyfit(x_values, y_values, 1)
            x_fit = np.array([0.0, limit])
            ax.plot(x_fit, np.polyval(fit, x_fit), color="#b22222", linestyle=":", label="Linear fit")
            current = metrics[key]
            ax.set_title(f"{period} — {scale}\n$R^2$ = {current['r2']:.3f}, RMSE = {current['rmse']:.3f}")
            ax.set_xlabel(f"Observed {axis_label}")
            ax.set_ylabel(f"Predicted {axis_label}")
            ax.set_xlim(0, limit)
            ax.set_ylim(0, limit)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper left", frameon=True)
    fig.suptitle(f"VectorNet external validation (Lazio, 2021) — {model_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "vectornet_scatter.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "vectornet_scatter.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "input": str(input_path),
        "observations": int(len(frame)),
        "traps": int(frame["id_trap"].nunique()) if "id_trap" in frame else None,
        "metrics": metrics,
        "uncertainty": "±1 standard deviation across the 25 observed-lag fold/seed predictions",
    }
    (output_dir / "vectornet_scatter_metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--label", default=None, help="Model label used in the figure title")
    args = parser.parse_args()
    metrics = make_plot(args.input.resolve(), args.output.resolve(), label=args.label)
    for label, values in metrics.items():
        print(f"{label}: R2={values['r2']:.6f}, RMSE={values['rmse']:.6f}")


if __name__ == "__main__":
    main()
