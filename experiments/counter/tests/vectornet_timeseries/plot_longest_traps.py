"""Plot observed and observed-lag predictions for the six longest VectorNet traps."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-traps", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.input, parse_dates=["date"])
    required = {"id_trap", "date", "target", "main"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    counts = frame.groupby("id_trap").size().sort_values(ascending=False)
    selected = counts.head(args.n_traps).index.tolist()
    selected_frame = frame[frame.id_trap.isin(selected)].copy()
    selected_frame = selected_frame.sort_values(["id_trap", "date"])
    ensemble_columns = [column for column in frame.columns if column.startswith("main_fold")]
    if len(ensemble_columns) != 25:
        raise ValueError(f"Expected 25 observed-lag ensemble columns, found {len(ensemble_columns)}")

    summary = []
    for trap in selected:
        part = selected_frame[selected_frame.id_trap == trap]
        summary.append({
            "id_trap": float(trap),
            "observations": int(len(part)),
            "first_date": str(part.date.min().date()),
            "last_date": str(part.date.max().date()),
            "r2": float(r2_score(part.target, part.main)) if len(part) > 1 and part.target.nunique() > 1 else None,
            "rmse": float(np.sqrt(mean_squared_error(part.target, part.main))),
        })

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharey=False)
    axes = axes.ravel()
    for ax, trap in zip(axes, selected):
        part = selected_frame[selected_frame.id_trap == trap]
        ax.plot(part.date, part.target, "o-", color="#1f77b4", linewidth=1.5,
                markersize=4, label="Observed")
        prediction_sd = part[ensemble_columns].std(axis=1, ddof=1).to_numpy(float)
        ax.errorbar(part.date, part.main, yerr=prediction_sd, fmt="s--", color="#d95f02",
                    linewidth=1.5, markersize=4, capsize=3, label="Predicted ± 1 SD")
        row = next(item for item in summary if item["id_trap"] == float(trap))
        r2_text = "NA" if row["r2"] is None else f"{row['r2']:.2f}"
        ax.set_title(f"Trap {trap:g} — n={row['observations']}, R²={r2_text}, RMSE={row['rmse']:.1f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Weekly egg-laying rate")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(loc="upper left")
    for ax in axes[len(selected):]:
        ax.set_visible(False)
    fig.suptitle("VectorNet external validation — six longest traps\nObserved-lag ensemble", y=0.99)
    fig.tight_layout()
    fig.savefig(args.output / "vectornet_longest_traps_timeseries.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "vectornet_longest_traps_timeseries.pdf", bbox_inches="tight")
    plt.close(fig)

    report = {
        "input": str(args.input.resolve()),
        "models_averaged": 25,
        "selection": "six traps with the largest number of VectorNet observations",
        "uncertainty_definition": "standard deviation across 25 observed-lag fold/seed predictions",
        "traps": summary,
    }
    (args.output / "vectornet_longest_traps_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
