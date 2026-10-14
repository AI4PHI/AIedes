"""Evaluate a VectorNet hybrid: observed-history model with climate-only fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA = ROOT / "experiments/counter/data/eggs_y_norm_7days_vectorNet.pkl"
DEFAULT_PREDICTIONS = ROOT / "experiments/counter/results/reports/vectornet_predictions.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_pickle(args.data).reset_index(drop=True)
    predictions = pd.read_csv(args.predictions)
    if len(data) != len(predictions):
        raise ValueError("Input data and prediction rows do not match")

    has_history = data["prev1_rates"].notna() | data["prev2_rates"].notna()
    predictions["prediction"] = predictions["climate_only"]
    predictions.loc[has_history, "prediction"] = predictions.loc[has_history, "main"]
    predictions["source_model"] = "climate_only_fallback"
    predictions.loc[has_history, "source_model"] = "observed_lag"
    predictions["date"] = pd.to_datetime(predictions["date"])
    columns = ["id_trap", "country", "date", "row_id", "target", "prediction", "source_model"]
    predictions[columns].to_csv(args.output / "vectornet_hybrid_predictions.csv", index=False)

    report = {
        "observations": int(len(predictions)),
        "traps": int(predictions["id_trap"].nunique()),
        "observed_lag_model": int(has_history.sum()),
        "climate_only_fallback": int((~has_history).sum()),
        "fallback_rule": "use observed-lag prediction if prev1_rates or prev2_rates is available; otherwise use climate-only prediction",
        "previous_values": {
            "prev1": int(data["prev1_rates"].notna().sum()),
            "prev2": int(data["prev2_rates"].notna().sum()),
            "both": int((data["prev1_rates"].notna() & data["prev2_rates"].notna()).sum()),
            "neither": int((~has_history).sum()),
        },
    }
    (args.output / "vectornet_hybrid_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
