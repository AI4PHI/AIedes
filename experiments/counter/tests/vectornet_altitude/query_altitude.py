"""Query terrain elevation for the processed VectorNet trap coordinates."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
INPUT = ROOT / "experiments/counter/data/eggs_y_norm_7days_vectorNet.pkl"
OUTPUT = Path(__file__).resolve().parent / "results"
API = "https://api.opentopodata.org/v1/srtm90m"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_pickle(INPUT)
    traps = (
        frame.groupby("id_trap")
        .agg(
            latitude=("decimalLatitude", "first"),
            longitude=("decimalLongitude", "first"),
            observations=("id_trap", "size"),
        )
        .reset_index()
        .sort_values("id_trap")
        .reset_index(drop=True)
    )

    locations = "|".join(
        f"{row.latitude:.7f},{row.longitude:.7f}"
        for row in traps.itertuples()
    )
    url = f"{API}?locations={quote(locations, safe=',|.-')}&interpolation=bilinear"
    with urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if payload.get("status") != "OK":
        raise RuntimeError(payload)

    results = payload["results"]
    if len(results) != len(traps):
        raise RuntimeError(f"Expected {len(traps)} elevation results, got {len(results)}")

    traps["elevation_m"] = [item["elevation"] for item in results]
    traps["elevation_dataset"] = [item.get("dataset") for item in results]
    traps.to_csv(OUTPUT / "vectornet_trap_altitudes.csv", index=False)

    summary = {
        "source": API,
        "dataset": "srtm90m",
        "interpolation": "bilinear",
        "traps": int(len(traps)),
        "elevation_m": {
            "minimum": float(traps.elevation_m.min()),
            "maximum": float(traps.elevation_m.max()),
            "mean": float(traps.elevation_m.mean()),
            "median": float(traps.elevation_m.median()),
        },
        "traps_by_elevation": traps.to_dict(orient="records"),
    }
    (OUTPUT / "vectornet_trap_altitudes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "traps_by_elevation"}, indent=2))


if __name__ == "__main__":
    main()
