#!/usr/bin/env python3
"""Validate and prepare the Copernicus classifier input for paper experiments.

The source archive is intentionally left unchanged.  This script creates a
row-level, documented modelling table and a manifest so that all later stages
use the same checked input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
TEMPERATURE_FEATURES = [f"temp_{month}_C" for month in MONTHS]
PRECIPITATION_FEATURES = [f"precip_{month}_mm" for month in MONTHS]
REQUIRED_COLUMNS = {
    "latitude", "longitude", "LocationCode", "CountryCode", "presence_numeric",
    "Precipitation Suitable", "Temperature Suitable", "Suitable",
    *TEMPERATURE_FEATURES, *PRECIPITATION_FEATURES,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(source)
    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {', '.join(missing)}")

    labelled = dataframe[dataframe["presence_numeric"].isin([0, 1])].copy()
    if labelled.empty:
        raise ValueError("No confirmed presence/absence observations were found.")
    if labelled[TEMPERATURE_FEATURES + PRECIPITATION_FEATURES].isna().any().any():
        raise ValueError("Climate features contain missing values in confirmed observations.")
    if labelled[["latitude", "longitude"]].isna().any().any():
        raise ValueError("Coordinates contain missing values in confirmed observations.")

    # This is the established label construction in the manuscript: confirmed
    # presence is retained only where the pre-defined conservative suitability
    # mask is true.  The unmasked surveillance status is retained for auditing.
    labelled["surveillance_presence"] = (labelled["presence_numeric"] == 1).astype("int8")
    labelled["target_presence"] = (
        (labelled["presence_numeric"] == 1) & (labelled["Suitable"] == 1)
    ).astype("int8")
    labelled["row_id"] = range(len(labelled))
    labelled["NUTS3_Code"] = labelled["LocationCode"].astype(str)

    prepared_path = output_dir / "copernicus_classifier_prepared.csv.gz"
    labelled.to_csv(prepared_path, index=False, compression="gzip")

    label_counts = labelled["target_presence"].value_counts().sort_index()
    manifest = {
        "source_file": str(source),
        "source_sha256": sha256(source),
        "source_rows": int(len(dataframe)),
        "prepared_rows": int(len(labelled)),
        "temperature_features": TEMPERATURE_FEATURES,
        "precipitation_features": PRECIPITATION_FEATURES,
        "target_definition": "presence_numeric == 1 and Suitable == 1",
        "target_counts": {str(key): int(value) for key, value in label_counts.items()},
        "n_regions": int(labelled["NUTS3_Code"].nunique()),
        "n_countries": int(labelled["CountryCode"].nunique()),
        "coordinate_columns": ["latitude", "longitude"],
    }
    manifest_path = output_dir / "copernicus_data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return prepared_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("experiments/classifier/data/ecdc_albopictus_cordex_2020.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/classifier/paper_results/input"),
    )
    arguments = parser.parse_args()
    prepared, manifest = prepare(arguments.source, arguments.output_dir)
    print(f"Prepared data: {prepared}")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
