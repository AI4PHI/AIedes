"""Validated group membership, compact inputs and dated climate; splits are never regenerated."""
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

from AIedes.data_loader.counter_splits import load_manifest, rotating_group_indices
from AIedes.evaluation.counter_reconstruction import build_weather
from AIedes.utils.counter_experiment import FIELDS, complete, mark_complete, write_json

METADATA = ["id_trap", "country", "end_date", "weeklyRates", "prev1_rates", "prev2_rates"]


def synthetic_counter_data():
    """Small synthetic, not scientific, data for complete integration checks."""
    rows = []
    for trap in range(10):
        for j in range(12):
            day = pd.Timestamp("2020-05-01") + pd.Timedelta(days=7 * j)
            row = dict(id_trap=trap, country=f"synthetic{trap % 2}", end_date=day,
                       weeklyRates=float((j + trap) % 5),
                       prev1_rates=float((j - 1 + trap) % 5) if j else np.nan,
                       prev2_rates=float((j - 2 + trap) % 5) if j > 1 else np.nan,
                       outer_group=trap % 5 + 1)
            days = np.arange(day.toordinal() - 89, day.toordinal() + 1)
            for k, field in enumerate(FIELDS):
                row[field] = (np.sin(days / (8 + k)) + 3 + trap / 10) * (k + 1)
            rows.append(row)
    frame = pd.DataFrame(rows)
    return frame, frame.copy()


def prepare_counter_data(output, config, experiment_id, smoke=False):
    """Write the audited observation cache, dated weather and balance tables."""
    output = Path(output) / "data_audit"
    if complete(output, experiment_id):
        return
    output.mkdir(parents=True, exist_ok=True)
    if smoke:
        frame, full = synthetic_counter_data()
        manifest_hash = "synthetic"
    else:
        manifest, manifest_hash = load_manifest(config["split_manifest"], config["weekly_data"])
        frame = pd.read_pickle(config["weekly_data"])[METADATA + FIELDS].copy()
        frame["outer_group"] = manifest["outer_test_fold"]
        full = pd.read_pickle(config["full_data"])[["id_trap", "end_date"] + FIELDS]

    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame))
    frame["end_date"] = pd.to_datetime(frame.end_date)
    if frame.duplicated(["id_trap", "end_date"]).any():
        raise ValueError("Duplicate trap observation dates")
    if not frame.groupby("id_trap").outer_group.nunique().eq(1).all():
        raise ValueError("Trap crosses group boundary")
    if set(frame.outer_group) != set(range(1, 6)):
        raise ValueError("Expected all five trap groups")
    targets = frame.weeklyRates.to_numpy(float)
    if not np.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("Invalid target")
    for field in FIELDS:
        values = np.stack(frame[field].to_numpy())
        if values.shape != (len(frame), 90) or not np.isfinite(values).all():
            raise ValueError(f"Invalid observed climate: {field}")

    weather, eligibility, gaps = build_weather(frame, full)
    frame["history_class"] = np.select(
        [frame.prev1_rates.notna() & frame.prev2_rates.notna(),
         frame.prev1_rates.isna() & frame.prev2_rates.isna()],
        ["both_present", "both_missing"], default="one_missing")
    frame = frame.merge(eligibility[["id_trap", "reconstruction_eligible", "plot_eligible"]],
                        on="id_trap", validate="many_to_one")

    for outer in range(1, 6):
        coverage = np.zeros(len(frame), int)
        for validation in set(range(1, 6)) - {outer}:
            indices = rotating_group_indices(frame.outer_group, outer, validation)
            traps = {name: set(frame.iloc[rows].id_trap) for name, rows in indices.items()}
            if traps["train"] & traps["val"] or traps["train"] & traps["test"] or traps["val"] & traps["test"]:
                raise ValueError("Partitions share traps")
            coverage[indices["val"]] += 1
        np.testing.assert_array_equal(coverage, frame.outer_group.ne(outer).to_numpy(int))

    frame.to_pickle(output / "observations.pkl")
    with (output / "weather.pkl").open("wb") as f:
        pickle.dump(weather, f, protocol=pickle.HIGHEST_PROTOCOL)
    eligibility.to_csv(output / "traps.csv", index=False)
    gaps.to_csv(output / "missing_climate_days.csv", index=False)
    frame.drop(columns=FIELDS).to_csv(output / "rows.csv", index=False)
    balance = frame.groupby("outer_group").agg(
        traps=("id_trap", "nunique"), observations=("row_id", "size"),
        reconstruction_points=("reconstruction_eligible", "sum"), plot_points=("plot_eligible", "sum"),
        countries=("country", "nunique"),
        positive_fraction=("weeklyRates", lambda x: float((x > 0).mean())))
    balance.to_csv(output / "balance.csv")
    write_json(output / "manifest.json", {
        "outer_groups": frame.outer_group.tolist(), "trap_ids": frame.id_trap.tolist(),
        "source_manifest_sha256": manifest_hash, "experiment_id": experiment_id,
        "climate_alignment": config["climate_alignment_evidence"],
        "climate_endpoint": config["climate_endpoint"],
        "autonomous_lag_days": config["autonomous_lag_days"], "smoke_only": smoke})
    mark_complete(output, experiment_id, ["observations.pkl", "weather.pkl", "traps.csv", "rows.csv",
                                          "missing_climate_days.csv", "balance.csv", "manifest.json"])
    print(balance.to_string(), flush=True)
