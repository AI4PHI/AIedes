"""Read-only data audits: saved lag availability and reconstruction eligibility.

Neither function fits models, changes split membership nor rewrites datasets.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from AIedes.data_loader.counter_splits import file_sha256, load_manifest
from AIedes.utils.counter_experiment import FIELDS

EXAMPLE_TRAPS = [360, 361, 363, 365]


def audit_lag_availability(weekly_path, full_path, output_dir):
    """Confirm the saved previous-rate columns match the audited ±3-day lookup."""
    output = Path(output_dir)
    frame = pd.read_pickle(weekly_path).reset_index(drop=True)
    full = pd.read_pickle(full_path).reset_index(drop=True)
    frame.index.name = "row_index"
    rows = frame[["id_trap", "country", "start_date", "end_date", "weeklyRates",
                  "samplingEffort Days", "prev1_rates", "prev2_rates"]].copy()
    for name in ["start_date", "end_date"]:
        rows[name] = pd.to_datetime(rows[name])
        full[name] = pd.to_datetime(full[name])
    rows["p1_missing"] = rows.prev1_rates.isna()
    rows["p2_missing"] = rows.prev2_rates.isna()
    rows["any_missing"] = rows.p1_missing | rows.p2_missing
    rows["both_missing"] = rows.p1_missing & rows.p2_missing
    rows["availability"] = np.select(
        [~rows.p1_missing & ~rows.p2_missing, rows.p1_missing & ~rows.p2_missing,
         ~rows.p1_missing & rows.p2_missing],
        ["both_available", "p1_only_missing", "p2_only_missing"], default="both_missing")
    rows["positive"] = rows.weeklyRates.gt(0)
    ordered = rows.sort_values(["id_trap", "end_date"])
    rows["weekly_series_position"] = ordered.groupby("id_trap").cumcount().add(1)
    rows["gap_from_previous_weekly_record_days"] = (
        ordered.groupby("id_trap").end_date.diff().dt.total_seconds() / 86400)
    if rows.duplicated(["id_trap", "end_date"]).any():
        raise ValueError("Duplicate trap observation dates")

    groups = {key: value.sort_values("end_date") for key, value in full.groupby("id_trap")}
    for period in [1, 2]:
        prefix = f"p{period}"
        for column in ("source_matches_saved", "has_weekly_source",
                       "history_window_before_first_record", "has_any_earlier_record"):
            rows[f"{prefix}_{column}"] = False
        for idx, row in rows.iterrows():
            source = groups[row.id_trap]
            center = row.end_date - pd.Timedelta(days=period * row["samplingEffort Days"])
            lower, upper = center - pd.Timedelta(days=3), center + pd.Timedelta(days=3)
            eligible = source[(source.end_date > lower) & (source.end_date < upper)
                              & (source.end_date != row.end_date)]
            expected = eligible.weeklyRates.mean()
            saved = row[f"prev{period}_rates"]
            rows.at[idx, f"{prefix}_source_matches_saved"] = (
                (pd.isna(expected) and pd.isna(saved)) or np.isclose(expected, saved))
            weekly = rows[rows.id_trap.eq(row.id_trap)]
            rows.at[idx, f"{prefix}_has_weekly_source"] = bool(
                ((weekly.end_date > lower) & (weekly.end_date < upper)
                 & (weekly.end_date < row.end_date)).any())
            rows.at[idx, f"{prefix}_history_window_before_first_record"] = upper <= source.end_date.min()
            rows.at[idx, f"{prefix}_has_any_earlier_record"] = bool((source.end_date < row.end_date).any())
        if not rows[f"{prefix}_source_matches_saved"].all():
            raise ValueError(f"Saved lag disagrees with the audited lookup: {prefix}")
        flags = np.vstack(frame["prev_weeklyRates" if period == 1 else "prev2_weeklyRates"])[:, 1]
        np.testing.assert_array_equal(flags, (~rows[f"{prefix}_missing"]).astype(float))

    summary = rows.groupby("availability").agg(
        n=("id_trap", "size"), traps=("id_trap", "nunique"), positives=("positive", "sum"))
    summary["percent_of_rows"] = 100 * summary.n / len(rows)
    summary["positive_percent"] = 100 * summary.positives / summary.n
    traps = rows.groupby("id_trap").agg(
        country=("country", "first"), n=("id_trap", "size"), start=("end_date", "min"),
        end=("end_date", "max"), positives=("positive", "sum"),
        p1_missing=("p1_missing", "sum"), p2_missing=("p2_missing", "sum"),
        any_missing=("any_missing", "sum"), both_missing=("both_missing", "sum"))
    countries = rows.groupby("country").agg(
        n=("id_trap", "size"), traps=("id_trap", "nunique"),
        p1_missing=("p1_missing", "sum"), p2_missing=("p2_missing", "sum"),
        any_missing=("any_missing", "sum"), both_missing=("both_missing", "sum"))
    for table in (traps, countries):
        for column in ["p1_missing", "p2_missing", "any_missing", "both_missing"]:
            table[column + "_percent"] = 100 * table[column] / table.n

    report = {
        "rows": len(rows), "traps": len(traps),
        "counts": {c: int(rows[c].sum()) for c in ["p1_missing", "p2_missing", "any_missing", "both_missing"]},
        "traps_with_any_missing": int(traps.any_missing.gt(0).sum()),
        "traps_with_at_least_half_rows_missing_any_lag": int(traps.any_missing_percent.ge(50).sum()),
        "traps_with_every_row_missing_any_lag": int(traps.any_missing.eq(traps.n).sum()),
        "traps_with_every_row_missing_both_lags": int(traps.both_missing.eq(traps.n).sum()),
        "longer_traps_n_ge_10": int(traps.n.ge(10).sum()),
        "longer_traps_n_ge_10_with_at_least_half_rows_missing_any_lag":
            int((traps.n.ge(10) & traps.any_missing_percent.ge(50)).sum()),
        "any_missing_after_first_two_weekly_records":
            int((rows.any_missing & rows.weekly_series_position.gt(2)).sum()),
        "both_missing_after_first_two_weekly_records":
            int((rows.both_missing & rows.weekly_series_position.gt(2)).sum()),
        "rows_after_first_two_weekly_records": int(rows.weekly_series_position.gt(2).sum()),
    }
    for period in [1, 2]:
        prefix = f"p{period}"
        missing = rows[f"{prefix}_missing"]
        before = rows[f"{prefix}_history_window_before_first_record"]
        report[prefix] = {
            "missing_before_recorded_history_window": int((missing & before).sum()),
            "missing_despite_history_extending_to_window": int((missing & ~before).sum()),
            "missing_with_any_earlier_record": int((missing & rows[f"{prefix}_has_any_earlier_record"]).sum()),
            "observed_lag_present_without_source_in_weekly_subset":
                int((~missing & ~rows[f"{prefix}_has_weekly_source"]).sum()),
            "rows_without_weekly_source_for_observation_date_recursion":
                int((~rows[f"{prefix}_has_weekly_source"]).sum()),
        }

    output.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output / "rows.csv")
    summary.to_csv(output / "availability_summary.csv")
    traps.sort_values(["any_missing_percent", "n"], ascending=False).to_csv(output / "traps.csv")
    countries.to_csv(output / "countries.csv")
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("\nAVAILABILITY\n" + summary.to_string())
    print("\nTRAPS WITH >=10 ROWS, SORTED BY MISSING-HISTORY SHARE\n" +
          traps[traps.n.ge(10)].sort_values(["any_missing_percent", "n"], ascending=False).head(12).to_string())
    examples = [t for t in EXAMPLE_TRAPS if t in traps.index]
    if examples:
        print("\nLONGEST EXAMPLE TRAPS\n" + traps.loc[examples].to_string())
    print(f"\nSaved: {output}")
    return report


def audit_reconstruction_eligibility(weekly_path, full_path, manifest_path, output_dir,
                                     test_group=1, plot_min_observations=11):
    """Count main-test, reconstruction and plotting cohorts in one existing group."""
    if plot_min_observations < 1:
        raise ValueError("plot_min_observations must be positive")
    manifest, manifest_hash = load_manifest(manifest_path, weekly_path)
    if test_group not in range(1, manifest["n_outer"] + 1):
        raise ValueError("Test group is outside the existing manifest")
    weekly = pd.read_pickle(weekly_path).reset_index(drop=True)
    weekly.index.name = "row_index"
    labels = np.asarray(manifest["outer_test_fold"])
    test = weekly.loc[labels == test_group].copy()
    development = weekly.loc[labels != test_group]
    if not set(test.id_trap).isdisjoint(development.id_trap):
        raise ValueError("Test and development share traps")
    full = pd.read_pickle(full_path)
    for frame in (test, full):
        frame["end_date"] = pd.to_datetime(frame.end_date)
    sources = dict(tuple(full.loc[full.id_trap.isin(test.id_trap)].groupby("id_trap")))
    trap_records, gap_records, overlap_days = [], [], 0
    fmt = lambda day: pd.Timestamp.fromordinal(int(day)).strftime("%Y-%m-%d")

    for trap_id, obs in test.groupby("id_trap"):
        chunks = []
        for row in sources[trap_id].itertuples(index=False):
            arrays = [np.asarray(getattr(row, c), dtype=float) for c in FIELDS]
            if not all(a.shape == (90,) and np.isfinite(a).all() for a in arrays):
                raise ValueError(f"Invalid cached climate at trap {trap_id}")
            end = row.end_date.toordinal()
            chunk = pd.DataFrame(np.stack(arrays, axis=1), columns=FIELDS)
            chunk["day"] = np.arange(end - 89, end + 1)
            chunks.append(chunk)
        weather = pd.concat(chunks, ignore_index=True)
        grouped = weather.groupby("day")[FIELDS]
        lo, hi = grouped.min(), grouped.max()
        if not np.isclose(lo.to_numpy(), hi.to_numpy(), rtol=1e-8, atol=1e-6).all():
            raise ValueError(f"Inconsistent overlapping climate at trap {trap_id}")
        overlap_days += len(weather) - len(lo)
        available = set(lo.index)
        first, last = obs.end_date.min().toordinal(), obs.end_date.max().toordinal()
        missing = sorted(set(range(first - 89, last + 1)) - available)
        if missing:
            starts, ends = [missing[0]], []
            for before, after in zip(missing[:-1], missing[1:]):
                if after != before + 1:
                    ends.append(before)
                    starts.append(after)
            ends.append(missing[-1])
            for a, b in zip(starts, ends):
                gap_records.append({"id_trap": int(trap_id), "country": obs.country.iloc[0],
                                    "start": fmt(a), "end": fmt(b), "days": b - a + 1})
        observed_valid = all(
            all(np.asarray(value).shape == (90,) and np.isfinite(np.asarray(value, dtype=float)).all()
                for value in obs[c]) for c in FIELDS)
        if not observed_valid:
            raise ValueError(f"Invalid observed climate at trap {trap_id}")
        record = {
            "id_trap": int(trap_id), "country": obs.country.iloc[0],
            "observations": len(obs), "positive_observations": int(obs.weeklyRates.gt(0).sum()),
            "first_observation": fmt(first), "last_observation": fmt(last),
            "span_days": last - first, "all_observed_climate_complete": observed_valid,
            "missing_climate_days": len(missing), "continuous_climate": not missing,
            "enough_observations_for_plot": len(obs) >= plot_min_observations,
        }
        for window in (30, 60, 90):
            record[f"complete_{window}d_inputs"] = set(range(first - window + 1, last + 1)).issubset(available)
            record[f"complete_{window}d_inputs_with_14d_prewarmup"] = set(
                range(first - 14 - window + 1, last + 1)).issubset(available)
        record["eligible_reconstruction"] = record["continuous_climate"]
        record["eligible_plot"] = record["continuous_climate"] and record["enough_observations_for_plot"]
        record["plot_exclusion_reason"] = (
            "climate_gap" if not record["continuous_climate"]
            else "short_series" if not record["enough_observations_for_plot"] else "eligible")
        trap_records.append(record)

    traps = pd.DataFrame(trap_records)
    rows = test[["id_trap", "country", "end_date", "weeklyRates"]].reset_index().merge(
        traps[["id_trap", "continuous_climate", "enough_observations_for_plot",
               "eligible_reconstruction", "eligible_plot", "plot_exclusion_reason"]],
        on="id_trap", validate="many_to_one")
    stages = []
    for name, mask in [
        ("Main test: all traps", pd.Series(True, index=traps.index)),
        ("Reconstruction: complete climate only", traps.continuous_climate),
        (f"Plots only: complete climate and > {plot_min_observations - 1} observations", traps.eligible_plot),
    ]:
        subset = traps.loc[mask]
        n = int(subset.observations.sum())
        positives = int(subset.positive_observations.sum())
        stages.append({"stage": name, "traps": len(subset), "observations": n,
                       "excluded_traps": len(traps) - len(subset), "excluded_observations": len(test) - n,
                       "countries": int(subset.country.nunique()), "positive_observations": positives,
                       "positive_percent": 100 * positives / n if n else None})
    summary = pd.DataFrame(stages)
    countries = rows.groupby("country").agg(
        test_traps=("id_trap", "nunique"), test_observations=("row_index", "size"),
        reconstruction_observations=("eligible_reconstruction", "sum"))
    for column, mask in (("reconstruction_traps", traps.eligible_reconstruction), ("plot_traps", traps.eligible_plot)):
        countries[column] = traps.loc[mask].groupby("country").size()
        countries[column] = countries[column].fillna(0).astype(int)

    report = {
        "status": "provisional eligibility audit; does not freeze or change the split",
        "test_group": test_group,
        "choice_rule": "group 1 selected by label, without model scores" if test_group == 1 else "explicit group argument",
        "plot_minimum_observations": plot_min_observations,
        "performance_minimum_series_length": None,
        "date_assumption": "90 oldest-to-newest daily climate values end on each observation end_date; absolute endpoint requires source confirmation",
        "initialization": "start at first retained observation with missing egg-lag flags; no pre-observation warm-up required for primary eligibility",
        "climate_source": "full merged cache, including non-weekly records; only climate fields used",
        "manifest_sha256": manifest_hash, "weekly_dataset_sha256": manifest["dataset_sha256"],
        "full_dataset_sha256": file_sha256(full_path),
        "development_traps": int(development.id_trap.nunique()), "development_observations": len(development),
        "overlapping_climate_days_checked": overlap_days, "overlap_mismatches": 0,
        "stages": stages,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    traps.to_csv(output / "traps.csv", index=False)
    rows.to_csv(output / "test_rows.csv", index=False)
    summary.to_csv(output / "eligibility_summary.csv", index=False)
    countries.to_csv(output / "countries.csv")
    pd.DataFrame(gap_records, columns=["id_trap", "country", "start", "end", "days"]).to_csv(
        output / "climate_gaps.csv", index=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"# Provisional test group {test_group}: reconstruction eligibility", "",
             "This audit uses existing balanced outer-group membership; it reads no model scores and changes no split, model, or plan.", "",
             f"Development: {report['development_traps']} traps / {len(development)} observations. "
             f"Test: {len(traps)} traps / {len(test)} observations.", "",
             "| Evaluation | Included traps | Included observations | Excluded traps | Excluded observations |",
             "|---|---:|---:|---:|---:|"]
    for stage in stages:
        lines.append(f"| {stage['stage']} | {stage['traps']} | {stage['observations']} | "
                     f"{stage['excluded_traps']} | {stage['excluded_observations']} |")
    lines += ["", "Climate exclusions apply only to supplementary uninterrupted reconstruction, not the main held-out test. The final table row applies to plots only; its exclusions must never enter either performance report. Counts refer to observed egg-rate measurements, not simulated daily outputs.", "",
              f"The agreed plotting criterion is strictly >10 observations (this run uses >{plot_min_observations - 1}). Climate-complete short series contribute to pooled reconstruction metrics. Single-observation traps cannot establish trajectory skill by themselves.", "",
              "Climate eligibility requires every daily input from the first through last retained observation to be constructible with a 90-day window. This common criterion also covers the 30- and 60-day candidates. We use all available climate histories at the same trap from the full merged cache, without using its egg targets for reconstruction.", "",
              "Arrays are assumed oldest-to-newest and ending on end_date. Overlapping values were checked for all three climate variables; absolute date alignment still requires extraction-source confirmation.", "",
              "Simulation is assumed to start at the first observation with missing egg-history flags. Earlier warm-up is not required here. Optional 14-day pre-observation climate coverage is recorded in traps.csv; requiring it would change eligibility. Scores should distinguish startup behavior once the lag calendar is fixed.", "",
              "traps.csv lists every test trap, test_rows.csv identifies every test observation with separate reconstruction and plot eligibility, climate_gaps.csv gives the missing calendar intervals, and countries.csv describes the retained geographical coverage.", "",
              "Reproduce from the experiment folder with:", "", "```bash",
              f"./run_experiment.sh --stage audit --audit-group {test_group}", "```", ""]
    (output / "README.md").write_text("\n".join(lines))

    if len(rows) != len(test) or rows.row_index.nunique() != len(test):
        raise ValueError("Row inventory does not match the test group")
    if int(traps.observations.sum()) != len(test):
        raise ValueError("Trap observation counts do not match the test group")
    if not rows.eligible_reconstruction.equals(rows.continuous_climate):
        raise ValueError("Reconstruction eligibility must equal climate continuity")
    if not (rows.eligible_plot <= rows.eligible_reconstruction).all():
        raise ValueError("Plot eligibility must imply reconstruction eligibility")
    print(summary.to_string(index=False))
    print("\nPlot exclusion reasons (mutually exclusive; not performance exclusions):")
    print(traps.groupby("plot_exclusion_reason").agg(
        traps=("id_trap", "size"), observations=("observations", "sum")).to_string())
    print("\nClimate gaps:")
    print(traps.loc[~traps.continuous_climate,
                    ["id_trap", "country", "observations", "missing_climate_days"]].to_string(index=False))
    print("\nCountries:")
    print(countries.to_string())
    print(f"\nSaved: {output}")
    return report
