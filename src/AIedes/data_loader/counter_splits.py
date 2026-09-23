"""Frozen balanced trap folds shared by diagnostics and Counter v2 training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BIN_NAMES = ["zero", "positive_to6", "6_to18", "18_to51", "51_to200", "over200"]


def bins(y):
    return np.searchsorted([0, 6, 18, 51, 200], y, side="left")


def features(d):
    t = d.groupby("id_trap", sort=True).agg(
        rows=("weeklyRates", "size"), country=("country", "first"),
        positives=("positive", "sum"), mean_rate=("weeklyRates", "mean"),
        peak=("weeklyRates", "max"), sum_rate=("weeklyRates", "sum"))
    t["traps"] = 1
    t["positive_traps"] = t.positives.gt(0).astype(int)
    hist = pd.crosstab(d.id_trap, d.rate_bin).reindex(index=t.index, columns=range(6), fill_value=0)
    hist.columns = BIN_NAMES
    country = pd.crosstab(d.id_trap, d.country).reindex(t.index, fill_value=0)
    a = pd.concat([t[["rows", "traps", "positive_traps"]], hist, country], axis=1)
    # Each block contributes its average squared relative deviation.
    weights = np.array([2, .3, .5] + [1/6]*6 + [.5/len(country.columns)]*len(country.columns))
    return t, a, weights


def balanced(d, k, seed):
    if k < 2 or d.id_trap.nunique() < k:
        raise ValueError("Each fold needs at least one trap; n_folds must be >= 2.")
    t, a, weights = features(d)
    raw = a.to_numpy(float)
    expected = raw.sum(axis=0)/k
    # A country with only four records cannot be evenly split five ways.
    floors = np.array([1]*9 + [10]*(raw.shape[1]-9))
    scaled = raw / np.maximum(expected, floors) * np.sqrt(weights)
    rng = np.random.default_rng(seed)
    counts = np.zeros((k, scaled.shape[1]))
    assignment = np.full(len(t), -1, dtype=int)
    # Allocate the most compositionally influential traps first.
    order = np.argsort(-(np.linalg.norm(scaled, axis=1) + rng.uniform(0, 1e-5, len(t))))
    for i in order:
        cost = 2*counts@scaled[i] + np.sum(scaled[i]**2)
        candidates = np.flatnonzero(np.isclose(cost, cost.min(), atol=1e-12))
        f = rng.choice(candidates)
        assignment[i] = f
        counts[f] += scaled[i]
    # Local moves/swaps improve the same fixed objective.
    for step in range(30000):
        i = rng.integers(len(t)); f = assignment[i]
        if step % 3 == 0:
            h = rng.integers(k)
            if h == f: continue
            v = scaled[i]
            delta = 2*np.dot(counts[h]-counts[f], v) + 2*np.dot(v,v)
            if delta < -1e-12:
                counts[f] -= v; counts[h] += v; assignment[i] = h
        else:
            j = rng.integers(len(t)); h = assignment[j]
            if h == f: continue
            v = scaled[j]-scaled[i]
            delta = 2*np.dot(counts[f]-counts[h],v) + 2*np.dot(v,v)
            if delta < -1e-12:
                counts[f] += v; counts[h] -= v
                assignment[i], assignment[j] = h, f
    mapping = pd.Series(assignment, index=t.index)
    return d.id_trap.map(mapping).to_numpy(), float(np.sum((counts-counts.mean(axis=0))**2))


def file_sha256(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_split_frame(input_file):
    frame = pd.read_pickle(input_file)[["id_trap", "country", "weeklyRates"]].reset_index(drop=True)
    if not frame.notna().all().all() or not np.isfinite(frame.weeklyRates).all() or (frame.weeklyRates < 0).any():
        raise ValueError("Trap IDs, countries and finite nonnegative targets are required.")
    if not frame.groupby("id_trap").country.nunique().eq(1).all():
        raise ValueError("Each trap must belong to one country.")
    frame["positive"] = frame.weeklyRates.gt(0)
    frame["rate_bin"] = bins(frame.weeklyRates)
    return frame


def build_manifest(input_file, split_seed=42, n_outer=5, n_inner=3):
    """Build the same outer/inner assignments as the reviewed diagnostic."""
    digest = file_sha256(input_file)
    frame = read_split_frame(input_file)
    outer, _ = balanced(frame, n_outer, split_seed)
    inner = {}
    for f in range(n_outer):
        mask = outer != f
        values, _ = balanced(frame.loc[mask], n_inner, split_seed + 100 + f)
        assignment = np.full(len(frame), -1, dtype=int)
        assignment[mask] = values + 1
        inner[str(f + 1)] = assignment.tolist()
    manifest = {
        "schema_version": 1, "algorithm": "balanced_traps_v1",
        "dataset_sha256": digest, "n_rows": len(frame), "n_traps": frame.id_trap.nunique(),
        "split_seed": split_seed, "n_outer": n_outer, "n_inner": n_inner,
        "target_bin_edges": [0, 6, 18, 51, 200],
        "objective": {"rows": 2, "traps": .3, "positive_traps": .5,
                      "target_bins_total": 1, "countries_total": .5,
                      "country_denominator_floor": 10, "local_proposals": 30000},
        "trap_ids": frame.id_trap.tolist(),
        "outer_test_fold": (outer + 1).tolist(), "inner_val_folds": inner,
    }
    if file_sha256(input_file) != digest:
        raise ValueError("Dataset changed while constructing folds.")
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or manifest.get("algorithm") != "balanced_traps_v1":
        raise ValueError("Unsupported split manifest format.")
    n, k, inner_k = (manifest[key] for key in ("n_rows", "n_outer", "n_inner"))
    if any(type(v) is not int for v in (n, k, inner_k)) or n < 1 or min(k, inner_k) < 2:
        raise ValueError("Invalid row/fold counts in manifest.")
    traps = pd.Series(manifest["trap_ids"])
    if len(traps) != n or traps.isna().any() or traps.nunique() != manifest["n_traps"]:
        raise ValueError("Invalid trap inventory in manifest.")

    def labels(values, allowed):
        a = np.asarray(values)
        if a.shape != (n,) or a.dtype.kind not in "iu" or not np.isin(a, allowed).all():
            raise ValueError("Invalid fold membership array.")
        return a

    outer = labels(manifest["outer_test_fold"], np.arange(1, k + 1))
    if set(outer) != set(range(1, k + 1)) or not pd.Series(outer).groupby(traps).nunique().eq(1).all():
        raise ValueError("Outer folds are empty or split a trap across partitions.")
    if set(manifest["inner_val_folds"]) != {str(f) for f in range(1, k + 1)}:
        raise ValueError("Missing or extra inner-fold assignments.")
    for f in range(1, k + 1):
        inner = labels(manifest["inner_val_folds"][str(f)], [-1] + list(range(1, inner_k + 1)))
        test = outer == f
        if not np.all(inner[test] == -1) or set(inner[~test]) != set(range(1, inner_k + 1)):
            raise ValueError("Outer test rows enter inner CV, or an inner fold is empty.")
        if not pd.Series(inner[~test]).groupby(traps[~test].reset_index(drop=True)).nunique().eq(1).all():
            raise ValueError("Inner folds split a trap across partitions.")


def load_manifest(path, input_file):
    """Reject stale or altered dataset order before any training starts."""
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    validate_manifest(manifest)
    if file_sha256(input_file) != manifest["dataset_sha256"]:
        raise ValueError("Dataset checksum differs from the frozen split manifest; regenerate/review folds.")
    frame = read_split_frame(input_file)
    if frame.id_trap.tolist() != manifest["trap_ids"]:
        raise ValueError("Dataset row/trap order differs from the split manifest.")
    return manifest, hashlib.sha256(payload).hexdigest()


def manifest_indices(manifest, outer_fold, inner_fold):
    if not 1 <= outer_fold <= manifest["n_outer"] or not 1 <= inner_fold <= manifest["n_inner"]:
        raise ValueError("Selected outer/inner fold is outside the manifest's fold range.")
    outer = np.asarray(manifest["outer_test_fold"])
    inner = np.asarray(manifest["inner_val_folds"][str(outer_fold)])
    indices = {
        "train": np.flatnonzero((outer != outer_fold) & (inner != inner_fold)),
        "val": np.flatnonzero(inner == inner_fold),
        "test": np.flatnonzero(outer == outer_fold),
    }
    return indices


def rotating_group_indices(group_labels, test_group, validation_group):
    """Use two distinct existing groups as test/validation, without rebalancing."""
    groups = np.asarray(group_labels)
    if test_group == validation_group or test_group not in set(groups) or validation_group not in set(groups):
        raise ValueError("Distinct existing test and validation groups required")
    return {
        "train": np.flatnonzero((groups != test_group) & (groups != validation_group)),
        "val": np.flatnonzero(groups == validation_group),
        "test": np.flatnonzero(groups == test_group),
    }
