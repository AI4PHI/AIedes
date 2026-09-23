"""Revision-only configuration, identities and atomic artifacts."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / "experiments/counter"
FIELDS = ["t2m_mean", "d2m_mean", "tp_sum"]


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_config(path=HERE / "config.json"):
    config = read_json(path)
    for key in ("weekly_data", "full_data", "split_manifest"):
        p = Path(config[key])
        config[key] = str((Path(path).resolve().parent / p).resolve()) if not p.is_absolute() else str(p)
    if config["schema_version"] != 1 or config["windows"] != [30, 60, 90]:
        raise ValueError("This protocol requires schema 1 and windows 30/60/90")
    if config["architectures"] != [[32], [64], [32, 16], [64, 32]] or config["dropouts"] != [0, .2, .4]:
        raise ValueError("Unexpected architecture/dropout grid")
    for key in ("max_epochs", "patience", "scheduler_horizon", "log_interval"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")
    if config["batch_size"] < 2 or config["learning_rate"] <= 0 or config["min_delta"] < 0:
        raise ValueError("Invalid training parameters")
    if config["final_seeds"] != [1, 2, 3, 4, 5] or config["screen_seed"] != 1:
        raise ValueError("Protocol seeds must be 1 and 1–5")
    if config["weight_bins"] != [1.] or config["autonomous_lag_days"] != [7, 14]:
        raise ValueError("Unexpected loss bins or autonomous lag rule")
    if config["climate_endpoint"] != "end_date" or config["autonomous_startup"] != "missing_flags_at_first_observation":
        raise ValueError("Unexpected climate/startup convention")
    if config["deployment_choice"] != "deferred" or config["plot_min_observations"] != 11:
        raise ValueError("Deployment must be deferred; plotting requires >10 observations")
    return config


def grid(config):
    inputs = []
    for fields, weekly, lags, window in itertools.product(
        [["t2m_mean", "tp_sum"], FIELDS], [False, True], [False, True], config["windows"]
    ):
        inputs.append(dict(fields=fields, weekly=weekly, lags=lags, window=window))
    inputs.append(dict(fields=[], weekly=False, lags=True, window=None))
    candidates = []
    for inp, widths, dropout in itertools.product(inputs, config["architectures"], config["dropouts"]):
        candidate = dict(inp, widths=widths, dropout=dropout)
        candidate["id"] = identity(candidate)[:12]
        candidates.append(candidate)
    assert len(candidates) == len({c["id"] for c in candidates}) == 300
    return sorted(candidates, key=lambda c: c["id"])


def source_hashes():
    files = [HERE / 'run.py', HERE / 'run_experiment.sh']
    for directory in ('data_loader', 'models', 'train', 'evaluation', 'utils'):
        files += list((ROOT / 'src/AIedes' / directory).glob('counter*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def mark_complete(folder, experiment_id, files):
    folder = Path(folder)
    write_json(folder / "done.json", {"experiment_id": experiment_id,
               "artifacts": {name: digest(folder / name) for name in files}})


def complete(folder, experiment_id):
    folder = Path(folder)
    if not (folder / "done.json").exists():
        return False
    record = read_json(folder / "done.json")
    if record["experiment_id"] != experiment_id:
        raise ValueError(f"Stale job identity: {folder}")
    if not record.get("artifacts"):
        raise ValueError(f"Empty completion record: {folder}")
    for name, sha in record["artifacts"].items():
        if not (folder / name).is_file() or digest(folder / name) != sha:
            raise ValueError(f"Corrupt completed artifact: {folder / name}")
    return True
