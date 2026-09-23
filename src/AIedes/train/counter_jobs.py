"""One screening or refit job: trap-disjoint rows in, verified artifacts out."""
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from AIedes.data_loader.counter_splits import rotating_group_indices
from AIedes.train.counter_train import fit_counter_candidate
from AIedes.utils.counter_experiment import complete, mark_complete, write_json

_FRAME = None


def initialize_counter_worker(data_path):
    """Load the prepared observations once per worker process."""
    global _FRAME
    torch.set_num_threads(1)
    _FRAME = pd.read_pickle(data_path)


def job_indices(group_labels, outer_group, validation_group):
    """Screening excludes the test and validation groups; refits only the test group."""
    labels = np.asarray(group_labels)
    if validation_group is None:
        if outer_group not in set(labels.tolist()):
            raise ValueError("Unknown outer test group")
        return np.flatnonzero(labels != outer_group), None
    indices = rotating_group_indices(labels, outer_group, validation_group)
    return indices["train"], indices["val"]


def execute_counter_job(job, frame=None):
    """Idempotent job execution; completed folders are verified, never refitted.

    Returns the folder, whether it was trained or already cached, and the
    measured fitting time in seconds, which the caller uses to estimate progress.
    """
    started = time.perf_counter()
    folder = Path(job["folder"])
    if complete(folder, job["experiment_id"]):
        return str(folder), "cached", time.perf_counter() - started
    folder.mkdir(parents=True, exist_ok=True)
    frame = _FRAME if frame is None else frame
    if frame is None:
        raise RuntimeError("Worker was not initialized with prepared observations")
    outer, validation = job["outer"], job.get("validation")
    train_indices, validation_indices = job_indices(frame.outer_group, outer, validation)
    package, history, predictions = fit_counter_candidate(
        frame, train_indices, validation_indices, job["candidate"], job["config"],
        job["seed"], job.get("epochs"),
    )
    package.update(experiment_id=job["experiment_id"], outer_group=outer, validation_group=validation,
                   training_row_ids=frame.iloc[train_indices].row_id.tolist())

    temporary = folder / f"model.{os.getpid()}.tmp"
    torch.save(package, temporary)
    os.replace(temporary, folder / "model.pt")
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    write_json(folder / "job.json", job)
    write_json(folder / "result.json", {"best_epoch": package["best_epoch"],
                                        "epochs_trained": package["epochs_trained"],
                                        "optimizer_steps_trained": package["optimizer_steps_trained"]})
    files = ["model.pt", "history.csv", "job.json", "result.json"]
    if predictions is not None:
        pd.DataFrame({"row_id": frame.iloc[validation_indices].row_id.to_numpy(),
                      "target": frame.iloc[validation_indices].weeklyRates.to_numpy(),
                      "prediction": predictions}).to_csv(folder / "validation.csv", index=False)
        files.append("validation.csv")
    mark_complete(folder, job["experiment_id"], files)
    return str(folder), "trained", time.perf_counter() - started
