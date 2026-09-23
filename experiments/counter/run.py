#!/usr/bin/env python
"""Run the Counter revision fixed-budget experiment; see README.md."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from AIedes.train.counter_jobs import execute_counter_job, initialize_counter_worker
from AIedes.utils.counter_experiment import complete

def format_duration(seconds):
    """Compact human-readable duration."""
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(float(seconds), 0.)
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds/60:.1f} min"
    if seconds < 172800:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} days"

def estimate_remaining(durations, remaining, workers):
    """Rough ETA: mean of the most recent finished jobs, scaled by parallelism.

    Deliberately crude. It assumes the remaining jobs behave like the recent ones
    and that every worker stays busy, so it moves as early stopping changes the
    typical job length. It is an indication, never a guarantee.
    """
    if not durations or remaining <= 0:
        return None, None
    mean = sum(durations) / len(durations)
    return mean, mean * remaining / max(int(workers), 1)

def run_jobs(jobs, output, workers, window=50, job_function=execute_counter_job):
    """Bounded parallel execution; verified completed jobs are never repeated."""
    pending = [j for j in jobs if not complete(j["folder"], j["experiment_id"])]
    print(f"Jobs: {len(jobs)} total, {len(jobs)-len(pending)} verified complete, {len(pending)} remaining", flush=True)
    if not pending:
        return
    pool = ProcessPoolExecutor(max_workers=workers, initializer=initialize_counter_worker,
                               initargs=(str(Path(output) / "data_audit/observations.pkl"),))
    futures, iterator = {}, iter(pending)
    finished, started, last_update = 0, time.monotonic(), time.monotonic()
    durations = deque(maxlen=window)

    def report():
        elapsed = time.monotonic() - started
        remaining = len(pending) - finished
        mean, eta = estimate_remaining(durations, remaining, workers)
        message = (f"Completed {finished}/{len(pending)} new jobs; running/queued {len(futures)}; "
                   f"elapsed {format_duration(elapsed)}")
        if remaining <= 0:
            message += "; stage finished"
        elif mean is None:
            message += "; estimating remaining time"
        else:
            finish = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + eta))
            message += (f"; mean job {format_duration(mean)} (last {len(durations)}); "
                        f"~{format_duration(eta)} left, estimated finish {finish}")
        print(message, flush=True)

    try:
        for _ in range(min(workers * 2, len(pending))):
            job = next(iterator)
            futures[pool.submit(job_function, job)] = job
        while futures:
            done, _ = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                try:
                    _, status, seconds = future.result()
                except BaseException as exc:
                    raise RuntimeError(f"Job failed: {job['folder']}: {exc}") from exc
                if status == "trained":
                    durations.append(seconds)
                finished += 1
                replacement = next(iterator, None)
                if replacement:
                    futures[pool.submit(job_function, replacement)] = replacement
            if done or time.monotonic() - last_update >= 30:
                report()
                last_update = time.monotonic()
        pool.shutdown(wait=True)
    except BaseException:
        for process in list(pool._processes.values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise

if __name__ == "__main__":
    from protocol import main
    main()
