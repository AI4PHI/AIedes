# Counter revision — running the experiment

`PLAN.md` defines the agreed protocol. The runner implements 516 configurations,
fixed 3,000-epoch training and five trap-disjoint 80/20 folds. All runs are retained.
It also performs seed replication, the focused one-lag comparison (25 new fits),
checkpoint sensitivity, five full-data deployment fits, external validation and
non-map paper figures. Country-specific performance analysis is excluded.

## Launch

```bash
cd /home/biazzin/git/AIedes/experiments/counter
./run_experiment.sh --dry-run
./run_experiment.sh
```

The second command runs in the foreground. For a run that survives closing the terminal:

```bash
tmux new-session -s counter-revision
cd /home/biazzin/git/AIedes/experiments/counter
./run_experiment.sh
```

Detach with Ctrl-b then d; reconnect with `tmux attach -t counter-revision`.

The launcher uses `aiedes-env`, reserves one physical CPU core, sets one thread per
worker and runs at niceness +19. It uses the remaining physical cores as workers
(79 on the current machine). Override with `--workers 16`, for example, or set
`PYTHON=/path/to/python` to use another environment. Output is appended to
`results/run.log`; progress includes completed fits and a rough time estimate.

The full run contains 2,775–2,815 fits; the exact seed-replication count depends on
how many selected configurations overlap. The earlier timing estimate is roughly
9 hours, plus the one-lag batch; the live estimate uses measured completed jobs.

## Resume and stages

Repeat the same launch command to resume. Completed artifacts are verified by
SHA-256 and reused, including seed-1 grid fits used in the replication stage.
Interrupted fits restart from the beginning; there is no mid-epoch resume.
Another process cannot use the same output directory concurrently.

```bash
./run_experiment.sh --status
./run_experiment.sh --stage prepare    # validate real inputs; no training
./run_experiment.sh --stage paper      # 1,620 fits: 90-day inputs, three losses
./run_experiment.sh --stage windows    # 960 fits: 30/60-day inputs, WMSLE
./run_experiment.sh --stage select     # requires both grid blocks
./run_experiment.sh --stage replicate
./run_experiment.sh --stage supplement # 25 one-lag + 25 checkpoint-sensitivity fits
./run_experiment.sh --stage deploy     # five seeds trained on all rows
./run_experiment.sh --stage evaluate
./run_experiment.sh --stage figures
```

With no `--stage`, all stages run in order. `--output /path/to/results` selects a
separate experiment directory. Changes to code, config, data or recorded package
versions require a new output directory; the runner refuses to mix experiments.
`--status` counts completion markers only; resume performs the integrity checks.

To regenerate figures from a completed experiment after plotting changes, run
from this directory using the experiment's Python environment:

```bash
/home/biazzin/conda-envs/aiedes-env/bin/python -m figures.make_figures --output results --regenerate
```

This restores the original Counter layouts, colors, labels and twelve-trap
selection using the revised results. Axis limits expand when needed. The main
prediction model is the selected WMSLE configuration `8bec14a3ec4c`: 90-day daily
temperature, precipitation and dewpoint plus both egg-rate lags. Scatter panels
show in-sample and out-of-fold ensembles; the time series retain the original
location basemap from a cached image. Historical figure versions are preserved
under `additional_results_not_for_commit/pilot_results/figure_versions/`;
plotting sources and notes accompany the new figures.
Training provenance, checkpoints and saved evaluation predictions are preserved.

## Outputs

- `experiment.json`, `grid.json`, `source_snapshot/`: protocol, input checksums,
  environment and source code used for the experiment.
- `data_audit/`: frozen fold membership, observations, daily climate coverage,
  recursive-eligibility flags and exclusion details.
- `grid/`, `final_models/`: checkpoints, training histories, fold predictions and
  metrics for every fit. `selection/grid_runs.csv` is the initial all-grid table;
  `reports/all_runs.csv` adds the extra seeds without duplicating seed 1.
- `selection/`: all configuration summaries, top-five WMSLE, best per loss,
  climate-only comparator and the selected two-lag reference.
- `lag_order/`: the 25 one-lag fits. Their two-lag reference is reused from
  `final_models/`. Inputs and hyperparameters otherwise stay identical.
- `sensitivity/`: 25 fits with three checkpoints per fit on the same 3/1/1 groups:
  fixed final, best raw validation error, and log-error early stopping.
- `evaluation/`: pooled observed-lag and recursive predictions, daily trajectories,
  matched-cohort metrics and whole-trap bootstrap intervals.
- `reports/`: CSV/TeX tables, paired lag-order differences, separate VectorNet
  predictions and metrics, training-seed variability, and `RESULTS.md`.
- `deploy/`: five full-data model packages for the later Europe-map work.
- `export/`: held-out-fold model ensembles and their manifests.
- `figures/`: dropout/loss ablations, window sensitivity, train/test scatter,
  Bland–Altman and twelve-trap time series, with notes for updating captions.

Tables average training seeds within each fold before calculating mean ± SE
across folds. Undefined metrics remain missing. VectorNet results are never
pooled with AIMSurv. ZINB is evaluated on standardized weekly rates, which may
be fractional; this is a count-inspired loss comparison.

The Europe map is deferred until the daily European climate grid is available.
It is not required to run any of the stages above. Figure generation is offline.
The earlier pilot and exploratory diagnostics are preserved locally under
`additional_results_not_for_commit/`, which is intentionally excluded from version
control. New scientific results are written to `results/reports/RESULTS.md`.

## Checks

```bash
./run_experiment.sh --smoke --output /tmp/counter-revision-smoke
/home/biazzin/conda-envs/aiedes-env/bin/python -m unittest discover -s tests -v
```

The smoke run uses synthetic data, two epochs and two seeds, exercises every stage,
and defaults to four workers. Its outputs are explicitly marked synthetic.
