# AIedes codebase refactoring plan

## Objectives

The refactor will separate reusable AIedes functionality from experiment
orchestration and publication plotting. The target is:

- reusable code in `src/AIedes`;
- one thin experiment entry point in each experiment directory;
- plotting scripts and generated figures in `img_for_paper`;
- unchanged scientific results and reproducible commands;
- no imports that depend on the current working directory;
- tests that exercise library APIs rather than script internals.

The classifier workflow should remain behaviourally unchanged. The Counter
workflow is the first complete migration target.

## Implementation status

The Counter migration described below is now implemented. The legacy training,
reporting, projection, and plotting script copies were removed after package
imports, protocol tests, dry-run execution, projection CLI checks, and figure
CLI checks passed. Historical `counter_v2` result/checkpoint artifacts were
preserved, while its temporary source code and caches were removed.

## Target layout

```text
src/AIedes/
├── data_loader/
│   ├── counter_data_loader.py
│   ├── counter_normalization_stats.py
│   ├── counter_prepare.py
│   └── counter_splits.py
├── models/
│   └── counter_models.py
├── train/
│   ├── counter_train.py
│   ├── counter_autoregressive.py
│   ├── counter_jobs.py
│   └── counter_protocol.py
├── evaluation/
│   ├── counter_eval.py
│   ├── counter_reconstruction.py
│   ├── counter_reporting.py
│   └── counter_audit.py
├── projection/
│   ├── counter_projection.py
│   ├── counter_autoregressive_projection.py
│   └── counter_ood.py
└── utils/
    ├── counter_losses.py
    └── counter_experiment.py

experiments/counter/
├── run.py
├── config.json
├── PLAN.md
├── AUTOREGRESSIVE_TRAINING_PLAN.md
├── REFACTOR_PLAN.md
├── data/
├── results/
├── tests/
└── img_for_paper/
    ├── make_figures.py
    ├── make_ablation.py
    ├── make_scatter.py
    ├── make_tables.py
    ├── make_timeseries.py
    ├── assets/
    └── generated figures
```

The top-level `experiments/img_for_paper` directory should eventually contain
only genuinely shared or cross-experiment plotting material. Counter-specific
plots belong under `experiments/counter/img_for_paper`.

## Module ownership

### Data and preprocessing

Keep the existing Counter data loaders in `src/AIedes/data_loader`. Move only
experiment-independent functionality there. Functions must accept explicit
paths, frames, indices, and configuration objects; they must not read files
from the current working directory implicitly.

### Models and losses

Keep Counter model definitions in `src/AIedes/models/counter_models.py` and
losses in `src/AIedes/utils/counter_losses.py`. Remove diagnostic printing from
model constructors or replace it with opt-in logging.

### Training

Move the reusable parts of `experiments/counter/protocol.py` into
`src/AIedes/train/counter_protocol.py`:

- candidate-grid construction;
- candidate identity and metadata;
- fold partition handling;
- job descriptions;
- fixed-budget training orchestration;
- checkpoint/package schemas.

Keep the actual optimizers in `counter_train.py` and
`counter_autoregressive.py`. The autoregressive trainer must expose a clear
API for:

- value-only predicted lags;
- zero-start initialization;
- fixed-epoch training;
- recursive feature regeneration;
- deterministic seeds;
- history callbacks.

### Evaluation and reporting

Move the experiment-level reporting code from `experiments/counter/reporting.py`
to `src/AIedes/evaluation/counter_experiment_reporting.py`; lower-level metric
reporting remains in `src/AIedes/evaluation/counter_reporting.py`. It should
provide functions for:

- fold-level metrics;
- pooled outer-fold metrics;
- seed aggregation;
- observed-history versus recursive evaluation;
- RMSLE, RMSE, R², log-R², and classification metrics;
- comparison tables.

Reporting must not choose a model implicitly. Selection rules must be passed as
explicit arguments and recorded in metadata.

### Projection

Split the current projection scripts into library modules:

- `counter_projection.py`: climate-only grid projection;
- `counter_autoregressive_projection.py`: recursive grid projection;
- `counter_ood.py`: climate-support and OOD diagnostics.

Projection functions must return datasets or structured results. They must not
create plots, write arbitrary files, or parse command-line arguments.

The safe parallelization boundary remains:

- parallel across independent latitude bands or grid chunks;
- batch independent grid points and ensemble members;
- keep dates sequential within each autoregressive trajectory.

### Plotting

Move all publication plotting into:

```text
experiments/counter/img_for_paper/
```

Plot scripts may read completed result artifacts and write PNG/PDF/CSV files,
but they must not train models or perform scientific selection. They should
receive an explicit `--results` directory and output directory.

Map plotting currently embedded in `project_europe.py` and related scripts must
be extracted into plotting modules under `img_for_paper`.

## Experiment entry point

Replace the current script-to-script imports with a single thin
`experiments/counter/run.py` CLI. It should expose subcommands such as:

```text
python experiments/counter/run.py prepare
python experiments/counter/run.py train
python experiments/counter/run.py select
python experiments/counter/run.py replicate
python experiments/counter/run.py evaluate
python experiments/counter/run.py project
python experiments/counter/run.py figures
python experiments/counter/run.py all
```

`run.py` should:

- resolve the repository root and configuration paths;
- configure logging and reproducibility;
- call `AIedes` library functions;
- preserve resumability and completion markers;
- never contain model architecture or plotting implementation details.

The existing `run_experiment.sh` should become a compatibility wrapper around
this CLI.

## Import and packaging rules

All imports from source code should use package-qualified names:

```python
from AIedes.train.counter_protocol import grid
from AIedes.evaluation.counter_reporting import evaluate_predictions
```

Remove imports that rely on `sys.path` mutation or the current directory:

```python
from protocol import ...
from figures.make_tables import ...
from project_europe import ...
```

Add any missing package `__init__.py` files and expose only stable public APIs.
Update `pyproject.toml` so the `src` layout is the canonical installation
path. Experiment scripts should work both from the repository root and through
the installed package.

## Checkpoint and artifact schemas

Define versioned schemas for:

- model packages;
- experiment manifests;
- prediction tables;
- metric summaries;
- projection NetCDF metadata.

Every model package must record:

- candidate configuration;
- input feature names and order;
- normalization statistics;
- lag semantics and initialization policy;
- training seed and fold;
- epoch budget;
- selection rule;
- source/data hashes.

The new value-only autoregressive package must be distinguishable from the old
value-plus-availability-flag package and must fail clearly if loaded by the
wrong projection code.

## Migration phases

### Phase 0 — Freeze and baseline

- Record current git status and source hashes.
- Verify existing Counter smoke tests.
- Record baseline metrics and artifact paths.
- Do not overwrite `experiments/counter/results`.

### Phase 1 — Introduce library APIs

- Add `counter_protocol.py` to `src/AIedes/train`.
- Add projection modules under `src/AIedes/projection`.
- Move reusable reporting functions into `src/AIedes/evaluation`.
- Keep compatibility wrappers in the old script locations.

### Phase 2 — Convert the runner

- Refactor `experiments/counter/run.py` into the only orchestration entry point.
- Replace local imports with package imports.
- Make every path explicit and relative to the configured experiment root.
- Verify that smoke runs produce equivalent artifacts.

### Phase 3 — Move publication plotting

- Create `experiments/counter/img_for_paper`.
- Move plotting scripts and assets there.
- Remove plotting from projection and evaluation libraries.
- Regenerate figures into a new output directory and compare checksums or
  numerical contents where exact rendering is not stable.

### Phase 4 — Add the autoregressive experiment

- Implement value-only zero-start preprocessing in `src/AIedes`.
- Add fixed-budget five-fold jobs for `[64]` and `[64,32]`.
- Add recursive production projection support.
- Add paired reporting against climate-only and the old recursive model.

### Phase 5 — Deprecate old paths

- Keep compatibility wrappers for one transition cycle.
- Mark old script modules as deprecated.
- Remove duplicate implementations only after smoke, unit, and projection
  checks pass.
- Update README and paper-generation commands.

## Tests required at each phase

### Unit tests

- candidate identity and grid generation;
- split disjointness and fold coverage;
- normalization fitted only on training rows;
- model package save/load;
- value-only zero-start lag construction;
- recursive rollout determinism;
- projection batching equivalence;
- metric aggregation and seed/fold pooling.

### Integration tests

- synthetic Counter smoke run;
- one real fold with two epochs;
- complete five-fold autoregressive test with a reduced epoch budget;
- production projection on a small climate grid;
- plotting from saved artifacts without retraining.

### Regression checks

For the old Counter pipeline, compare before/after:

- selected configuration IDs;
- fold membership;
- test prediction arrays;
- pooled metrics;
- recursive trajectory outputs;
- figure input tables.

Small floating-point differences are acceptable only when documented. Changes
to scientific results must not be hidden by the refactor.

## Completion criteria

The refactor is complete when:

1. `experiments/counter/run.py` is the only training/projection entry point.
2. No Counter library code imports from `experiments`.
3. No experiment module mutates `sys.path` to import sibling scripts.
4. Plotting code is confined to `img_for_paper`.
5. Existing Counter results can be reproduced or verified from their saved
   manifests.
6. The new autoregressive experiment has versioned packages and pooled
   five-fold reports.
7. The Europe projection can load the winning model without importing plotting
   code.
8. Unit, smoke, integration, and regression checks pass.
