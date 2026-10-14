# Classifier paper workflow

This directory contains the reproducible analyses used for the
AIedes-Classifier results in the manuscript. The legacy `run_exp.py` and the
notebooks under `results/` are retained temporarily for historical reference;
they are not the source of the revised paper metrics or figures.

## Environment

Create and activate the repository environment, install PyTorch for the local
platform, and install the package in editable mode:

```bash
conda env create -f environment.yml
conda activate aiedes-env
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

The commands below assume they are executed from the repository root. The
source table is the tracked archive
`experiments/classifier/data/ecdc_albopictus_cordex_2020.zip`.

## Complete run

```bash
AIEDES_JOBS=32 experiments/classifier/run_all_paper_results.sh
```

Each model fit uses one CPU thread. `AIEDES_JOBS` controls the number of
independent fits run concurrently. Existing `paper_results` are moved to a
timestamped local backup before a complete run.

## Analysis stages

The complete launcher performs the following operations:

1. checks the source data and constructs the climate-filtered binary target;
2. creates a fixed stratified 80/20 random split;
3. compares 14 architectures and two input sets by five-fold validation within
   the 80% training partition (140 fits);
4. evaluates five seeds of the selected model on the untouched 20% partition;
5. creates full-coverage random out-of-fold predictions for the no-hidden-layer
   logistic-equivalent model and selected neural network, using five seeds per
   fold;
6. repeats the two-model comparison while withholding complete NUTS3 regions;
7. renders the main random comparison, SI NUTS3 comparison, and model-selection
   figures from saved result tables.

The selected configuration is held fixed during the random out-of-fold and
NUTS3 comparisons. The NUTS3 experiment is a spatial-transfer evaluation, not
another architecture search.

To rerun only the two full-coverage comparisons after selection has completed:

```bash
python experiments/classifier/run_paper_experiments.py \
  --stage comparisons \
  --data experiments/classifier/paper_results/input/copernicus_classifier_prepared.csv.gz \
  --output-dir experiments/classifier/paper_results \
  --epochs 500 \
  --batch-size 4096 \
  --jobs 32
```

Completed comparison fits are checkpointed under
`paper_results/checkpoints/`, so the command resumes after interruption.

## Principal outputs

- `selected_configuration.json`: selected input set and architecture.
- `inner_validation_metrics.csv`: architecture/input results for every fold.
- `main_outer_test_ensemble_metrics.json`: untouched 20% test performance.
- `main_random_oof_map_predictions.csv`: full-coverage random held-out
  predictions for logistic and AIedes models.
- `main_random_oof_map_metrics_by_fold.csv`: random per-seed and ensemble fold
  metrics.
- `main_random_oof_map_metrics_pooled.csv`: random pooled map metrics.
- `nuts3_held_region_predictions.csv`: full-coverage predictions from models
  that did not train on the corresponding NUTS3 region.
- `nuts3_metrics_by_fold.csv`: NUTS3 per-seed and ensemble fold metrics.
- `nuts3_metrics_pooled.csv`: pooled held-region metrics for both models.
- `nuts3_ensemble_summary.csv`: mean and standard deviation across regional
  folds for both models.
- `nuts3_split_manifest.csv`: NUTS3 regions assigned to each fold.

All row-level prediction tables include stable row identifiers, coordinates,
the target, and fold assignment. Feature scaling is fitted separately using
only the training observations in each fit.

## Figure generation

```bash
python experiments/classifier/img_for_paper/make_classifier_comparison_map.py \
  --evaluation random

python experiments/classifier/img_for_paper/make_classifier_comparison_map.py \
  --evaluation nuts3

python experiments/classifier/img_for_paper/make_combined_loss_acc.py
```

The map script uses Natural Earth coastlines and borders from the local Cartopy
cache. The generated paper files are written directly to `paper/img/`.

## Files intended for version control

Commit the scripts, documentation, compact CSV/JSON metrics, split manifests,
paper-facing prediction tables, and final figures. Do not commit prepared input
copies, model weights, per-fit checkpoints, partial files, timestamped backups,
Python caches, or LaTeX build intermediates.
