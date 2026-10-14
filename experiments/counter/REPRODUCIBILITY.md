# Counter paper reproducibility package

This directory contains the Counter source code, frozen audit inputs, selected
model packages, numerical reports, and publication figures used by the paper
and Supporting Information. The Europe projection uses the canonical
five-fold, five-seed value-only autoregressive ensemble with hidden widths
`[64,32]`.

## Environment

Run commands from the repository root with the project source on `PYTHONPATH`.
The recorded analyses used Python 3.12 and the environment listed in the
repository setup documentation.

```bash
export PYTHONPATH="$PWD/src"
```

## Integrity and result checks

The fixed-protocol tests cover preprocessing, held-out splits, model reloads,
losses, and canonical autoregressive package validation:

```bash
python -m unittest discover -s experiments/counter/tests -v
```

The frozen evaluation tables are in `results/evaluation/`, the model-selection
tables are in `results/selection/` and `results/reports/`, and the dedicated
autoregressive summary is:

```text
results/autoregressive_jobs/comparison_64_32.json
```

## Figures and paper

The Counter figure scripts consume the committed frozen result tables:

```bash
python experiments/counter/img_for_paper/make_scatter.py \
  --output experiments/counter/results \
  --destination /tmp/aiedes-counter-figures
python experiments/counter/img_for_paper/make_autoregressive_map.py \
  --input experiments/counter/results/europe_projection_autoregressive_64_32/predictions_2020_autoregressive_all_folds_monthly.nc \
  --destination paper/img
```

The manuscript and Supporting Information can be rebuilt with:

```bash
cd paper
./build.sh all
```

The committed `paper/main.pdf`, `paper/SI.pdf`, `paper/img/`, and Counter
result figures provide the publication-ready outputs corresponding to the
checked source and tables.

## Committed artifact scope

The compact package retains the processed audit observations and weather cache,
split manifest, all 25 canonical autoregressive model packages, evaluation and
selection reports, canonical weekly/monthly projection outputs, map outputs,
paper figures, and manuscript/SI sources. Redundant intermediate logs and
oversized raw input or daily-grid files are not part of the GitHub package.
