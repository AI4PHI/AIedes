# Counter autoregressive-training extension

## Scientific question

The original Counter experiment selected `8bec14a3ec4c` as the main WMSLE
model. It was trained with observed egg-rate lags, but its recursive deployment
performance was lower than the climate-only model. This extension tests whether
training directly on model-generated previous counts removes that deployment
gap.

The experiment is deliberately separate from the original 516-configuration
grid. It keeps the original Counter protocol: fixed 3,000 epochs, Adam, no
validation-based checkpoint selection, five trap-disjoint outer folds, and all
five training seeds.

## Frozen candidates

The selected historical reference is `8bec14a3ec4c`:

```json
{
  "fields": ["t2m_mean", "d2m_mean", "tp_sum"],
  "weekly": false,
  "window": 90,
  "lags": true,
  "lag_order": 2,
  "loss": "WMSLE",
  "learning_rate": 0.001,
  "dropout": 0.2
}
```

Only the hidden architecture changes:

```text
candidate A: [64]
candidate B: [64, 32]
```

Both candidates use value-only lag inputs. The two availability flags used by
the historical model are removed. Thus the input is 270 climate values plus
two predicted lag values, not 274 climate/lag/value-flag channels.

## Zero-start recursive training

For every trap, the trajectory starts at the beginning of the season with zero
eggs. The raw zero is normalized with the training-fold lag statistics:

```text
normalized_zero = (0 - lag_mean) / lag_std
```

After each prediction, only the model prediction is used as future lag input.
Observed `prev1_rates` and `prev2_rates` are never supplied to the model.

The training loop is:

```text
for epoch in 1..3000:
    generate recursive training features with current weights
    train on that generated feature table with minibatch WMSLE
    update weights
```

The training feature table is regenerated after every epoch. No observed lag
values or availability flags are used. For a literal January start, the
production/reconstruction path must generate any required January warm-up
climate rows before the first retained observation; the first implementation
must record how many warm-up rows are available per trap.

## Splits and selection

Use the existing `balanced_seed42.json` outer trap groups. For each outer fold:

- train on the other four trap groups;
- recursively evaluate the held-out trap group from zero initialization;
- repeat for seeds 1–5;
- pool outer-fold predictions so every observation is evaluated once.

There is no validation split and no early stopping. Checkpoints are the fixed
3,000-epoch final models, matching the original Counter experiment.

Because the loss is WMSLE, the primary comparison metric should be recursive
RMSLE (the square root of log-MSE). Raw R², RMSE, log-R², and classification
accuracy remain secondary metrics. The climate-only and historical models must
be evaluated on the same outer-fold rows.

The final deployment comparison is made from pooled five-fold predictions after
averaging the five seeds within each fold, never from a best fold, seed, or
checkpoint. The [64,32] candidate is used for the Europe projection because it
improves raw-scale R² and RMSE over the climate-only comparator while its
log-scale metrics remain effectively unchanged. It is reported as a
deployment-oriented candidate, not as a claim of global model optimality.

## Files to add

### `counter_autoregressive_experiment.py`

Experiment entry point and resumable job runner. It should:

- load the original weekly data and frozen split manifest;
- create the two frozen candidates above;
- create one job per candidate × outer fold × seed;
- fit fold-specific normalization statistics on training rows only;
- call the closed-loop trainer;
- save a model package, history, metadata, and completion marker;
- support `--workers`, `--candidate`, `--fold`, `--seed`, `--epochs`, and
  `--dry-run`.

Suggested output:

```text
results/autoregressive_training/
  experiment.json
  candidates.json
  jobs/<candidate>_fold<fold>_seed<seed>/
    model.pt
    history.csv
    test.csv
    job.json
    result.json
    done.json
  reports/
    outer_predictions.csv
    summary.csv
    comparison.csv
```

### `autoregressive_features.py`

Shared value-only recursive feature construction. It should expose:

- `fit_autoregressive_preprocessing(...)`;
- `transform_climate_features(...)`;
- `recursive_feature_table(...)`;
- `zero_start_metadata(...)`.

This keeps feature construction identical between training, held-out
evaluation, and Europe projection.

### `report_autoregressive.py`

Aggregate the 25 new fits and compare them with the saved historical results.
It should report, for each candidate and evaluation mode:

- mean and standard error across outer folds;
- pooled R², log-R², RMSE, RMSLE, accuracy, sensitivity, and specificity;
- per-fold and per-seed values;
- paired differences versus climate-only and the old recursive model;
- the number of valid January warm-up trajectories.

### `counter_autoregressive_projection.py`

Production adapter for the winning value-only model. It should:

- load the new model package schema;
- initialize every grid-cell trajectory at zero in January;
- use only predicted lag values afterward;
- batch independent grid points and ensemble members;
- parallelize independent latitude bands with processes;
- preserve sequential ordering over dates;
- write daily, weekly, monthly NetCDF files and the Europe maps.

The existing `project_europe_autoregressive.py` already demonstrates the safe
parallelization boundary: latitude bands and grid points are parallel/batched;
dates remain sequential within each trajectory.

### Tests

Add `tests/test_autoregressive_training.py` covering:

- no observed lag values reach the model;
- zero initialization is normalized correctly;
- first and second lag transitions are correct;
- recursive features are reproducible after reload;
- candidate input dimensions are 272;
- fold groups do not overlap;
- batched production inference matches ordinary model inference;
- a synthetic trajectory remains finite and non-negative.

## Execution order

1. Implement and test value-only zero-start feature construction.
2. Run a two-epoch synthetic smoke test.
3. Run one real fold and one seed for both architectures.
4. Run all 25 fits for `[64]`.
5. Run all 25 fits for `[64,32]`.
6. Aggregate pooled five-fold comparisons.
7. Select the winning architecture using the predeclared metric rule.
8. Refit the winner on all observations for five seeds, still for 3,000 fixed
   epochs and zero-start recursive training.
9. Run the value-only production projection and generate the maps.

## Required comparison table

The final report must contain:

```text
climate-only recursive reference
old observed-lag model in recursive mode
new zero-start [64]
new zero-start [64,32]
```

All four must be scored on the same pooled outer-fold observations. The report
must clearly distinguish observed-history scores from recursive scores and
must not select a checkpoint by repeatedly inspecting the outer test results.
