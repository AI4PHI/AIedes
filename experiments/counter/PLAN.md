# Counter experiment — complete plan

Purpose: rerun every Counter result shown in the paper under one current protocol that
answers the reviewers. Classifier results are untouched.

## What the reviewers require

| # | Request | How this plan answers it |
|---|---|---|
| R1.1 | Stricter blocked validation (by trap) | 5 trap-disjoint folds; every trap is test exactly once |
| R1.2 | Drop the top-25% filter; report all runs | All runs reported, mean ± SE; no filtering |
| R1.4 | Separate observed-lag hindcast from recursive forecast | Both scored on identical rows |
| R1.2b | Report VectorNet external validation separately | Separate table, 2021 VectorNet traps |
| R2.1 | Complete ablation: windows, lags, dropout, depth, width, losses | Grid below, plus a focused one-lag versus two-lag comparison |

Terminology, masking and interpretation edits are text-only and out of scope here.
Country-specific performance analysis is excluded from this experiment; the author
will explain this scope decision in the response to Reviewer 2.

## Data (in `data/`, checksums recorded in every run)

| file | use | sha256 (first 16) |
|---|---|---|
| `eggs_y_norm_7days.pkl` | training/test: 5,129 observations, 387 traps, 2020 | `7b55357d50deb7dd` |
| `eggs_y_norm.pkl` | dated daily climate for recursive simulation | `8d75bb6b12dd9355` |
| `eggs_y_norm_7days_vectorNet.pkl` | external validation: 280 observations, 40 traps, 2021 | `9c3cf34077790cf0` |
| `AIMSurv_albopictus_2020_era5_land.pkl` | provenance only, not read during training | `1598e2431727dce2` |
| `balanced_seed42.json` | frozen trap-group membership | `0a27eaf4ac9e273a` |
| `dataset_source_code/` | scripts that built the weekly dataset | — |

Counts to correct in the paper: **5,129 rows / 387 traps** (not 5,432 / 395); 4,103–4,104 train
and 1,025–1,026 test per fold (not 4,345).

## Splits

The five existing balanced trap groups (77–78 traps, 1,025–1,026 observations, positive
share 0.199–0.202). For fold *k*: train on the other four groups (4,103–4,104 rows, approximately 80%),
test on group *k* (1,025–1,026 rows, approximately 20%). Same proportions as the paper, now blocked by trap;
no trap on both sides. Pooling the five test folds yields out-of-fold predictions for all
5,129 observations, each from a model that never saw its trap. No separate validation
group, preserving the agreed 80/20 proportions and fixed training budget.

## Grid — 516 configurations

Common axes: **3 dropouts** (0.0, 0.2, 0.4) × **4 architectures** ([32], [64], [32,16],
[64,32], i.e. the paper's `num_layers 2/3 × hidden 32/64`).

| block | losses | input sets | configs |
|---|---|---|---:|
| Paper block (90-day windows) | WMSE (lr 1e-4), WMSLE (lr 1e-3), ZINB (lr 5e-4) | the paper's 9 | 324 |
| Window block | WMSLE only | 30 d and 60 d variants, 16 sets | 192 |

The nine 90-day sets are exactly the paper's: {T+P, T+D+P} × {daily, weekly means} ×
{with, without lags}, plus lag-only. × 5 folds = **2,580 fits**. Then 4 extra seeds × 5
folds for the reported configurations: top five WMSLE, best per loss, the climate-only
comparator and the two-lag reference. Deduplicate overlapping choices: 7–9 distinct
configurations require **140–180 additional fits**, giving 25 runs each for the tables.

**Focused lag-order test:** select the best two-lag WMSLE configuration with climate
inputs by mean fold R² and include it among the seed-replicated configurations above.
Train its one-lag variant using only the previous-week egg rate, keeping climate inputs,
window, aggregation, architecture, dropout, learning rate, epochs, folds and seeds fixed.
This adds **5 folds × 5 seeds = 25 fits**; reuse the two-lag fits. Report the paired
comparison for this configuration. The 516-configuration grid is unchanged and already
includes no-lag versus two-lag input ablations.

## Training

Fixed **3,000 epochs**, Adam, batch 64, linear LR decay 1→0.9 over 3,000 epochs, gradient
clipping 10, final checkpoint. Dropout is the only regulariser, as in the paper. **No
early stopping.**

Rationale, measured on this data: early stopping on validation log error costs 0.104 raw
R² and 0.012 accuracy while gaining 0.010 log R², so by the paper's own combined criterion
it yields a worse model. A fixed uniform budget also makes no implicit choice between raw
and log scales, whose optima sit 3–6× apart in epochs.

Dropping the top-25% filter is justified empirically: seed-to-seed SD of test R² is
0.002–0.010 on fixed trap folds versus 0.052–0.071 on random splits. The filter removed
split variance, not badly trained runs, so nothing remains to exclude.

Preprocessing is fitted on each fold's training rows only and saved with the model.
Missing lags carry an availability flag distinct from zero. Corrected WMSLE bin-edge
weighting throughout.

## Metrics

Per fit on its test fold: raw R², log R², RMSE, log error, presence/absence accuracy
(observed > 0 vs predicted ≥ 1 egg/week), sensitivity, specificity, slope, always-absent
baseline; train R² for the overfitting check. Reported as mean ± SE across folds, all runs
included. Undefined quantities on degenerate subsets are null, never 0.

## Analyses

1. **Ablation**: every configuration's fold-averaged metrics — inputs, window, lags,
   dropout, depth, width, loss.
2. **Selected model**: best mean test R² across folds, seed-replicated.
3. **Pooled out-of-fold predictions**: all 5,129 observations from the five fold models,
   seed-averaged — basis for the scatter, Bland–Altman and trap time series.
4. **Hindcast vs recursive**: identical rows, observed lags versus the model's own
   recursive predictions, plus a climate-only comparator; traps with continuous daily
   climate only, coverage and exclusions reported.
5. **External validation**: VectorNet 2021 (280 observations, 40 traps), scored by the
   fold ensemble, reported separately and never pooled with AIMSurv.
6. **Deployment model for the map**: refit on all 5,129 rows, 5 seeds, same fixed budget.
   Its accuracy is quantified by the cross-validation, not by a held-out set.
7. **One versus two lags**: focused WMSLE comparison described above, using the same
   five folds and five seeds; 25 additional one-lag fits.

## Figure and table reproduction

Legacy notebooks are copied to `figures/legacy/` for reference. They read
`filtered_results.pkl` written by `create_complete_df.py`, whose grouping is
`input_variables × num_layers × hidden_dim × dropout` and which **applies the top-25%
filter** — the exact rule the reviewer objected to. Replacement: the run emits a tidy
`runs.csv`, one row per loss/configuration/fold/seed, using the legacy column names
(`input_variables`, `num_layers`, `hidden_dim`, `dropout`, `test_r2`, `test_loss`,
`test_rmse`, `test_binary_accuracy`, `train_r2`) plus `loss`, `fold`, `seed`, `window`.
The ablation plotting code then works with the filter step deleted and its label mapping
intact, provided `input_variables` keeps the legacy string form — fields joined by `"; "`,
weekly means marked with `(mean)`, lags as `p1zn; p2zn`, e.g.
`t2m_mean_znorm(mean); d2m_mean_znorm(mean); tp_sum_znorm(mean); p1zn; p2zn`. The nine
90-day sets reuse the existing nine mapping keys; the 30/60-day variants need new keys and
a window column for the new panel.

Code in `figures/`, run through `run.py --stage figures` after evaluation, with no
manual notebook execution. `protocol.py` implements training and orchestration;
`reporting.py` writes the analysis tables through `figures/make_tables.py`.
The complete non-map workflow is launched with `./run_experiment.sh`.

For completed results, plotting changes are applied with
`python -m figures.make_figures --output results --regenerate` from this directory.
This separate figure-only command verifies the existing results and inference
sources, archives previous figures and records the new plotting sources without
changing the training provenance. The figures preserve the original Counter
layouts and colors, including dropout columns, the selected WMSLE star, train/test
scatter panels and the original twelve traps with their cached location basemap.
Labels identify the revised ensembles; limits expand only as needed. The selected
prediction model remains daily 90-day temperature, precipitation and dewpoint
plus both egg-rate lags (`8bec14a3ec4c`, WMSLE).

| paper item | file | script | status |
|---|---|---|---|
| Main Fig. 5 | `nn_results_by_dropout.pdf` | `make_ablation.py` | implemented |
| Main Tab. 2 | top-five configurations | `make_tables.py` | implemented |
| Main Fig. 6 | `R2_scatter_plot.pdf` | `make_scatter.py` | implemented |
| Main Fig. 7 | `most_populated_traps_time_series.pdf` | `make_timeseries.py` | implemented; offline time-series panels |
| Main Fig. 8 | `weekly_rates_map.png` | `make_map.py` | **blocked, see below** |
| SI Fig. | `nn_results_by_dropout_3loss.pdf` | `make_ablation.py` | implemented |
| SI Fig. | `nn_results_by_dropout_3loss_acc.pdf` | `make_ablation.py` | implemented |
| SI Tab. | best configuration per loss | `make_tables.py` | implemented |
| SI Fig. | `bland_altman_test.pdf` | `make_scatter.py` | implemented |
| new Tab. | hindcast vs recursive | `make_tables.py` | implemented |
| new Tab. | VectorNet external validation | `make_tables.py` | implemented |
| new panel | window sensitivity 30/60/90 | `make_ablation.py` | implemented |
| new Tab. | one-lag versus two-lag WMSLE comparison | `make_tables.py` | implemented; 25 additional fits |
| SI note | early-stopping sensitivity | `make_tables.py` | implemented |

**Blocker — Main Fig. 8.** Regenerating predictions requires a Europe-wide daily
climate grid, `data/europe_copernicus_data_2020.nc`, which is missing locally.
The legacy notebook also opens cached `predictions_2020_weekly_rates.nc` before its
prediction-generation cell; the replacement must generate new predictions before
plotting them rather than require that old cache.

**ZIP dependency correction (2026-09-22).** In `figures/legacy/weekly_rates_map.ipynb`,
code cell 4 (zero-based notebook cell index) reads
`albopictus_presence_absence_ecdc_copernicus_2020.zip`, combines `presence_numeric`
and `Suitable` into a label, and feeds monthly climate features to a saved classifier.
It creates `df_temp` with suitability scores and a threshold of 0.3. No later cell
uses `df_temp`, those scores, or those labels: cell 5 masks land and lakes using
Natural Earth and coordinates; cell 13 predicts Counter abundance using daily climate
and recursive egg-rate lags; cell 7 plots `WeeklyRates`. Thus the ZIP is an unused
classifier dependency in this notebook, **not an applied presence mask**, and is not
required for Counter training or this map's abundance predictions. Omit that unused
classifier cell from the replacement. Any future suitability-masked map would need
an explicit masking implementation and separately documented inputs.

Environment: `xarray`, `cartopy`, `regionmask`, `geopandas`, `contextily`, `pyproj`,
`netCDF4`, `seaborn` are installed; `pycountry` 26.2.16 was installed in `aiedes-env`
on 2026-09-22 for country labels and added to the dependency declarations;
the new non-map figures use no basemap downloads. `cartopy` and `contextily` remain
legacy/map dependencies.

## Compute and staging

~11–12 min per fit measured at 4,103 rows (WMSE 7.8–11.7, WMSLE 7.9–15.2, ZINB 11.0–15.5),
79 workers, one thread each, niceness +19.

| stage | fits | estimate |
|---|---:|---:|
| 1. Paper block, 3 losses × 9 input sets × 3 dropouts × 4 architectures | 1,620 | 5 h |
| 2. Window block, WMSLE at 30/60 d | 960 | 2.5–3 h |
| 3. Seeds for 7–9 distinct reported configurations | 140–180 | ~0.5 h |
| 4. Deployment refit on all rows, 5 seeds | 5 | minutes |
| 5. Evaluation, external validation, figures | — | <1 h |
| 6. Early-stopping sensitivity (SI): selected configuration, 3/1/1 arrangement, three checkpoints per fit | 25 | 0.2 h |
| 7. One-lag variant of the selected two-lag WMSLE configuration | 25 | ~10–20 min with available workers |

Total ≈ 9 h plus the one-lag comparison (~5 CPU-hours, or ~10–20 min with available
workers, based on earlier fit timings). The full grid is declared up front so the experiment identity is fixed;
stages are job filters, so resume stays valid. Selection and final figures follow both
grid blocks. Total: **2,775–2,815 training fits**, including seed replication, the one-lag
comparison, checkpoint sensitivity and deployment. Disk usage depends on model sizes;
allow several GB for checkpoints, histories, predictions and figures.

## Reproducibility

Each run records config, dataset checksums, environment, and a hash plus snapshot of every
source file. Completed jobs are verified by artifact checksum and never refitted; changed
code or config requires a new output directory. All fits are single-threaded, seeded and
deterministic.

## Directory layout

```
counter/
    PLAN.md  README.md  config.json  run.py  run_experiment.sh
    data/            copied datasets, split manifest, dataset build scripts
    data_audit/      read-only dataset audits
    figures/         figure scripts; figures/legacy/ holds the original notebooks
    results/         this experiment
    additional_results_not_for_commit/  local exploratory archive; excluded from commits
    tests/
```

## Open items

1. **Fig. 8 map**: locate or supply `europe_copernicus_data_2020.nc`. The Copernicus
   presence/absence ZIP is not required by the Counter map computation.
2. Launch the implemented experiment using the commands in `README.md`. The full
   scientific training run has not been started by the implementation checks.

Completed on 2026-09-22: installed `pycountry`; the authoritative revised run is
under `results/`. Earlier pilot outputs and exploratory recursive-training tests are
preserved under `additional_results_not_for_commit/` and are excluded from commits.
