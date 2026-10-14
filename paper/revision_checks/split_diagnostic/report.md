# Counter dataset composition and trial trap splits

Diagnostic performed 2026-09-19. No models were trained. The revision plan and
production training/splitting code were not changed. The candidate allocation
below is a proposal supported by data checks, not an adopted training protocol.

## Finding

Five-fold evaluation by whole traps is feasible with closely balanced sample
counts, positive fractions, broad egg-rate categories, and country observation
counts. An explicit country-balancing objective improves geographic consistency
compared with stratifying only on egg-rate categories.

However, large egg counts remain concentrated in a few traps. Good composition
balance does not establish stable R² or equal prediction difficulty across folds.
The remaining variability must be reported when models are trained.

## Dataset composition

- **5,129 observations, 387 traps, 13 countries**, collection end dates from
  2020-01-03 to 2020-12-30.
- **4,100 zero observations (79.94%)**, 1,029 positive (20.06%).
- **290 traps have only zeros**; 97 have at least one positive observation.
- Observations per trap: median 15, range 1–33.
- Positive weekly egg rates: median 18, quartiles 6 and 51, maximum 661.5.
- Missing first and second lag values: 680 and 1,091 observations respectively.
  This diagnostic does not drop them; production evaluation must preserve the
  intended lag missingness/initialization policy.

| Country | Observations | Traps | Positive observations |
|---|---:|---:|---:|
| Austria | 2,798 | 167 | 1 |
| Spain | 916 | 79 | 392 |
| Albania | 455 | 15 | 312 |
| Italy | 378 | 24 | 297 |
| Greece | 277 | 17 | 0 |
| Portugal | 175 | 15 | 13 |
| Netherlands | 55 | 21 | 1 |
| Serbia | 29 | 10 | 9 |
| Moldova | 13 | 8 | 0 |
| Cyprus | 12 | 12 | 0 |
| Slovakia | 9 | 9 | 0 |
| Switzerland | 8 | 8 | 0 |
| Hungary | 4 | 2 | 4 |

Austria supplies 54.55% of all rows. Spain, Albania and Italy together supply
1,001 of the 1,029 positive observations (97.28%). Balanced splitting preserves
this sample composition; it does not correct the geographic bias of the dataset.

## Trial splitting methods

Compared two methods using seeds **42 and 1–20** (21 allocations per method):

1. Scikit-learn 1.9.0 `StratifiedGroupKFold`, five folds, shuffling enabled,
   grouping on `id_trap` and stratifying on broad egg-rate bins.
2. A custom grouped allocation balancing observation count, trap count, number
   of positive traps, the same target-bin counts, and observations by country.
   A greedy assignment is followed by a fixed budget of 30,000 proposed moves
   or swaps, accepting only improvements to a fixed composition objective.

The bins are zero, (0,6], (6,18], (18,51], (51,200], and >200 eggs/week. The first
three positive boundaries come from the full dataset's positive quartiles; 200
is an explicit tail threshold. These strata were set for this diagnostic before
examining model outcomes. A future production protocol must document that split
construction uses outcome strata; no test outcomes enter model fitting or tuning.

The custom objective is a weighted sum of squared relative deviations from equal
fold totals: rows weight 2, traps 0.3, positive traps 0.5, all six target bins
together weight 1, country counts together weight 0.5. Country expected-count
denominators have a floor of 10 so that tiny countries do not dominate the
objective. No predicted values, model scores, continuous target sums or target
variances enter this objective. Seed 42 is the prespecified example, not the best
of the tested seeds.

## Example: balanced outer test folds, seed 42

| Fold | Observations | Traps | Positive traps | Positive observations | Positive % | Observations >200 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1,026 | 78 | 20 | 207 | 20.18 | 11 |
| 2 | 1,026 | 77 | 19 | 206 | 20.08 | 12 |
| 3 | 1,025 | 77 | 19 | 204 | 19.90 | 11 |
| 4 | 1,026 | 77 | 19 | 205 | 19.98 | 12 |
| 5 | 1,026 | 78 | 20 | 207 | 20.18 | 12 |

Country observation counts are also closely matched:

| Country | Minimum–maximum observations per fold, balanced | Minimum–maximum, target-stratified only |
|---|---:|---:|
| Austria | 559–561 | 550–579 |
| Spain | 183–184 | 143–251 |
| Albania | 90–92 | 60–143 |
| Italy | 74–79 | 53–97 |
| Greece | 55–56 | 36–75 |
| Portugal | 35–35 | 20–47 |

These balance country **observations**, not country trap counts simultaneously:
Italy, for example, contributes 4–8 traps per fold. Hungary cannot occur in all
five folds because there are only two Hungarian traps. All other countries occur
in every example fold. Month-of-year proportions are broadly similar without
explicit temporal balancing (largest month-specific spread is about two
percentage points); this is not temporal holdout evaluation.

## Sensitivity to allocation seed

| Quantity across 21 allocations | Balanced allocation | Target-stratified only |
|---|---:|---:|
| Median maximum fold-size deviation from ideal | 0.078% | 0.507% |
| Worst maximum fold-size deviation | 0.117% | 1.189% |
| Median within-allocation spread of positive fractions | 0.370 percentage points | 0.622 percentage points |
| Worst spread of positive fractions | 0.761 percentage points | 1.473 percentage points |

Across all 105 balanced test folds, sizes range from 1,025 to 1,027 observations,
positive fractions from 19.61% to 20.47%, and >200 observations from 11 to 12.
These are composition checks; the split seeds are not model-training repetitions.

## Inner validation feasibility and integrity

For each example outer split, the remaining traps were allocated into three inner
folds using the same composition objective. Across the 15 inner validation folds:

- 1,367–1,368 observations per validation fold;
- positive fractions 19.96–20.18%;
- 15–16 observations above 200 eggs/week.

Checked all five outer and 15 inner allocations: every trap stays whole, every
row occurs exactly once as an outer test observation, outer test traps never
enter inner training/validation, and inner training/validation traps are disjoint.

For clarity, this nested three-fold diagnostic uses roughly 53% of the full data
for inner training, 27% for inner validation and 20% for outer testing. It is not
the existing v2 70/15/15 holdout. The number of inner folds, refitting policy and
training seeds remain to be agreed before production.

## Remaining variability and scope

The largest 1% of egg rates (values above the 99th-percentile threshold) account
for **66.8% of the total squared deviations from the overall mean**. These are
exactly the target deviations used in the denominator of pooled R².

For the example balanced folds, mean rates range from 8.62 to 10.11 and target
standard deviations from 33.57 to 45.45. The ratio of largest to smallest within-
fold target sum of squares is 1.83. Across the 21 seeds its median is 1.81, versus
1.67 for target-stratified-only splitting: the custom allocation improves counts
and geographic balance, but does not improve every measure of distributional
balance. It is not evidence that it will reduce R² fluctuations.

Once trained, report pooled out-of-fold predictions as well as per-fold results
and training-seed variability. Do not force identical target moments or discard
folds with poor scores. Stratification reduces some composition variability;
fold-level SD alone is not a complete estimate of uncertainty.

Trap splitting tests unseen traps within the sampled geographic mixture. It does
not establish transfer to unseen countries, remove dependence between nearby
traps, or balance the original countries' surveillance coverage.

## Files and reproduction

Run from the repository root:

```bash
/home/biazzin/conda-envs/aiedes-env/bin/python experiments/counter_v2/diagnose_splits.py
```

`dataset.json` contains the source SHA-256 and composition summary. `traps.csv`
and `countries.csv` describe the data. `trial_membership.csv` gives source row
positions, traps, outer test folds and inner validation folds (-1 means outer
test, excluded from inner CV). `*_folds.csv`, `*_countries.csv`,
`balanced_group_country_traps.csv`, `balanced_group_months.csv`, and
`seed_comparison.csv` retain the diagnostic measurements.
