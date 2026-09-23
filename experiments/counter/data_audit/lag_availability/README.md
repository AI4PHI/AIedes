# Missing previous egg-rate inputs

Audit of the current 5,129-row, 387-trap weekly AIMSurv dataset. The audit reads the saved raw `prev1_rates` and `prev2_rates`, checks their availability flags, and reconstructs them against the full merged observation dataset using the existing effort-dependent ±3-day lookup. All reconstructed lag values and flags agree with the cached data. It does not alter any data or model.

| Saved observed-history inputs | Rows | Percentage |
|---|---:|---:|
| Both available | 3,853 | 75.12% |
| Only p1 missing | 185 | 3.61% |
| Only p2 missing | 596 | 11.62% |
| Both missing | 495 | 9.65% |
| At least one missing | 1,276 | 24.88% |

Including overlaps, p1 is missing in 680 rows (13.26%) and p2 in 1,091 (21.27%). Missing is encoded separately from an observed zero; these counts do not count zero egg rates as missing.

## Trap distribution and series position

- All 387 traps have at least one row with a missing lag.
- 144 traps have at least one missing lag in at least half their rows.
- 86 traps never have both lags available together; 35 have neither lag available in any row.
- Of 241 traps with at least ten observations, 32 have at least one missing lag in at least half their rows. The issue is therefore not restricted to very short series.
- Beyond the first two records of each weekly series, 575 of 4,408 rows still have at least one missing lag; 136 have both missing. This is a series-position summary, not a precise measure of all gaps in underlying surveillance history.
- Using the full merged observation history, 326 missing p1 windows and 612 missing p2 windows precede the first possible source record. The remaining 354 p1 and 479 p2 missing windows lack a match despite history extending to the lookup window. Sampling gaps, cadence and the lookup tolerance all contribute; these are not necessarily physically lost samples.
- The four previously plotted 33-observation traps (360, 361, 363, 365) each have p1 missing once and p2 missing twice. Only two rows per trap (6.06%) lack any lag. These examples underrepresent the missing-history problem.

## Observed history and the current recursive implementation differ

The cached observed lags were created before selecting the 6–8-day interval subset. Some available observed lag values come from records outside that weekly subset: 156 p1 values and 118 p2 values have no matching source in the weekly subset.

The current `counter_v2/run.py::recursive_test_predictions` generates predictions only at the test observation dates and requires an earlier generated prediction within the date window. With whole-trap held-out evaluation on this weekly dataset, its date lookup has no predecessor for 836 p1 inputs (16.30%) and 1,209 p2 inputs (23.57%). These are structural availability counts from the lookup, not a newly executed recursive model evaluation.

Thus the current observation-date recursion does not eliminate missing history. A continuous simulator with complete climate inputs can generate intervening predictions and eventually supply both lags after initialization. It must be evaluated separately using the same dates, initialization and update rules intended for maps.

## Interpretation and proposed evaluation

Replacing missing observations with a continuous prediction history can help, but the replacement values carry model error. Removing lag inputs can also help if feedback is unreliable. Missingness counts alone cannot establish which approach predicts better. Errors in autoregressive predictions can propagate when predicted history replaces observed history; see [Bengio et al., 2015](https://proceedings.nips.cc/paper_files/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) for the general training/inference distinction, not as empirical evidence about this dataset.

Keep the proposed observed-history, autonomous, and climate-only comparison. For each, additionally report metrics on the same observation subsets defined by the original observed-history availability: both available, one missing, and both missing. Show sample counts and the fraction with eggs in each subset. These strata differ in outcome distribution and country composition; differences between strata do not establish that missingness caused a performance change. All-zero strata have undefined R².

Use validation trajectories to inspect these behaviors before freezing the final testing procedure. No training or selection rule was changed by this audit.

Files: `summary.json`, `availability_summary.csv`, `traps.csv`, `countries.csv`, and `rows.csv`. Reproduce the audit with `src/AIedes/evaluation/counter_audit.py` using the project Python environment.
