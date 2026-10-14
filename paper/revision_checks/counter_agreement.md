# Counter legacy-rerun agreement check

Checked 2026-09-17 against the local manuscript, SI, historical notebook outputs,
and archived model. No training was launched and no existing results or manuscript
figures were overwritten.

## Finding

The rerun closely reproduces the headline WMSLE performance under the original
top-25% reporting rule. The selected WMSLE configuration is still ranked first.
It does not reproduce every numerical entry or configuration ranking exactly.
This establishes agreement of the legacy performance level, not completion of
the stronger validation requested by the reviewers.

## Method and completeness

- Loaded all 6,480 result pickles: 2,160 each for WMSE, WMSLE, and ZINB.
- Verified completion markers, one result row per file, 324 distinct
  configurations, exactly seeds 1–20 per configuration, no duplicate
  configuration/seed records, and 3,000 epochs per run.
- Verified that the five audited scalar metrics are finite and match their last
  recorded history values.
- Reproduced the executed selection rule in `create_complete_df.py`: within each
  loss/input/dropout/layer/width configuration, retain the five highest final
  `test_r2` values out of 20 runs. The plotting notebooks summarize the final
  metrics, not the last-ten-history averages. Several old script comments are
  inconsistent with this executed rule.
- Computed means and sample standard errors (`pandas.sem`, ddof=1), matching the
  historical notebook procedure. Top-selected-run SEs are descriptive; they do
  not account for selection bias or overlap between random splits.

## Headline configuration

WMSLE, daily temperature/dew point/precipitation plus lagged egg counts, dropout
0.2, CLI `num_layers=2`, `hidden_dim=64`; original rule retains five runs.

| Metric | Main manuscript mean ± SE | Rerun mean ± SE |
|---|---:|---:|
| Test R² | 0.627 ± 0.006 | 0.626691 ± 0.010504 |
| Test loss | 0.59 ± 0.02 | 0.585856 ± 0.033272 |
| Presence/absence accuracy | 0.920 ± 0.002 | 0.918519 ± 0.000904 |
| Test RMSE | 22.6 ± 1.3 | 23.331348 ± 1.787330 |

The original single-loss notebook instead records R² 0.629114 ± 0.005494,
accuracy 0.918519, and RMSE 22.551855 for this configuration. These agree with
the rounded SI row but not exactly with the main-text table. Thus there is an
existing inconsistency among the historical reference summaries.

## Other reported configurations

Comparisons below hold each reported configuration fixed, rather than substituting
the rerun winner. Ranks are within each loss across all 108 configurations.

| Reference | Original R² ± SE | Rerun R² ± SE | Rerun rank |
|---|---:|---:|---:|
| Main WMSLE row 1 | 0.627 ± 0.006 | 0.626691 ± 0.010504 | 1 |
| Main WMSLE row 2 | 0.626 ± 0.014 | 0.618086 ± 0.017520 | 3 |
| Main WMSLE row 3 | 0.619 ± 0.005 | 0.616754 ± 0.009634 | 4 |
| Main WMSLE row 4 | 0.617 ± 0.008 | 0.613083 ± 0.005115 | 7 |
| Main WMSLE row 5 | 0.616 ± 0.010 | 0.610300 ± 0.004908 | 9 |
| SI WMSE winner | 0.643 ± 0.005 | 0.630941 ± 0.006603 | 5 |
| SI WMSLE winner | 0.629 ± 0.005 | 0.626691 ± 0.010504 | 1 |
| SI ZINB winner | 0.621 ± 0.014 | 0.619001 ± 0.011103 | 3 |

Selecting the best configuration anew within each loss gives WMSE 0.637240,
WMSLE 0.626691, and ZINB 0.620176. The ordering of the best R² by loss is retained,
but the WMSE and ZINB winning configurations change. The differences are
descriptively small; no equivalence or significance test was performed.

## Direct archived-model check

Evaluated the archived seed-14 WMSLE model used by `R2_scatter_plot.ipynb` and the
new seed-14 model on identical current data and the notebook's stratified 80/20
split (`random_state=14`, bins `[1]`).

| Model | Train R² | Test R² | Test RMSE | Test accuracy |
|---|---:|---:|---:|---:|
| Archived figure model | 0.697841 | 0.631491 | 19.360212 | 0.919103 |
| Rerun, same seed/configuration | 0.692256 | 0.621935 | 19.609627 | 0.918129 |

The archived test R² reproduces the historical notebook's 0.631491. The paper's
saved scatter-plot PDF displays train R² 0.70 and test R² 0.63, also matching this
evaluation. The prose's train R² 0.68 is inconsistent with that figure.

The original sweep did not explicitly seed all training randomness; the rerun
adds Python/NumPy/PyTorch seeding. A matching split seed therefore does not imply
identical fitted weights. This is a plausible reason for run differences, not
a demonstrated attribution of every difference. Original per-run result files
for the full historical sweep are unavailable locally, so a full paired
comparison against the original runs is not possible.

## All-run reporting changes the headline

For the same selected WMSLE configuration, retaining all 20 rerun seeds gives:

- Test R² mean **0.558612**, SE **0.011667**, SD **0.052179**.
- Median **0.556194**, range **0.437759–0.654429**.
- Mean RMSE **26.986275**, SE **0.852175**.
- Mean accuracy **0.922710**, SE **0.001138**.

Top-25% selection raises the R² mean by **0.068079**. The same configuration is
still the best WMSLE configuration by all-run mean, but its revised headline
must reflect all runs. These are still legacy random-split results, not blocked
validation or recursive forecast results.

## Reporting discrepancies to resolve

1. Main text gives 20 runs and top 25%, but the table caption says five of ten.
   This rerun has 20 runs per configuration and selects five for this audit.
2. Main table and SI/notebook numbers differ slightly, as described above.
3. The manuscript describes two hidden layers of 64 units. Both the archived
   and rerun selected models actually have **one hidden layer of 64 units plus
   the output layer**. `run.py` passes `num_layers - 1` to the model constructor.
   Higher-depth models also halve widths between hidden layers; the manuscript
   architecture terminology needs reconciliation with the implementation.
4. The current weekly dataset contains **5,129 rows**, with **4,103 training**
   and **1,026 test** records, rather than the manuscript's 5,432 and 4,345.
   Reproducing the archived model's test score supports consistency with its
   evaluation data. The follow-up below explains 295 of the 303-record
   discrepancy; eight source records remain unexplained.
5. The archived figure's train R² is 0.70, not the prose's 0.68.

## Follow-up: dataset-size discrepancy

Tracing the current source pickle on 2026-09-17 gives:

| Stage | Weekly egg records (6–8 days) | Unique traps |
|---|---:|---:|
| Manuscript claim | 5,432 | 395 |
| Current source, before merging | 5,424 | 387 |
| After merging by trap/start date/end date | 5,129 | 387 |
| Saved training dataset | 5,129 | 387 |

The `keep` and `climate_nan` filters remove no weekly egg records in this source.
The merge consolidates 414 rows in 119 repeated trap/date groups into 119 rows,
reducing the count by 295. There are 5,010 singleton groups; repeated-group sizes
are 49 pairs, seven triples, 20 groups of four, and 43 groups of five. The row
reductions occur in Austria (147), Spain (143), and Cyprus (5).

Reconstructing the grouping and mean weekly rates directly from the source gives
exactly the saved 5,129 trap/date keys, with no unmatched keys and zero difference
in target values. Thus the reduction happens during dataset preparation, not
training or train/test splitting.

The likely explanation is that the manuscript retained a count from before
aggregation, possibly from a slightly different source version. Only the
aggregation portion is established: the remaining eight records and eight-trap
difference cannot be attributed to a specific historical change with the local
evidence. Equal differences do not prove one missing record per missing trap.

These are repeated trap/date records, not necessarily identical duplicate rows:
29 groups contain different egg rates, and 14 contain both `Culicidae` and
`Aedes albopictus (Skuse, 1894)` taxonomic labels. `merge_dataframe` averages
numeric values and keeps the first text value. Whether that is the intended
treatment of these distinct occurrence/event records should be checked against
their observation semantics before the revised production run.

## Reproduction and outputs

From the repository root, using the installed project environment:

```bash
/home/biazzin/conda-envs/aiedes-env/bin/python experiments/counter/check_agreement.py
/home/biazzin/conda-envs/aiedes-env/bin/python experiments/counter/check_reference_model.py
```

- `counter_runs.csv`: scalar metrics, last-ten-history means, selection flag,
  and original result path for every rerun.
- `counter_top25_summary.csv`: original-rule summaries for all configurations.
- `counter_all_runs_summary.csv`: unfiltered summaries for all configurations.
- `counter_manuscript_comparison.csv`: reference means/SEs and rerun differences
  for the five main-table rows and three SI rows, across available metrics.
- `counter_reference_model.json`: archived and rerun model architectures,
  data counts, and independently evaluated metrics.
