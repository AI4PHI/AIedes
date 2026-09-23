# Implementation checks — 2026-09-22

- 14 unit tests passed: fold integrity, all three losses, fixed epochs, learning
  rates, model reloads, deterministic fits, training-only normalization, one-lag
  independence from the second lag, recursive startup, metrics and artifact checks.
- Complete synthetic workflow passed through training, selection, replication,
  one-lag and checkpoint comparisons, deployment, evaluation, tables and figures.
  Check output: `/tmp/aiedes-revision-smoke-final/` (synthetic; not paper results).
- A second invocation reused completed fits; model checkpoint modification times
  were unchanged. No country-specific performance output was generated.
- Real-data preparation passed in `/tmp/aiedes-revision-data-check/`: 5,129 AIMSurv
  rows, 387 traps, five frozen folds; 5,036 rows eligible for recursive evaluation.
  The external input has 280 rows and 40 traps, dated May–November 2021.
- Shell syntax, production dry-run, status command and whitespace checks passed.
  The launcher detects 79 available worker cores after reserving one physical core.

The full 3,000-epoch scientific experiment has not been launched. The checks validate
execution and data contracts; scientific performance will come from that run.
