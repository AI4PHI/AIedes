# Counter revision: figures, tables, and text plan

## Frozen references

The restored manuscript files were copied before revision:

- `main_old.tex`: exact copy of `main.tex` at the start of this revision pass.
- `SI_old.tex`: exact copy of `SI.tex` at the start of this revision pass.

The reference copies must remain unchanged. Manuscript edits will be made only in
`main.tex` and `SI.tex` after the figure/table structure is approved.

## Revision principle

The main paper should contain the results needed to understand and evaluate the
Counter, while the complete sensitivity analyses should be placed in the SI. The
number of main-text figures and tables can remain unchanged by replacing the current
top-five configuration table with the forecasting-mode comparison and moving the
updated top-five table to the SI.

All manuscript wording will describe the final analysis directly. Revision history
and comparisons with the former protocol belong only in the response to reviewers.

## Recommended main-text structure

### Table 1: input variables

Update the existing input-variable table rather than adding a table. It should show:

- daily climate histories of 30, 60, and 90 days;
- corresponding weekly aggregations over the same intervals;
- one- and two-week previous egg-laying values;
- availability indicators for missing previous observations;
- the climate-only input configuration.

### Figure 3: principal Counter sensitivity analysis

Retain the existing figure position and role, but regenerate it from the complete
trap-grouped experiment without filtering runs. The main figure should remain focused
on the 90-day configurations and show how performance changes with climate inputs,
mosquito-history inputs, network structure, and dropout. Complete loss and time-window
comparisons should be moved to the SI.

### Table 2: prediction and forecasting modes

Replace the current top-five-configuration table with the comparison that is central
to the revised scientific interpretation. All four rows use the matched cohort of
5,036 observations from 375 traps.

| Mode | Previous mosquito values | R² | log-R² | RMSE | Accuracy |
|---|---|---:|---:|---:|---:|
| Observed-history prediction | Measured previous values | 0.565 | 0.678 | 26.36 | 0.921 |
| Observed-history model used recursively | Its own predictions during evaluation | 0.456 | 0.649 | 29.48 | 0.916 |
| Dedicated autoregressive forecast | Its own predictions during training and evaluation | 0.535 | 0.663 | 27.25 | 0.919 |
| Climate only | No mosquito-history input | 0.519 | 0.664 | 27.72 | 0.921 |

The text should also report the complete observed-history result on all 5,129
observations from 387 traps: R² = 0.566, log-R² = 0.680, RMSE = 26.12, and
accuracy = 0.922.

### Figure 5: observed versus predicted values

Replace the current training/test scatter with two held-out-trap panels:

1. observed versus predicted weekly rates on the original scale;
2. the same observations after `log(1+x)` transformation.

Both panels should report R², log-R² where appropriate, RMSE, and the sample size.
Predictions should be the average of the five seed models trained without the
corresponding trap. If prediction variability is displayed, use subtle vertical
error bars representing one standard deviation across those five models; confirm
legibility before retaining them.

### Figure 6: representative trap time series

Replace the current mixed 12-panel figure with six of the longest AIMSurv trap
series. Each panel should show observations, the mean held-out prediction, and a
light band or error bars representing one standard deviation across the five seed
models. The caption must identify these as observed-history predictions at traps
excluded from training.

VectorNet time series should remain outside the main paper. They may be retained as
diagnostic material in the experiment folder.

### Figure 7: European seasonal map

Replace the current map output with the dedicated autoregressive `[64,32]` ensemble.
The caption should state that:

- 25 models were trained (five trap folds by five random seeds);
- generated 7- and 14-day values were used during both training and projection;
- the recursive sequence was initialized at zero before the first prediction;
- mapped values are ensemble means.

## Recommended Supplementary Information structure

### SI Figure: complete 90-day ablation

Regenerate the current loss/dropout/input figures using all completed runs. Show raw
R² and zero/non-zero accuracy for WMSE, WMSLE, and ZINB across input combinations,
dropout rates, and architectures. No run filtering or favourable-run selection should
be used.

### SI Figure: climate-window sensitivity

Add the completed 30-, 60-, and 90-day comparison. It should distinguish:

- daily and weekly climate inputs;
- climate-only and two-lag configurations;
- temperature plus precipitation and temperature plus dew point plus precipitation.

The main result is that performance generally increases from 30 to 60 to 90 days and
daily inputs generally outperform weekly aggregation. The analysis makes no claim
about windows longer than 90 days.

### SI Table: five leading WMSLE configurations

Move the current main-paper top-five table to the SI and replace its values with the
trap-grouped, seed-replicated results. Report mean ± standard error across the five
held-out trap folds after averaging the five seeds within each fold. Include R²,
log-R², RMSE, and accuracy.

### SI Table: best configuration for each loss

Update the existing SI table with the completed results for WMSE, WMSLE, and ZINB.
Include the input set, climate window, daily/weekly aggregation, architecture,
dropout, R², log-R², RMSE, and accuracy. State that weekly rates can be fractional,
so ZINB is a count-inspired comparison rather than an exact discrete likelihood.

### SI Table: previous-value sensitivity

Add the controlled one-versus-two-lag comparison for the selected WMSLE climate
configuration:

- one previous weekly value: R² = 0.529 ± 0.033, log-R² = 0.673 ± 0.016,
  accuracy = 0.922 ± 0.004;
- two previous weekly values: R² = 0.553 ± 0.023, log-R² = 0.670 ± 0.013,
  accuracy = 0.923 ± 0.004.

The climate-only model remains a separate reference and should not be described as
part of this lag-order comparison.

### SI Table: autoregressive architecture comparison

Report the limited architecture check used for the slower autoregressive training:

- `[64]`: R² = 0.521 and log-R² = 0.660;
- `[64,32]`: R² = 0.535 and log-R² = 0.663.

State that this is a focused comparison, not a complete repetition of the main
architecture search.

### SI Table: training variability

Provide a compact summary of seed-to-seed variability for the reported
configurations. This supports retention of all runs without expanding the SI to a
table containing thousands of fits. The complete machine-readable results table
(`all_runs.csv`) should be included in the released analysis outputs.

### SI Figure and table: VectorNet external evaluation

Add one four-panel scatter figure:

1. original scale before September;
2. log scale before September;
3. original scale from September onward;
4. log scale from September onward.

Use the observed-history 25-model ensemble. Subtle error bars may show one standard
deviation across the 25 predictions when legible.

Report the following values in an accompanying compact table or in the figure caption:

| Period | n | R² | log-R² | RMSE |
|---|---:|---:|---:|---:|
| All observations | 280 | -0.004 | 0.170 | 69.82 |
| Before September | 162 | 0.072 | 0.360 | 59.98 |
| September onward | 118 | -0.071 | -0.076 | 81.41 |

The overall zero/non-zero accuracy is 0.754. The seasonal division is a post hoc
diagnostic motivated by the visually missed autumn peak and should be labelled as
such.

### SI Bland--Altman analysis

Retain the existing analysis but regenerate it from pooled held-out-trap predictions.
The current result has a mean prediction--observation difference of -2.11 eggs per
week and descriptive limits of -53.13 to 48.91.

## Analyses to retain outside the paper

The following diagnostics should remain in the experiment folders unless a reviewer
specifically requests them:

- VectorNet geographic and altitude maps;
- six-trap VectorNet time-series diagnostics;
- the hybrid VectorNet fallback experiment;
- the dedicated-autoregressive VectorNet diagnostic;
- early-stopping pilots;
- detailed startup and lag-availability diagnostics;
- country-by-country performance estimates;
- every individual row of the 516-configuration experiment in the PDF SI.

These outputs remain available for reproducibility, while the paper reports the
analyses needed to support its claims.

## Order of implementation after approval

1. Finalize and regenerate the main and SI figures and tables listed above.
2. Update the Counter Methods to match the final data, validation, preprocessing,
   architecture, lag, and autoregressive procedures.
3. Update the Counter Results using only values represented in the approved figures
   and tables.
4. Update the Discussion and Conclusion with the observed-history/forecasting
   distinction, geographic limits, and VectorNet interpretation.
5. Revisit each editor and reviewer response and add a concise summary identifying
   the final manuscript changes associated with that comment.

