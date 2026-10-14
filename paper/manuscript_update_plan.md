# Manuscript revision plan

## 1. Central revision strategy

Keep the main scientific message centred on the multicountry AIMSurv dataset. The VectorNet analysis is a secondary external stress test, not the central validation result.

The revision should make five methodological changes explicit:

1. Replace the previous random split/model-selection presentation with trap-disjoint cross-validation: five folds and five training seeds per fold, retaining all 25 models.
2. Report out-of-fold (OOF) predictions and distinguish observed-history estimation from autonomous forecasting.
3. Extend the existing sensitivity analysis by adding 30-, 60-, and 90-day climate histories and a controlled one-versus-two-lag comparison; use the existing climate-only models as a separate no-history reference. Retain and report more clearly the existing daily/weekly, loss-function, architecture, and dropout comparisons.
4. Introduce dedicated autoregressive training, in which lagged mosquito values are generated recursively by the model during training and forecasting.
5. Report raw-scale and log-scale performance, including both R² and log-R².

No additional main-text figure is required. Reuse or replace existing tables and figures, and put the complete sensitivity analyses and VectorNet scatter plot in the Supplementary Information (SI).

## 2. Concise description of what is new

Suggested manuscript text:

> For the revision, we rebuilt the evaluation pipeline using five trap-disjoint folds and five independent training seeds per fold, yielding an ensemble of 25 retained models. All preprocessing and normalization parameters were estimated within each training fold. We extended the sensitivity analysis by adding 30-, 60-, and 90-day climate histories and a controlled comparison of one versus two previous weekly mosquito values; climate-only models provide a separate no-history reference. The existing comparisons of daily and weekly inputs, three loss functions, network widths, and dropout rates were retained and are reported more completely. We also separated observed-history prediction from autonomous forecasting and trained a dedicated autoregressive model whose lagged inputs are recursively generated from its own previous predictions. Performance is now reported using R², log-R², RMSE, and classification accuracy, with an additional external stress test on the VectorNet dataset.

## 3. Methods changes

### 3.1 Data and prediction target

- State that the target is weekly ovitrap-derived egg-laying rate, used as a proxy for *Aedes* activity; it is not a direct measure of adult abundance or disease risk.
- Correct the AIMSurv description to 5,129 observations from 387 traps in 13 countries; 4,100 observations are zero (79.94%).
- Acknowledge the geographic imbalance: Austria accounts for 54.6% of all observations, while Spain, Albania, and Italy contain 97.3% of positive observations.
- Therefore describe the validation as generalization to unseen traps within the sampled multicountry mixture, not as a leave-country-out transfer test.

For VectorNet:

- Report 280 weekly observations collected from May to November 2021.
- Report the external dataset as containing 40 traps. Multiple traps may occur at the same named location, and traps at the same location may have distinct coordinates.
- State that available metadata did not permit confirmation that all collection and processing procedures were equivalent to AIMSurv. Do not claim protocol non-adherence.

### 3.2 Inputs and preprocessing

- Explain the three climate variables and the tested 30-, 60-, and 90-day histories.
- State that the archived harmonized inputs contain 90 climate days per observation. Thus, 30- and 60-day windows are exact subsets of the same records; a 120-day analysis was not performed and no conclusion is made for windows longer than 90 days.
- Daily and weekly variants cover the same exact 30-, 60-, and 90-day windows. Weekly aggregation retains an oldest partial week when necessary; avoid describing them simply as 4, 9, and 13 weeks.
- State that normalization is fitted only on each training fold and then applied to its held-out fold and external data.
- Describe lagged mosquito observations and their availability indicators for the observed-history model. Missing lag values are not filled using future observations.

### 3.3 Validation and ensemble construction

- Five folds are defined by trap identity, so observations from a trap do not occur in both training and validation within a fold.
- Each fold is trained with five random seeds. All 25 fold/seed models are retained.
- Every AIMSurv OOF prediction is the mean of the five seeds trained without that observation’s trap; it is not an average over all 25 models.
- External VectorNet predictions and the European projection use all 25 models.
- Clarify that seed-to-seed standard deviation represents ensemble/model variability, not a confidence interval.

### 3.4 Model architecture

Correct the current architecture description; the selected observed-history model does **not** have two 64-unit hidden layers.

Observed-history model:

- Input dimension: 274 = 270 climate values (3 variables × 90 daily values), two lagged mosquito values, and two lag-availability indicators.
- One hidden block: Linear(274, 64), batch normalization, ReLU, and dropout 0.2.
- Softplus output; 17,793 trainable parameters.

Dedicated autoregressive model:

- Input dimension: 272 = 270 climate values and two recursively generated lag values; no lag-availability indicators.
- Two hidden blocks of 64 and 32 units, each with batch normalization, ReLU, and dropout 0.2.
- Softplus output; 19,777 trainable parameters.

The standard model search evaluated `[32]`, `[64]`, `[32,16]`, and `[64,32]` architectures within 516 configurations. Autoregressive training compared only `[64]` and `[64,32]`, because recursive feature regeneration at every epoch is substantially more expensive. Present this as a targeted architecture sensitivity analysis, not an equally exhaustive search.

### 3.5 Forecasting modes

Define these modes consistently:

1. **Observed-history/hindcast:** uses actual preceding mosquito observations where available. This is the principal OOF estimation result.
2. **Legacy recursive diagnostic:** the observed-history model is forced to consume its own predictions. This tests deployment mismatch but is not the final forecasting model.
3. **Dedicated autoregressive forecast:** trained using recursively generated 7- and 14-day lag values and initialized from zero before the first observation. It never receives observed mosquito lags.
4. **Climate-only ablation:** excludes mosquito-history inputs.

Do not initialize autonomous predictions with invented values two weeks before the series begins. Zero initialization is reproducible and reflects the absence of prior observations; discuss the resulting early-season limitation.

### 3.6 Metrics

Report R², log-R², RMSE, and binary classification accuracy. Define:

\[
R^2_{\log}=1-\frac{\sum_i[\log(1+y_i)-\log(1+\hat y_i)]^2}
{\sum_i[\log(1+y_i)-\overline{\log(1+y)}]^2}.
\]

Explain that log-R² complements raw R² by reducing the dominance of the largest counts in this highly skewed target and aligns with the motivation for WMSLE. Do not describe it as universally superior.

## 4. Results changes

### 4.1 Classifier

Report the NUTS3-blocked evaluation explicitly:

- Accuracy: 0.973
- Specificity: 0.983
- Sensitivity: 0.954
- Critical success index: 0.926

### 4.2 Main AIMSurv OOF result

For the complete OOF set of 5,129 observations, report the observed-history model as:

- R² = 0.566
- log-R² = 0.680
- RMSE = 26.12
- Accuracy = 0.922

Describe the sensitivity findings concisely: 90-day windows generally performed better than 30- and 60-day windows; daily inputs generally outperformed weekly aggregation. Keep the full numerical comparisons in the SI.

### 4.3 Forecasting-mode comparison

Use the common matched cohort of 5,036 observations from 375 traps for a fair comparison:

| Mode | Lag source | R² | log-R² | RMSE | Accuracy |
|---|---|---:|---:|---:|---:|
| Observed history | Measured preceding values | 0.565 | 0.678 | 26.36 | 0.921 |
| Legacy recursive diagnostic | Predictions from a model trained with observed lags | 0.456 | 0.649 | 29.48 | 0.916 |
| Dedicated autoregressive `[64,32]` | Recursively generated during training and inference | 0.535 | 0.663 | 27.25 | 0.919 |
| Climate only | None | 0.519 | 0.664 | 27.72 | 0.921 |

The poor R² = 0.456 result belongs to the **legacy observed-lag model used recursively**, not to the final autoregressive model. The dedicated autoregressive training recovers much of the performance lost through deployment mismatch and is therefore the appropriate model for autonomous forecasting and the European projection.

Mention that the alternative one-layer autoregressive architecture produced R² = 0.521 and log-R² = 0.660 on the same cohort, supporting selection of `[64,32]` without claiming a comprehensive autoregressive architecture search.

### 4.4 External VectorNet stress test

Report the overall result briefly in the main text:

- n = 280
- R² = −0.004
- log-R² = 0.170
- RMSE = 69.82
- Accuracy = 0.754

The post hoc seasonal split can be reported in the SI and summarized in the text:

| Period | n | R² | log-R² | RMSE |
|---|---:|---:|---:|---:|
| Before September | 162 | 0.072 | 0.360 | 59.98 |
| September onward | 118 | −0.071 | −0.076 | 81.41 |

Interpretation must remain cautious. The larger later-season error, particularly around the second autumn peak, is consistent with unresolved local variation. Plausible factors include mountain/valley microclimates that are poorly represented by the climate grid and differences in sampling or processing. These are hypotheses, not demonstrated causes.

Optional descriptive context for the SI: SRTM-derived elevations at the coordinates range from approximately 89 to 1,201 m (mean 602 m; median 673 m). Label these as remotely estimated elevations, not field measurements.

## 5. Tables and figures

### Main text

- **Table 1:** extend the input description to include 30-, 60-, and 90-day climate histories for air temperature, dew point, and total precipitation, in daily and weekly aggregated forms. State that 90 days was selected.
- **Model-performance table:** replace the current top-five-configuration table with the compact four-mode comparison above. Move the former top-five table to the SI; this avoids increasing the number of main tables.
- **Figure 3:** retain a readable 90-day summary in the main text. Move complete 30/60/90-day, daily/weekly, and three-loss comparisons to the SI.
- **Observed-versus-predicted figure:** replace the current version with two OOF panels: (A) raw observed versus predicted, annotated with R² and RMSE; (B) log1p observed versus log1p predicted, annotated with log-R². State that each point is the mean of five seed models from its held-out fold.
- **Classifier maps:** retain them but improve readability and ensure captions identify the spatially blocked evaluation.
- **Time-series example:** retain it if legible; specify whether it shows observed-history or autoregressive predictions.
- **European projection:** use the dedicated `[64,32]` autoregressive ensemble of 25 models and state the zero-initialization and ensemble procedure in the caption.

Do not add error bars to all 5,129 AIMSurv scatter points; they would obscure the figure. If uncertainty must be shown, add a compact SI summary of seed variability or binned uncertainty.

### Supplementary Information

- Complete window, aggregation, lag, loss, architecture, and dropout sensitivity plots.
- Former top-five model-configuration table.
- VectorNet four-panel scatter plot: before September and September onward, each on raw and log1p scales, with the corresponding R² values.
- For VectorNet, prediction bars may show ±1 standard deviation across all 25 model predictions because there are only 280 points. Call this ensemble spread, not a confidence interval.
- Do not include the street map or elevation map unless a reviewer explicitly requests geographic detail.

## 6. Discussion changes

- Lead with the robust AIMSurv OOF result and trap-disjoint evaluation.
- Explain that observed-history prediction performs best when recent surveillance measurements are available.
- Emphasize that applying this model recursively creates a training/deployment mismatch; dedicated autoregressive training reduces that gap and is selected for forecasting.
- Discuss the value of both raw and log-scale metrics for a zero-heavy, right-skewed target.
- Present VectorNet as a deliberately difficult, geographically compact external stress test. Its weak result does not replace the AIMSurv findings, but it limits claims of direct portability.
- Discuss the deterioration after August as an exploratory observation, not proof of a mechanism.
- Mention unresolved mountain/valley microclimate, coarse gridded climate data, and uncertain protocol comparability only as possible explanations.
- Acknowledge country imbalance, uneven positive counts, limited external sample size, target-proxy limitations, and the absence of leave-country-out validation.
- Frame operational use as decision support that requires local calibration and continued surveillance, not as a substitute for surveillance.

## 7. Abstract and conclusion

### Abstract

- Replace any performance obtained from random splitting or a single representative model with the five-fold, five-seed OOF values.
- State that validation was trap-disjoint.
- Distinguish observed-history estimation from autonomous autoregressive forecasting.
- Avoid broad claims of geographic transfer based on VectorNet.

### Conclusion

- Conclude that climate plus recent surveillance history can support prediction at unseen traps within the sampled surveillance network.
- State that dedicated autoregressive training is preferable when observed lags will not be available at deployment.
- Note that the external stress test demonstrates the need for local calibration, finer-scale environmental covariates, and harmonized surveillance protocols.
- Avoid claims of direct disease-risk prediction, continent-wide validation, or replacement of monitoring programmes.

## 8. Reviewer-response structure

For each response, use: **change made → location in manuscript → result → interpretation/limitation**.

Key responses:

- **Validation leakage/reproducibility:** implemented trap-disjoint five-fold validation, five seeds per fold, fold-specific preprocessing, and retained all 25 models.
- **Input-history sensitivity:** added 30-, 60-, and 90-day daily and weekly comparisons; retained 90-day daily input based on performance.
- **Forecasting realism:** separated observed-history, legacy recursion, climate-only, and dedicated autoregressive modes; selected the dedicated autoregressive model for autonomous projections.
- **Additional metrics:** added log-R² alongside R², RMSE, and accuracy.
- **Architecture clarity:** corrected the observed-history network to one 64-unit hidden layer and documented the separate `[64,32]` autoregressive architecture.
- **External validation:** added VectorNet R² and log-R² and an SI scatter plot; explicitly limited portability claims and discussed plausible scale/protocol limitations.
- **Uncertainty:** predictions are ensemble means; reported spread is variability across trained models, not inferential confidence.

## 9. Final consistency checks

- Use one term for each mode throughout: observed history, legacy recursive diagnostic, dedicated autoregressive forecast, and climate only.
- Do not associate R² = 0.456 with the final autoregressive model.
- Keep the 5,129-observation full OOF metrics separate from the 5,036-observation matched comparison.
- Distinguish five models contributing to each AIMSurv OOF point from the complete 25-model ensemble used externally.
- Verify that text, table, figure, caption, and code agree on input dimensions, hidden layers, lag availability flags, initialization, and selected architecture.
- Label the VectorNet seasonal split as post hoc and the altitude values as SRTM estimates.
- Ensure all reported values use consistent rounding and that every reviewer response points to a concrete manuscript or SI location.
