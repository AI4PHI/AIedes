# AIedes Major Revision Plan

Decision: major revision at Ecological Informatics.

Deadline: 2026-09-12.

## Status checked 2026-09-28

- The legacy Counter sweep completed on 2026-08-22: 6,480 runs, with all result
  files and completion markers present. The AIMSurv training data are available.
- Agreement check completed: the historical selected WMSLE configuration remains
  close to the manuscript result (top-25% mean test R2 = 0.626691 versus 0.627
  in the main manuscript), while the all-run mean R2 is 0.558612. This legacy
  comparison is retained for auditability only; revised reporting will use all
  runs and will not describe the top 25% as the final performance estimate.
- Full evidence and reproducible audit outputs:
  [Counter agreement check](revision_checks/counter_agreement.md).
- Counter v2 production runs are complete. The revised evidence now includes
  all-run summaries, trap-blocked validation, the observed-lag versus recursive
  comparison, window/lag/loss ablations, and the VectorNet external check.
- The climate-only European rerun is complete: five full-data models were fit
  and their predictions were averaged. The remaining Counter work is the
  autoregressive-training extension intended to improve the European map,
  followed by manuscript and response-letter updates.
- No external-country performance analysis or new nested cross-validation is
  planned. The sampling protocol is internationally harmonised in AIMSurv;
  the available sample size does not support a comprehensive country-by-country
  validation analysis, and the paper will state this scope explicitly.

## Classifier

### Text Changes

1. Report the NUTS3 blocked-validation numbers in the main text, not only in the SI. [Reviewer 1]
   - Add a compact table comparing random grid-cell split vs NUTS3 split.
   - Present random split as interpolation performance and NUTS3 split as the stricter spatial-generalization check.

2. Explain the climate-based masking in the main Methods. [Reviewer 1]
   - Include the actual thresholds:
     - annual precipitation >= 200 mm/year;
     - coldest-month mean temperature >= -3 C;
     - annual mean temperature >= 10 C.
   - Clarify that the mask reduces administrative downscaling artefacts.
   - Acknowledge that the classifier is trained on climate-filtered surveillance labels, so high performance must be interpreted cautiously.
   - State that rerunning without the mask is not an appropriate sensitivity analysis, because the unmasked labels include ecologically implausible grid cells created by coarse administrative downscaling.
   - Add how many originally positive grid cells were relabelled by the mask:
     - original positive cells before filtering: 20,832;
     - retained positive cells after filtering: 16,164;
     - relabelled positive cells: 4,668, i.e. 22.4% of originally positive cells;
     - reason: 4,662 temperature-unsuitable cells and 6 precipitation-unsuitable cells.
   - Possible response: we agree that the masking procedure needed clearer reporting, but we do not remove it because it is an essential correction for coarse administrative labels projected onto a fine climate grid. Without the mask, training would include positive labels in climatically implausible cells, degrading the response variable rather than providing a valid sensitivity analysis. The approach follows the same ecological-suitability rationale used in the source modelling work that motivated the classifier setup.

3. Soften performance claims. [Reviewer 1]
   - Avoid overconfident phrasing such as "excellent" without qualification.
   - State that performance is high under the tested validation schemes, but affected by spatial structure, surveillance-derived labels, and climate-based masking.

4. Discuss excluded label categories and absence uncertainty. [Reviewer 1]
   - Explain why "introduced", "unknown", and "no data" categories were excluded.
   - Note that absence certainty varies with surveillance intensity.
   - Treat predictions in unknown/no-data areas as prioritization or suitability outputs, not confirmed presence/absence.

5. Fix minor reporting issues. [Reviewer 1]
   - Change "three standard evaluation metrics" to "four".
   - Check classifier figure captions and legends for clarity.

### New Simulations / Training

1. Do not add a new Random Forest/XGBoost baseline.
   - The comparison remains anchored to the previous modelling approach in the
     literature that motivated AIedes.
   - The contribution is a released dataset, protocol, codebase, and benchmark
     that allows other groups to test their preferred models on identical data
     and compare results reproducibly.
   - State explicitly that AIedes is not claimed to be the best model overall;
     it is a reproducible reference framework intended to facilitate future
     model comparisons.

## Counter

### Revised deployment decision

The paper will distinguish three valid use cases rather than call one model
universally best:

- **Observed-lag model:** the selected model with previous observed egg-rate
  inputs, used for hindcasting or local prediction when those observations are
  available. This remains the strongest AIMSurv configuration.
- **Climate-only model:** a non-recursive comparator and fallback European
  predictor, with no previous egg-rate inputs.
- **Autoregressive model:** the intended European mapping model. A new training
  procedure is being tested in which previous lag inputs are generated from the
  model's own predictions, starting from the declared seasonal initialization.
  It will be used for the map only if it improves recursive performance under
  the predeclared criterion below.

The existing climate-only European output is an ensemble of five independently
seeded, full-data models and remains a reproducible comparator. If the
autoregressive-training extension passes its prespecified evaluation rule, the
main European output will instead be an ensemble of the corresponding
autoregressive models; otherwise the climate-only map will remain the
operational result.
The AIMSurv out-of-fold evaluation remains a five-seed ensemble within each of
five outer folds (25 fitted models in total), so that deployment and validation
are described consistently.

The current climate-only configuration (`450f9e361ceb`) is selected primarily
by mean blocked-fold raw R² within the climate-only candidate set. Log-R², RMSE,
and zero/nonzero accuracy will be reported as complementary criteria: the
configuration is not the winner on every metric, and the paper must not claim
that it is universally optimal.
Log-R² is especially useful because egg-rate variability is scale-dependent and
the loss is logarithmic. Accuracy will be presented as a secondary diagnostic
for distinguishing zero from nonzero observations, not as a replacement for
continuous-rate performance.

### Text Changes

1. Use more precise terminology for the model target. [Reviewer 1]
   - Replace broad "abundance" wording where needed with "weekly egg-laying rate", "ovitrap-derived egg abundance", or "ovitrap-derived weekly egg-laying rate".
   - Keep "abundance" only when explicitly framed as a proxy.
   - Explain that egg counts are influenced by trap placement, local breeding-site availability, trap attractiveness, sampling protocol, and micro-environmental conditions.

2. Separate observed-lag hindcasting from recursive forecasting. [Reviewer 1]
   - Clarify that the observed-lag model is strongest when previous observations
     are available, while the climate-only model is the non-recursive European
     comparator and the autoregressive-trained model is the candidate for the
     final Europe-wide prediction.
   - Report the original recursive experiment as a quantified diagnostic and
     explain why its underestimation/error accumulation makes the historical
     observed-lag model unsuitable as the main European predictor.
   - Treat autoregressive training as a separate attempt to close this
     deployment gap, evaluated under the same recursive protocol before it is
     used for mapping.

3. Remove or reframe the "top 25% of R2 values" model-selection language. [Reviewer 1]
   - Report all independent runs, or report mean/median and uncertainty across all runs.
   - If unstable runs exist, treat instability as part of model performance unless an objective failure criterion was defined in advance.
   - Do not introduce a new nested-CV claim; state clearly what the five outer
     folds and five seed replications represent.

4. Discuss the limited VectorNet check without overinterpreting it. [Reviewer 1, Reviewer 2]
   - Report the Lazio/VectorNet metrics separately and identify the dataset as a
     small, geographically narrow subset collected in a distinctive setting,
     including valley/high-altitude environments.
   - Check whether the poor raw R² is driven primarily by failure to reproduce
     the second seasonal peak; show a seasonal error or time-series diagnostic
     if the data support it.
   - Discuss the coarse climate-grid resolution and different local
     microclimates/sampling context as plausible explanations, without claiming
     that they establish the cause.
   - Do not add country-specific performance analyses. Explain that AIMSurv uses
     an internationally harmonised sampling protocol and that the available
     data do not support a comprehensive country-stratified analysis.

5. Clarify practical application claims. [Reviewer 2]
   - Explain realistic uses: surveillance prioritization, identifying periods/areas of higher expected oviposition activity, and supporting monitoring-site planning.
   - Avoid claiming direct dengue/chikungunya early-warning capability unless coupled transmission/importation models are actually added.
   - Possible response: disease-risk coupling is an important future extension, but the current contribution is an oviposition-rate modelling and benchmarking framework.

6. Explain interpretability scope. [Reviewer 2]
   - The main goal is reproducible benchmarking with shared data, explicit validation, open code, and quantitative metrics, not a full mechanistic interpretation of learned nonlinear effects.
   - Add a light transparency analysis if feasible, such as input ablations or permutation importance.
   - Avoid causal claims about temperature, precipitation, dew point, or lagged oviposition effects.

7. Document the final Counter architecture and parameterization.
   - State that each hidden layer is followed by batch normalization, ReLU, and dropout.
   - Explain that batch normalization has learnable scale and shift parameters and uses stored running statistics during evaluation.
   - Report the parameter count for the exact final model and reconcile the manuscript architecture description with the saved-model metadata.
   - State explicitly how climate inputs and the `p1zn`/`p2zn` lag inputs are z-score normalized using training-fold statistics, and how each lag is accompanied by an availability flag when missing.

### New Simulations / Training

1. Reproduce the Counter training sweep with explicit reproducibility controls. [Reviewer 1, Reviewer 2]
   - Use the AIMSurv 7-day training dataset expected by `experiments/counter/run_all.sh`.
   - Keep `random_state` as the train/test split and training seed.
   - Completed locally: both `eggs_y_norm_7days.pkl` and `AIMSurv_albopictus_2020_era5_land.pkl` are present, and all 6,480 legacy runs completed. See the agreement check above.

2. Add stricter blocked validation for the Counter if feasible. [Reviewer 1]
   - Use the completed leave-trap-out-style grouped validation as the selected
     stricter check.
   - Output: compact table comparing random split vs blocked validation.
   - Status: trap-blocked validation is complete; report it as a limitation-aware
     generalization check within the internationally harmonised sampling frame,
     not as a country-independent transfer claim. No country-out validation is
     planned because the available data do not support a comprehensive analysis.

3. Quantify observed-lag versus recursive-predicted-lag performance if feasible. [Reviewer 1]
   - Compare performance when previous observed egg counts are used against performance when previous predictions are recursively fed back.
   - This directly addresses deployment/forecasting distinction.
   - Status: complete on the matched AIMSurv subset. Add the raw R², log-R²,
     RMSE, accuracy, and coverage in the results table.

4. Recompute model summaries using all runs. [Reviewer 1]
   - Replace top-25%-filtered results with all-run mean/median and uncertainty.
   - Update tables/figures accordingly.
   - Status: complete in the revised reports; remove any remaining manuscript
     language that implies selection from only the top 25% of runs.

5. Make existing ablation/loss-function work clearer. [Reviewer 2]
   - The SI already compares WMSE, WMSLE, and ZINB; make this more visible in the main text or response.
   - Existing input-combination results can answer part of the ablation request.
   - Agreed window tests: 30/60/90 days, with WMSLE only for the 30/60-day extensions.
   - Status: the 30/60/90-day windows, no-lag/two-lag comparison, and loss
     comparisons are available in the revised results. Make their role in
     selecting the operational climate-only configuration explicit.

6. Improve the European prediction with autoregressive training.
   - Keep the completed five-seed climate-only projection as a comparator.
   - Train the frozen autoregressive candidates using model-generated lag values,
     zero seasonal initialization, five trap-disjoint folds, and five seeds.
   - Use the fixed 3,000-epoch final checkpoint for every autoregressive fit,
     matching the main Counter protocol; do not restore an earlier epoch using
     training, validation, or outer-test performance.
   - Use pooled outer-fold recursive RMSLE (square root of log-MSE) as the
     primary selection metric, with raw R², log-R², RMSE, and zero/nonzero
     accuracy as secondary metrics.
   - Permit European projection only if the candidate improves recursive RMSLE
     over the climate-only recursive reference without a material deterioration
     in raw R²; decide from pooled five-fold results, never from the best fold or
     best epoch.
   - Verify that the climate grid uses the same variable definitions, units,
     temporal aggregation, coordinate convention, and training-derived input
     normalization as the AIMSurv model.

7. Update the performance figures and tables.
   - Regenerate the main climate-only scatter figure with both raw-scale and
     log-scale panels for train/test or out-of-fold predictions, as appropriate.
   - Report raw R² and log-R² together, with RMSE and zero/nonzero accuracy as
     complementary metrics. Do not report accuracy for continuous European
     grid maps, where no observed labels exist.
   - Keep the observed-lag and recursive comparisons in the validation table or
     SI, so the operational choice is transparent without making the recursive
     model the headline result.

8. Add a light interpretability/importance analysis if feasible. [Reviewer 2]
   - Minimum: input ablation summary already partly available.
   - Optional: permutation importance for climate variables and lagged egg-count inputs.

9. Check Counter metric and selection coherence before manuscript finalization.
   - Distinguish configuration-selection metrics from final reporting metrics:
     mean blocked-fold raw R² selects the benchmark configuration; log-R², RMSE,
     accuracy, seed variability, and pooled OOF/bootstrap intervals describe its
     uncertainty and practical behaviour.
   - Report fold-mean and pooled OOF metrics with different labels. They answer
     related but non-identical summaries and must not be presented as
     contradictory estimates.
   - Keep the observed-lag model (`8bec14a3ec4c`), climate-only comparator
     (`450f9e361ceb`), and any autoregressive-trained model as separate rows and
     use-case claims. Do not rank them as if they solved the same deployment task.
   - For the scatter figure, call the selected object a configuration, not a
     best individual model: the screening configuration is selected by mean
     five-fold raw R², then all five seeds are averaged within each held-out
     fold. Each held-out observation therefore receives the mean prediction of
     five seed models; no seed is selected as the winner.
   - Label the in-sample scatter separately: each observation is averaged over
     the 20 models from the four folds that trained on it (four folds × five
     seeds), whereas the out-of-fold scatter uses only the five models from its
     held-out fold. The latter is the relevant generalization display.
   - Compare recursive models only on identical recursive rows, with coverage
     and initialization reported. Do not compare VectorNet metrics directly with
     AIMSurv metrics without identifying the different cohort and sampling
     context.

### Climate-data acquisition for the European rerun

The reusable downloader was found in the sibling repository
`/home/biazzin/git/Aedes-Albopictus-datasets`, rather than in the current
Counter experiment repository:

- `data/counter/src/copernicus_downloader.py` downloads hourly ERA5-Land from
  the Copernicus CDS through `cdsapi`, using the Europe area
  `[75, -25, 25, 45]`, a 0.1° grid, and monthly request chunks.
- It supports the required `2m_temperature`, `2m_dewpoint_temperature`, and
  `total_precipitation` variables and aggregates them into processed daily
  NetCDF files.
- `data/counter/src/copernicus_data.py` enables the downloader, checks for
  missing climate files, and links processed climate data to trap records.
- The sibling checkout already contains local, Git-ignored 2020 artifacts under
  `data/counter/input_data/climate/`: the processed daily files are about 1.6
  GB in total and the raw hourly files about 15 GB. In particular, the 2020
  processed temperature, dew-point, and precipitation files are potential
  inputs for the map rerun, so a fresh download was not necessary. A consolidated
  2020 map input has now been created locally at
  `experiments/counter/data/europe_copernicus_data_2020.nc`.
- The sibling repository documents the required CDS credentials and the
  `REBUILD_CLIMATE=1 bash make_counter_dataset.sh` workflow.

The 2020 artifact metadata and variable compatibility have been checked: the
grid contains 366 daily timestamps, 0.1° longitude/latitude spacing, Kelvin
temperature/dew-point fields, and metre precipitation sums. The climate-only
model's saved preprocessing expects these raw fields and performs the final
90-day weekly aggregation and training-derived z-score normalization at
prediction time. Do not commit the multi-gigabyte source artifacts or the
340-MB consolidated grid; document their acquisition or provide a reproducible
download command instead. The climate-only five-model map rerun is complete.
The remaining implementation step is the autoregressive-training experiment
and, if it passes the prespecified recursive criterion, its corresponding
European projection.

### Autoregressive-training extension

The implementation protocol is documented in
`experiments/counter/AUTOREGRESSIVE_TRAINING_PLAN.md`. It is a separate,
predeclared extension rather than a post-hoc replacement of the completed
Counter grid. It uses model-generated lag values throughout training and
evaluation, a declared zero-start seasonal initialization, the existing
trap-disjoint folds, five seeds, fixed 3,000-epoch training, and recursive
RMSLE as the primary deployment metric. The climate-only recursive comparator
must be evaluated on the same rows, and the European map decision must follow
the rule above.
