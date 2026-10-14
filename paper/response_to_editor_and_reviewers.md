# Response to the Editor and Reviewers

**Manuscript:** ECOINF-D-26-01884  
**Title:** *AIedes: a neural-network framework for modelling the distribution and abundance of Aedes albopictus in Europe*

> **Working-draft note:** This document contains the intended complete responses. Section names are provided now; page and line numbers must be inserted after the revised manuscript is finalized. Remove this note before submission.

Dear Editor and Reviewers,

Thank you for the careful and constructive assessment of our manuscript. We have substantially redesigned the validation and reporting of the AIedes-Counter, clarified the construction and validation of the AIedes-Classifier, and moderated the ecological and operational claims. The principal revisions are:

1. five-fold grouped cross-validation, using trap identity as the grouping variable so that complete traps, rather than individual observations, are held out from training;
2. five independent random-seed repetitions for each fold, retaining all 25 models and calculating data scaling from the training data only;
3. separate evaluation of predictions using observed previous counts, forecasts using the model's own previous predictions, and a climate-only model;
4. comparisons of 30-, 60-, and 90-day climate histories and one versus two previous weekly mosquito values, together with a separate climate-only reference; the existing comparisons of loss functions, network sizes, and dropout rates are retained;
5. performance evaluation on both the original and logarithmic scales, using R², log-R², RMSE, and zero/nonzero accuracy;
6. separate reporting of the results on the external VectorNet data; and
7. revised terminology and conclusions that refer specifically to ovitrap-derived weekly egg-laying rate and acknowledge the limits of geographic transfer and disease-risk interpretation.

Our point-by-point responses follow. The editor and reviewer comments are reproduced verbatim in block quotations.

---

## Response to the Editor

### Decision

> Thank you for submitting your manuscript to Ecological Informatics.
>
> I have completed my evaluation of your manuscript. The reviewers recommend reconsideration of your manuscript following major revision. I invite you to resubmit your manuscript after addressing the comments below. Please resubmit your revised manuscript by Sep 12, 2026.
>
> When revising your manuscript, please consider all issues mentioned in the reviewers' comments carefully: please outline every change made in response to their comments and provide suitable rebuttals for any comments not addressed. Please note that your revised submission may need to be re-reviewed.

**Response:** We thank the Editor for the opportunity to revise and resubmit the manuscript. We have addressed every editorial and reviewer comment below. For requests that could not be implemented within the present study, we provide a specific explanation and have adjusted the scope of the manuscript and its conclusions accordingly.

### Cheryl A Johnson — Editorial assessment

> both reviewers agree that the MS is relevant and the analyses provide a novel contribution. However, the reviewers have pointed out a number of areas that could greatly improve the quality of the work. This includes creating a more transparent and repeatable workflow and reconsideration of some of the analytical approaches (e.g. model run selection, stronger validation section). The MS would also benefit from further consideration of the application of the results. The reviewers have provided a number of additional helpful comments the authors should consider to increase the impact of their work.

**Response:** We fully agree that transparency, reproducibility, and stronger validation are central to the quality and impact of this work, and we have made them the main focus of the revision. The primary objective of the paper is to establish a quantitative and reproducible framework for modelling *Aedes albopictus* presence and ovitrap-derived weekly egg-laying activity. By providing harmonized public data, documented processing and training procedures, explicit validation rules, shared performance measures, code, and reference results, the study creates a common basis on which alternative statistical, machine-learning, and mechanistic approaches can subsequently be tested and compared.

The purpose of the present work is therefore to assess the predictive performance and limitations of this modelling approach, rather than to provide definitive projections under future climate scenarios or a validated disease-risk forecasting system. These applications are outside the scope of the present paper, which focuses on quantitative evaluation of model performance and on establishing a framework for comparison among modelling approaches. 

To strengthen the quantitative assessment, we substantially revised the validation and reporting. We now evaluate the Counter on traps that were not used to train the corresponding models, retain all repeated model runs, and examine how the main modelling choices affect performance. We distinguish predictions supported by earlier mosquito measurements from forecasts that rely on the model's own previous predictions. We also strengthened the spatial validation of the Classifier, clarified how its labels were constructed, and report the independent VectorNet evaluation separately. Together, these changes provide a more transparent and reliable account of model performance and its limitations.

Finally, we have reconsidered the interpretation and application of the results throughout the manuscript. We describe the Counter target specifically as ovitrap-derived egg-laying activity, and avoid presenting the outputs as direct estimates of adult-mosquito abundance or disease-transmission risk. The Discussion now presents climate-scenario projections, broader external validation, mechanistic comparison, explainability, and disease-risk modelling as future developments that can build on the quantitative framework established here.

**Location in revised manuscript:** Abstract; Materials and methods—Validation, model training, and metrics; Results—Classifier and Counter; Discussion—Applications and limitations; revised SI.

---

## Response to Reviewer 1

> The manuscript presents AIedes, a neural-network framework for modelling the distribution and weekly egg-laying activity of Aedes albopictus across Europe. The topic is timely and relevant, and the attempt to integrate harmonised surveillance data, climate covariates, open code, and reproducible workflows is well aligned with the scope of Ecological Informatics. The manuscript is generally well structured, and the combination of a presence/absence classifier with an abundance-oriented counter model is potentially useful for vector surveillance and ecological forecasting.
>
> However, I recommend major revision before the manuscript can be considered for publication. My main concerns relate to model validation, possible information leakage, and the strength of the ecological interpretation of the abundance predictions.

**Response:** We thank Reviewer 1 for recognizing the relevance of the framework and for identifying the validation and interpretation issues that required the most substantial revision. We address every point below.

### Major comment 1 — Validation and separation of training and test data

> First, the validation strategy needs to be strengthened and reported more transparently. Both the presence/absence and abundance models rely substantially on random train/test splits. For species distribution and abundance modelling, random splitting can overestimate predictive performance when spatially or temporally autocorrelated records from similar regions, traps, or time periods occur in both the training and test sets. The authors mention an additional NUTS3-level separation analysis in the Supporting Information, but the numerical results should be reported clearly in the main manuscript. For the abundance model, I also recommend a stricter blocked validation, for example by trap, region/country, or time period, to evaluate whether the model generalizes beyond nearby observations.

**Response:** We agree. For the revised Counter evaluation, we no longer use a random division of individual observations into training and test sets. As suggested by the reviewer, we now separate the data by trap: the complete series from a trap is assigned either to training or to evaluation and can never appear in both. We implement this as **five-fold grouped cross-validation**, with trap identity defining the groups, and repeat training with **five random seeds per fold**. These are standard procedures in neural-network evaluation. Grouped cross-validation keeps all observations from the same sampling unit together, while repeated random seeds measure the variation caused by random initialization and training.

Specifically, we divided the traps into five groups and withheld each group from training in turn. Thus, observations from the same trap can never occur in both the training and validation data for the same test. For each held-out group, we trained the model five times from different random starting points and retained all 25 resulting models. Data normalization and all other data-dependent preparation use the training data only. Each observation is evaluated using only the five models that did not train on its trap, and their predictions are averaged. On 5,129 observations from 387 traps, the model that uses actual previous observations obtains R² = 0.566, log-R² = 0.680, RMSE = 26.12, and zero/nonzero accuracy = 0.922.

For the Classifier, we now report a five-fold stratified grouped validation in which complete NUTS3 regions are withheld from training. Each observation is evaluated only by models trained without data from its region, and the predictions of five random-seed models are averaged within each fold. Across the five held-out regional folds, the selected neural network has mean accuracy 0.920 $\pm$ 0.007, specificity 0.936 $\pm$ 0.025, sensitivity 0.890 $\pm$ 0.048, and CSI 0.797 $\pm$ 0.020. Under the same held-region evaluation, the logistic reference obtains 0.888 $\pm$ 0.020, 0.911 $\pm$ 0.038, 0.844 $\pm$ 0.035, and 0.728 $\pm$ 0.036, respectively. Absolute performance is lower and more variable than in the random-split evaluation, while the improvement of the selected neural network over the logistic reference is maintained for all four metrics. We present this as an additional regional-transfer stress test; the random split remains the principal evaluation of interpolation across the observed European domain.

**Changes made in the manuscript:** The Counter Methods now define five-fold validation by complete traps, repeated training, training-fold preprocessing, and averaging of predictions from models trained on other traps. The corresponding Results, Discussion, scatter plot, time-series plot, and Supporting Information report predictions for held-out traps and the pooled performance across all 5,129 observations. The Classifier Methods describe the NUTS3-grouped spatial validation, its numerical results are reported explicitly in the main Results and Discussion, and the Supplementary Information compares the logistic reference and selected neural network on the same held regions.

### Major comment 2 — Selection of only the top 25% of runs

> Second, the procedure used for selecting model runs should be reconsidered. The manuscript states that only the top 25% of R² values were retained for each configuration to reduce the influence of poorly trained or overfitted runs. This approach can produce optimistically biased performance estimates. All runs should be reported, or at least the mean/median and confidence intervals across all independent runs should be provided. If unstable runs occur, this should be treated as part of model performance rather than excluded unless there is a pre-defined and objectively justified failure criterion.

**Response:** We agree and have removed the top-25% rule and the selection of a single “representative” model. Under the repeated grouped cross-validation procedure described above, all five random-seed repetitions are retained for every fold. Each observation is predicted by averaging the five models that did not train on its trap; none of these models is selected because it performed particularly well. The complete comparisons include all training repetitions and show the variation among them. The previous top-quartile results have been removed from both the manuscript and the Supplementary Information and play no role in the revised estimates or conclusions.

**Location:** Materials and methods—Model selection and validation; Results—Counter evaluation; revised model-performance table and SI sensitivity tables.

### Major comment 3 — Interpretation of “abundance”

> Third, the abundance model should be described more cautiously. The model predicts ovitrap-derived weekly egg-laying rates, not direct adult mosquito abundance. Egg counts are a useful proxy, but they are influenced by ovitrap placement, local breeding-site availability, sampling protocol, trap attractiveness, and micro-environmental factors. Therefore, statements referring broadly to "abundance" should be revised to "weekly egg-laying rate" or "ovitrap-derived egg abundance" where appropriate.

**Response:** We agree. We now use “ovitrap-derived weekly egg-laying rate” or “egg-laying activity” for the prediction target, retaining “abundance” only when explicitly described as a proxy. The Methods and Discussion explain that egg-laying observations may be affected by trap placement, trap attractiveness, competing breeding sites, sampling protocol, and local environmental conditions. Nevertheless, ovitrap egg counts provide a useful and widely available proxy for mosquito activity when direct adult-mosquito observations are limited. They should not be interpreted as a direct measure of adult abundance or pathogen-transmission risk. When sufficiently large and harmonized adult-trap datasets become available, the same data-processing, training, and validation framework described in this manuscript could be adapted and retrained to predict adult mosquito abundance directly.

**Location:** Title/Abstract where appropriate; Materials and methods—Data and target; Results; Discussion—Interpretation and limitations.

### Major comment 4 — Previous observations versus true forecasting

> Fourth, the use of lagged egg counts is biologically reasonable, but it also creates a forecasting distinction that should be clarified. When previous egg counts are available, the model partly uses recent entomological observations. In the spatial prediction maps, however, the authors appear to initialize the season using only climate variables and then iteratively use model-predicted previous values. The manuscript should more clearly separate hindcasting with observed lagged values from true forecasting using predicted lagged values, and should quantify performance under both conditions.

**Response:** We agree. The original manuscript did not clearly distinguish a prediction made with recent field observations from a forecast that must continue without new mosquito measurements. We now describe and evaluate these as different uses of the model.

For a fair comparison, all four methods below were evaluated on exactly the same 5,036 observations from 375 traps. The harmonized model dataset stores the 90-day climate history associated with each mosquito observation, rather than a separate complete daily climate series for every trap. For 375 traps, the overlapping stored windows were sufficient to reconstruct an uninterrupted climate sequence for recursive prediction. For 12 traps, containing 93 observations, gaps remained between the available windows. We restricted this comparison to sequences that could be reconstructed entirely from the archived model inputs, ensuring that every method used the same climate data and preprocessing. This also avoids combining the released analysis dataset with climate values retrieved and processed separately at a later stage. The exclusion was determined only by continuity of the stored climate inputs, not by egg counts or prediction performance.

We used **five-fold grouped cross-validation by trap, with five random-seed repetitions per fold**. As described above, this combines two standard neural-network evaluation practices: keeping related observations together during cross-validation and repeating stochastic model training from different random initializations. In practical terms, the traps were divided into five groups, each group was withheld from training in turn, and the model was trained five times for every held-out group. Therefore, every reported prediction is the average of five models that had never been trained on the corresponding trap. The four methods differ only in how information about previous mosquito activity is used:

1. **Prediction using observed history.** During both training and evaluation, the model receives 90 days of daily climate data and, when available, the measured egg-laying rates from approximately 7 and 14 days earlier. Additional indicators tell the model when either previous measurement is unavailable. This represents prediction during an active surveillance programme, where recent trap measurements are known. It is not a fully independent forecast.

2. **Earlier model used recursively.** This is the same model described above and it is still trained with measured previous egg-laying rates. During evaluation, however, the measured previous rates are withheld. The model proceeds forward in time and its predictions from 7 and 14 days earlier are used in place of measurements. Before enough earlier predictions exist, these inputs are marked as unavailable. This experiment measures the loss of performance caused by using the model differently from the way it was trained.

3. **Model trained for recursive forecasting.** This model never receives measured previous egg-laying rates, either during training or evaluation. Whenever no earlier model prediction is available—including at the start of each trap series—the corresponding previous rate is set to zero. Unlike the observed-history model, it does not receive separate indicators for missing previous values. The observations for each trap are processed in chronological order, and the model's own earlier predictions are used as the 7- and 14-day inputs. During training, the sequence is regenerated after every training cycle using the model's updated predictions. The model is therefore trained under the same conditions in which it is later used for forecasting.

4. **Climate-only model.** This model receives the climate history but no previous measured or predicted mosquito values. It provides a reference showing how much predictive information is gained by feeding earlier model predictions back into the forecasting model.

| Mode | Previous mosquito information during training | Previous mosquito information during evaluation | R² | log-R² | RMSE | Accuracy |
|---|---|---|---:|---:|---:|---:|
| Prediction using observed history | Measured values | Measured values | 0.565 | 0.678 | 26.36 | 0.921 |
| Earlier model used recursively | Measured values | Model predictions | 0.456 | 0.649 | 29.48 | 0.916 |
| Model trained for recursive forecasting | Model predictions, starting from zero | Model predictions, starting from zero | 0.535 | 0.663 | 27.25 | 0.919 |
| Climate only | None | None | 0.519 | 0.664 | 27.72 | 0.921 |

The model using measured previous values performs best, but it requires continuing trap observations. When this model is instead forced to use its own predictions, R² decreases from 0.565 to 0.456 and RMSE increases from 26.36 to 29.48. The R² = 0.456 value therefore describes the earlier model used in a way for which it was not trained; it is not the performance of the final forecasting model.

Training directly with model-generated previous values increases recursive-forecast R² to 0.535 and reduces RMSE to 27.25. Its log-R² and zero/nonzero accuracy are similar to those of the climate-only model. We therefore do not claim that it is superior for every metric. We select it for forecasts without continuing trap observations because its training procedure matches its intended use and because it improves raw-scale R² and RMSE relative to the climate-only model while retaining information about temporal continuity. The same zero-start procedure must be used when generating the European projections.

**Location:** Materials and methods—Forecasting modes and initialization; Results—Forecasting-mode comparison; Discussion—Forecasting use cases.

### Major comment 5 — Climate-based presence-label mask and circularity

> Fifth, the presence/absence label construction should be clarified. The manuscript indicates that presence labels were assigned to grid cells within positive administrative units only when key climatic indicators satisfied predefined criteria. Because the model itself uses climate variables as predictors, the authors should explain how this masking procedure avoids circularity or inflated performance. The criteria used for this masking should be clearly summarized in the main manuscript, not only in the Supporting Information.

**Response:** We agree that the original description was insufficient. The revised Methods report the three criteria: annual precipitation ≥ 200 mm, coldest-month mean temperature ≥ −3 °C, and annual mean temperature ≥ 10 °C. Of 20,832 cells initially assigned positive labels from positive administrative units, 16,164 were retained and 4,668 (22.4%) were relabelled; 4,662 failed a temperature criterion and six failed the precipitation criterion.

The mask is a conservative, predefined correction for the difference in spatial resolution between the surveillance and climate data. A record of establishment at the NUTS3 level shows that the species occurs somewhere within that administrative region; it does not imply presence in every 12.5-km grid cell within it. Assigning the positive NUTS3 label indiscriminately to all cells would therefore create implausible positive labels, for example in very cold parts of otherwise positive regions. The mask only removes such positive labels; it never creates a presence record where surveillance did not report one.

The thresholds were informed by the established ecological limits reported by ECDC but were deliberately set more conservatively for this filtering step. They were defined independently of model training and were neither fitted nor tuned using our training or test data. The filtering also follows the same general approach used by Zardini et al. for downscaling mosquito-distribution data. The procedure is applied before model training and identically across all data. It therefore does not constitute leakage of test observations into model training.

We agree that the classifier can learn the broad climatic limits represented by the mask, but this is consistent with its intended purpose: estimating climate-constrained suitability within the observed distribution of the species. An evaluation using the unfiltered NUTS3 labels would not provide a more reliable test, because it would treat every cell in a positive administrative region as a confirmed presence even when no cell-level observation exists and the local climate is incompatible with establishment.

To make the procedure transparent, we now report the thresholds, their independent source, the methodological link to Zardini et al., and the number of cells affected. We also clarify that the reported accuracy refers to the climate-filtered distribution target. We do not claim that it is validation against independent cell-level occurrence observations, which are not available at the same European scale. The additional validation that withholds complete NUTS3 regions addresses the separate concern that neighbouring cells from the same administrative region could otherwise occur in both training and testing.

**Location:** Materials and methods—Presence-label construction; Results—Classifier; Discussion—Classifier limitations; SI—Climate-based filtering.

> Additional points:

### Additional point 1 — Exact results when regions are withheld

> 1. The authors should provide the exact performance of the spatially blocked validation in the main text, not only state that performance was "slightly reduced but still high."

**Response:** Done. The revised main text reports the results obtained when complete NUTS3 regions are withheld. Across the five held-out regional folds, mean accuracy is 0.920 $\pm$ 0.007, specificity is 0.936 $\pm$ 0.025, sensitivity is 0.890 $\pm$ 0.048, and CSI is 0.797 $\pm$ 0.020. These values are lower than the random-split results because the NUTS3 analysis measures geographic transfer to regions excluded from training. Under the same held-region evaluation, the selected neural network continues to outperform the logistic reference for all four metrics.

**Location:** Results—Predicting local mosquito presence and absence.

### Additional point 2 — Separate VectorNet metrics

> 2. The external validation using VectorNet data from Lazio is valuable, but performance metrics for this independent dataset should be reported separately.

**Response:** Done. VectorNet is reported as a separate external stress test consisting of 280 weekly observations from 40 traps between May and November 2021. The overall metrics are R² = −0.004, log-R² = 0.170, RMSE = 69.82, and accuracy = 0.754. These results are not pooled with AIMSurv performance.

Visual inspection of the trap time series suggested that the model frequently failed to reproduce a second seasonal peak occurring around September–October. To examine whether this later peak was an important source of error, we performed an exploratory split of the external data into observations before September and observations from September onward. Before September, R² = 0.072 and log-R² = 0.360 (n = 162), whereas from September onward R² = −0.071 and log-R² = −0.076 (n = 118). The deterioration in the later period is consistent with the missing second peak contributing to the weak overall result. However, performance before September also remains limited, particularly on the original scale, so the autumn peak does not fully explain the poor external performance. The SI provides separate raw- and log-scale scatter plots for the two periods, and the split is clearly identified as a post hoc diagnostic rather than a pre-specified validation analysis.

Several characteristics of the external dataset may contribute to this result. The observations come from a geographically restricted, elevation-focused study that includes mountain and valley environments. Local conditions in such settings—for example differences between valley bottoms, slopes, and higher-elevation sites—may not be represented adequately by the spatial resolution of the gridded climate inputs. The available metadata also do not establish that all aspects of trap placement, collection, and data processing were identical to the harmonized AIMSurv protocol. Finally, the external dataset is small and covers only one region and one season, making its performance metrics particularly sensitive to a limited number of seasonal peaks. We present these factors as plausible explanations, not as demonstrated causes, and interpret VectorNet as a demanding but limited external test rather than evidence of broad geographic validation.

**Location:** Materials and methods—External data; Results—External stress test; SI—VectorNet evaluation.

### Additional point 3 — Surveillance bias, absence certainty, and excluded categories

> 3. The manuscript should discuss limitations related to surveillance bias, heterogeneous absence certainty, and the exclusion of "introduced," "unknown," and "no data" categories.

**Response:** We have expanded this discussion. “Introduced” is excluded because detection without confirmed establishment is not equivalent to either established presence or absence. “Unknown” and “no data” are excluded because they do not provide verified binary labels. The “absent” category is retained but its certainty depends on surveillance effort; consequently, absence labels may contain false absences. Predictions in unsampled or unknown areas are framed as priorities for further surveillance, not confirmation of presence or absence. For the Counter, we quantify the strong geographic and positive-record imbalance and limit our claims about how broadly the model can be applied.

**Location:** Materials and methods—Presence data; Discussion—Surveillance and geographic bias.

### Additional point 4 — Cautious interpretation of classifier performance

> 4. The classifier reports very high accuracy, sensitivity, specificity, and CSI. These values are promising, but they should be interpreted cautiously given the potential effects of spatial autocorrelation and climate-based masking.

**Response:** We agree that the high scores should be reported with their methodological context. We now explain that the climate mask is a conservative, predefined correction for downscaling NUTS-level records and was not fitted using the training or test data. We have also removed unqualified terms such as “excellent” and added the results obtained by withholding complete NUTS3 regions. The classifier shows strong agreement with the defined climate-filtered distribution target under random held-out evaluation. The NUTS3 stress test gives lower absolute performance, while preserving the improvement over the logistic reference. Because NUTS3 regions are administrative rather than climatic units, we interpret this analysis as geographic transfer to regions excluded from training. We do not present either evaluation as direct validation against independent cell-level field observations.

**Location:** Results—Classifier; Discussion—Classifier limitations.

### Additional point 5 — Terminology and inconsistent numbers of runs/metrics

> 5. Please correct minor inconsistencies in terminology and reporting. For example, the manuscript refers to "three standard evaluation metrics" but lists four metrics. The number of independent runs also appears inconsistent in places, with the text referring to 20 runs and the table caption referring to the five best runs out of ten total runs.

**Response:** Corrected. The text now refers to four classifier metrics. For the revised Counter, the design is described consistently as five-fold grouped cross-validation by trap with five random-seed repetitions per fold, giving 25 retained models. The earlier references to 20 runs, ten runs, the top five, and a selected representative model have been removed from the principal analysis. We also distinguish the broad comparison of model settings from the repeated training used for the final evaluation.

**Location:** Materials and methods—Metrics and training; table and figure captions; SI.

### Additional point 6 — Figure readability

> 6. Figure quality is generally acceptable, but some map panels and time-series panels would benefit from improved readability, larger labels, and clearer legends.

**Response:** We have revised the relevant maps and time-series panels with larger text, clearer legends, and more explicit captions. The scatter plot now shows predictions made only by models that did not train on the corresponding trap. It has separate panels for the original and logarithmic scales and reports R², log-R², and RMSE. The main sensitivity figure remains focused on the selected 90-day setting for readability, while complete comparisons of input periods and loss functions are moved to the SI.

**Location:** Revised main figures and captions; SI sensitivity figures.

### Reviewer 1 concluding assessment

> Overall, the manuscript has clear potential and addresses an important applied problem in vector ecology. However, the validation framework and interpretation of model performance need substantial clarification before the conclusions can be considered fully supported.

**Response:** We thank the reviewer for this assessment. The revised validation framework, forecasting comparison, external-data reporting, and more cautious interpretation of the target and applications directly address these central concerns.

---

## Response to Reviewer 2

We thank Reviewer 2 for proposing a broader set of sensitivity, benchmark, interpretability, external-validation, and application analyses. We implemented the requests central to validating the present framework and clarified where other suggestions require data or modelling components beyond the scope of this study.

### Comment 1 — Input periods, previous observations, model structure, and loss functions

> 1. The AIedes-Counter model relies on 90-day daily time-series climate inputs and autoregressive lagged oviposition variables, yet the manuscript does not systematically test the effects of different time window lengths (30d/60d/120d) and autoregressive lag orders on model performance. The hyperparameter screening of dropout regularization, network layers and neuron numbers only presents optimal results, without complete ablation experiments to support the rationality of the optimal configuration selection. In addition, for the zero-inflated feature that 80% of egg counts in the dataset equal zero, only WMSLE loss is adopted, while the performance differences of loss functions suitable for discrete count data such as zero-inflated negative binomial are not compared. It is suggested to supplement complete ablation experiments, time window sensitivity tests and comparative experiments of multiple loss functions to improve the basis for model stability and hyperparameter selection.

**Response:** The submitted manuscript already compared WMSE, WMSLE, and a ZINB-inspired loss. It also included a systematic search across different input combinations, daily and weekly climate summaries, network structures, and dropout rates. These comparisons were described in the main text and presented in detail in the Supplementary Information; WMSLE was therefore selected after comparison with alternative loss functions.

Following the reviewer’s suggestion, we extended this analysis by comparing 30-, 60-, and 90-day climate histories. We also added a controlled comparison between models using one and two previous weekly mosquito values for the selected 90-day configuration, while the existing climate-only models provide a separate reference without mosquito-history inputs. Across the full analysis, 516 combinations of inputs, loss functions, network structures, and dropout rates were evaluated. Performance generally improves from 30 to 60 to 90 days, and daily inputs generally perform better than weekly summaries, supporting the choice of 90 days of daily input.

The reviewer specifically suggested testing 30-, 60-, and 120-day windows. We added 30- and 60-day windows to the existing 90-day analysis, but we did not test 120 days. The harmonized model dataset stores the 90 climate days preceding each mosquito observation. The 30- and 60-day inputs can therefore be obtained as exact subsets of the same archived climate records, preserving the observations, climate source, and preprocessing used in the original analysis. A 120-day window would require retrieving additional climate data and rebuilding the complete climate-linked dataset rather than performing a directly comparable sensitivity analysis on the frozen model inputs. We now state explicitly that the tested range is 30–90 days and do not draw conclusions about whether a window longer than 90 days could provide further improvement.

**Location:** Materials and methods—Sensitivity analysis and network structures; Results—Input and forecasting comparisons; SI—Complete sensitivity figures and tables.

### Comment 2 — Future RCP4.5 projections, disease-risk coupling, and operational use

> 2. This study only completes the spatio-temporal simulation of mosquito abundance under current climatic conditions for a single year (2020), and fails to make full use of the RCP4.5 climate scenario mentioned in the manuscript to predict the expansion of Aedes albopictus under future climate change. Meanwhile, the model output results lack coupling analysis with the transmission risk of vector-borne infectious diseases, which cannot directly support practical application scenarios such as vector control and public health early warning. It is recommended to supplement the prediction results of distribution and abundance under medium and long-term climate scenarios, correlate the model's oviposition rate output with the local transmission risk of dengue and chikungunya, and elaborate on how this framework can serve practical CDC operations including targeted disinfection and monitoring site planning across European regions.

**Response:** We thank the reviewer for these suggestions. However, medium- and long-term climate projections and disease-risk modelling are beyond the objective of the present paper. The purpose of this study is to establish a quantitative and reproducible framework for predicting the ovitrap-derived weekly egg-laying rate of *Aedes albopictus* and for comparing the performance of alternative modelling approaches. We describe the harmonized public surveillance dataset released in the companion data paper, define a clear validation procedure and performance measures, and provide a reference neural-network model, code, and results. This gives other groups a common basis on which to test alternative methods and compare their performance with ours.

Applying this framework to future climate scenarios is an important subsequent research objective, but it is not part of the model evaluation undertaken here. Such an analysis would require a dedicated experimental design and an assessment of extrapolation beyond the climatic conditions represented in the training data. We have therefore removed or revised wording that could imply that an RCP4.5 projection was evaluated in the present study and identify future-climate forecasting as a distinct next step.

Similarly, translating predicted egg-laying activity into dengue or chikungunya transmission risk would constitute a further modelling stage, rather than an additional output of the current model. Disease-risk estimation requires epidemiological and entomological information that is not contained in the present dataset. We therefore do not claim that the current egg-laying predictions directly quantify disease risk or provide a validated system for targeted control. Instead, the present work supplies the quantitative modelling and benchmarking framework on which future climate-projection and disease-risk studies may build.

**Location:** Materials and methods—Climate data; Discussion—Operational scope and future work; Conclusion.

### Comment 3 — Additional benchmark algorithms and explainability

> 3. This paper only compares the classification performance between basic logistic regression and the optimal deep neural network, lacking comprehensive horizontal comparisons with mainstream species distribution models (MaxEnt, GAM) and classic ensemble machine learning algorithms (XGBoost, stacked machine learning) in this field. In addition, as a black-box model, the neural network fails to carry out interpretability analysis on variable importance, climate threshold effects and nonlinear interaction mechanisms, making it difficult to clarify how temperature, precipitation and dew point temperature drive the distribution and oviposition dynamics of *Aedes albopictus*. It is suggested to supplement complete control experiments of multiple benchmark models, add explainable AI analysis such as SHAP and permutation feature importance, and systematically interpret the ecological mechanisms driven by climate in the Results and Discussion sections.

**Response:** We thank the reviewer for this suggestion. Logistic regression was selected as the initial reference because it has the same functional form as the model used by Zardini et al. and provided the starting point for the first AIedes-Classifier experiment. The comparison therefore shows whether adding hidden nonlinear layers improves prediction relative to this established and directly related baseline; it was not intended as an exhaustive comparison with every species-distribution or machine-learning method.

Comparisons with MaxEnt, GAM, XGBoost, ensemble approaches, and mechanistic models would be valuable. However, each approach requires specific expertise, assumptions, implementation choices, and model tuning. One of the central aims of this work is to make such comparisons easier and more rigorous for the wider research community. The harmonized dataset is publicly available through the companion data paper, and the present study supplies reproducible code, explicit validation procedures, common performance measures, and reference results. Other research groups can therefore apply their preferred statistical, machine-learning, or mechanistic models to the same prediction task and compare them under a shared quantitative framework. We have clarified that our results establish a reference benchmark and do not demonstrate that neural networks are universally superior to other modelling approaches.

We also already discuss the limited interpretability of neural networks and their black-box character as an important limitation. A systematic investigation of variable importance, climate thresholds, nonlinear interactions, and comparisons with mechanistic explanations would constitute a substantial study in its own right. Although explainable-AI and hybrid data-driven/mechanistic approaches are important future directions, the present paper focuses on predictive performance, reproducibility, and quantitative comparison. We therefore do not add a full SHAP or permutation-importance analysis here, but we identify interpretability and comparison with mechanistic models explicitly as priorities for future research.

**Location:** Introduction—Scope and contribution; Results—Input comparisons; Discussion—Benchmarking and interpretability limitations.

### Comment 4 — Broader external validation and geographic imbalance

> 4. The training dataset of the AIedes-Counter abundance prediction model established in this study heavily relies on the standardized AIMSurv surveillance dataset, while external validation only adopts a small number of VectorNet samples from a single mountainous area (Lazio, Italy). The verification of the model's generalization ability across regions and monitoring systems is insufficient. It is recommended to significantly expand independent observation datasets covering multiple countries, diverse landforms (high-altitude, coastal, inland urban areas) and different sampling protocols for stratified external validation. Meanwhile, the impact of geographical imbalance of surveillance data (enriched samples in Southern Europe, scarce samples in Northern and Eastern Europe) on prediction bias should be quantitatively analyzed, and the limitations of the model caused by data bias should be supplemented in the Discussion section.

**Response:** We agree that additional independent observations would strengthen the assessment of model transfer. However, AIMSurv is not a small or single-centre surveillance dataset. It is the first pan-European harmonized surveillance initiative for invasive *Aedes* mosquitoes and, to our knowledge, one of the most extensive, coherent, and openly available sources of standardized ovitrap observations in Europe. The original initiative involved 42 volunteer teams from 24 countries (Miranda et al., 2022). Its protocol harmonized sampling methods, sampling frequency, minimum survey duration, and data reporting, while defining both minimum and recommended protocols to achieve a practical balance between measurement comparability, geographic participation, and the resources available to each team. The present Counter analysis uses the subset of these data that satisfies the species, life-stage, sampling-duration, and climate-data requirements of our prediction task.

The geographical distribution of the observations reflects the reality of mosquito surveillance: monitoring groups and longer time series are concentrated mainly in areas where *Ae. albopictus* is established and routinely surveyed. We now quantify this imbalance in the manuscript: Austria contributes 54.6% of the observations, while Spain, Albania, and Italy account for 97.3% of the positive observations. We acknowledge that the dataset is therefore not a geographically uniform sample of Europe. At present, however, we are not aware of a comparably extensive open dataset with sufficiently harmonized weekly ovitrap measurements that could be added as a second broad external test.

We do not consider separation by country to be the most appropriate primary validation for this entomological prediction task. National borders are administrative boundaries rather than ecological ones, and a single country may contain markedly different climatic, topographic, and environmental conditions, while neighbouring areas in different countries may be ecologically similar. Moreover, given the uneven number of observations and positive records among countries, country-level performance would be dominated by a few surveillance programmes and would be unstable for sparsely sampled countries. We therefore use grouped cross-validation by trap, which evaluates prediction at locations not used for training while retaining the range of environmental conditions represented in the available data. We explicitly limit the interpretation to unseen traps within those sampled conditions and do not claim that this validation proves transfer to every country or environmental setting in Europe.

We nevertheless searched for an independent source and used the compatible VectorNet observations as an external stress test. After applying the same inclusion and processing criteria, only 280 weekly observations from 40 traps in one mountainous area of Lazio were available. We report the resulting weak performance transparently, but the restricted geography, small sample, local mountain-and-valley conditions, and possible protocol differences prevent this subset from providing a definitive test of continental generalization. Broader external validation will become possible only when additional compatible, harmonized, and openly available surveillance data are released; these data can then be incorporated and evaluated using the same framework.

**Location:** Materials and methods—Dataset composition; Results—VectorNet; Discussion—Geographic transfer and surveillance bias; Conclusion.

---

## Summary of revised quantitative results

For clarity, the principal values used throughout the revised manuscript are:

- **Classifier, with complete NUTS3 regions withheld:** accuracy 0.920 $\pm$ 0.007; specificity 0.936 $\pm$ 0.025; sensitivity 0.890 $\pm$ 0.048; CSI 0.797 $\pm$ 0.020.
- **Counter using measured previous values, evaluated only on traps excluded from training (n = 5,129):** R² 0.566; log-R² 0.680; RMSE 26.12; accuracy 0.922.
- **Model trained to use its own previous predictions (n = 5,036):** R² 0.535; log-R² 0.663; RMSE 27.25; accuracy 0.919.
- **Earlier model forced to use its own predictions:** R² 0.456; log-R² 0.649; RMSE 29.48; accuracy 0.916.
- **VectorNet external stress test (n = 280):** R² −0.004; log-R² 0.170; RMSE 69.82; accuracy 0.754.

We thank the Editor and Reviewers again. Their comments led to a substantially more transparent validation design, a clearer distinction between estimation and forecasting, and conclusions that better reflect the available evidence.
