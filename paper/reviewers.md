

From: em.ecoinf.0.9d15ab.04605726@editorialmanager.com <em.ecoinf.0.9d15ab.04605726@editorialmanager.com> On Behalf Of Ecological Informatics
Sent: Wednesday, July 29, 2026 5:50 PM
To: BIAZZO Indaco (JRC-ISPRA) <Indaco.BIAZZO@ec.europa.eu>
Subject: Decision on submission to Ecological Informatics

CC: cheryl-ann.johnson@ec.gc.ca
Manuscript Number: ECOINF-D-26-01884  

AIedes: a neural-network framework for modelling the distribution and abundance of Aedes albopictus in Europe

Dear Dr Biazzo,

Thank you for submitting your manuscript to Ecological Informatics.

I have completed my evaluation of your manuscript. The reviewers recommend reconsideration of your manuscript following major revision. I invite you to resubmit your manuscript after addressing the comments below. Please resubmit your revised manuscript by Sep 12, 2026.

When revising your manuscript, please consider all issues mentioned in the reviewers' comments carefully: please outline every change made in response to their comments and provide suitable rebuttals for any comments not addressed. Please note that your revised submission may need to be re-reviewed. 

To submit your revised manuscript, please log in as an author at https://www.editorialmanager.com/ecoinf/, and navigate to the "Submissions Needing Revision" folder.  

Research Elements (optional)
This journal encourages you to share research objects - including your raw data, methods, protocols, software, hardware and more – which support your original research article in a Research Elements journal. Research Elements are open access, multidisciplinary, peer-reviewed journals which make the objects associated with your research more discoverable, trustworthy and promote replicability and reproducibility. As open access journals, there may be an Article Publishing Charge if your paper is accepted for publication. Find out more about the Research Elements journals at https://www.elsevier.com/authors/tools-and-resources/research-elements-journals?dgcid=ec_em_research_elements_email.


Ecological Informatics values your contribution and I look forward to receiving your revised manuscript.

Kind regards,  
George Arhonditsis  
Editor-in-Chief  

Ecological Informatics

Editor and Reviewer comments:



Cheryl A Johnson: both reviewers agree that the MS is relevant and the analyses provide a novel contribution. However, the reviewers have pointed out a number of areas that could greatly improve the quality of the work. This includes creating a more transparent and repeatable workflow and reconsideration of some of the analytical approaches (e.g. model run selection, stronger validation section). The MS would also benefit from further consideration of the application of the results. The reviewers have provided a number of additional helpful comments the authors should consider to increase the impact of their work.


Reviewer #1: The manuscript presents AIedes, a neural-network framework for modelling the distribution and weekly egg-laying activity of Aedes albopictus across Europe. The topic is timely and relevant, and the attempt to integrate harmonised surveillance data, climate covariates, open code, and reproducible workflows is well aligned with the scope of Ecological Informatics. The manuscript is generally well structured, and the combination of a presence/absence classifier with an abundance-oriented counter model is potentially useful for vector surveillance and ecological forecasting.

However, I recommend major revision before the manuscript can be considered for publication. My main concerns relate to model validation, possible information leakage, and the strength of the ecological interpretation of the abundance predictions.

First, the validation strategy needs to be strengthened and reported more transparently. Both the presence/absence and abundance models rely substantially on random train/test splits. For species distribution and abundance modelling, random splitting can overestimate predictive performance when spatially or temporally autocorrelated records from similar regions, traps, or time periods occur in both the training and test sets. The authors mention an additional NUTS3-level separation analysis in the Supporting Information, but the numerical results should be reported clearly in the main manuscript. For the abundance model, I also recommend a stricter blocked validation, for example by trap, region/country, or time period, to evaluate whether the model generalizes beyond nearby observations.

Second, the procedure used for selecting model runs should be reconsidered. The manuscript states that only the top 25% of R² values were retained for each configuration to reduce the influence of poorly trained or overfitted runs. This approach can produce optimistically biased performance estimates. All runs should be reported, or at least the mean/median and confidence intervals across all independent runs should be provided. If unstable runs occur, this should be treated as part of model performance rather than excluded unless there is a pre-defined and objectively justified failure criterion.

Third, the abundance model should be described more cautiously. The model predicts ovitrap-derived weekly egg-laying rates, not direct adult mosquito abundance. Egg counts are a useful proxy, but they are influenced by ovitrap placement, local breeding-site availability, sampling protocol, trap attractiveness, and micro-environmental factors. Therefore, statements referring broadly to "abundance" should be revised to "weekly egg-laying rate" or "ovitrap-derived egg abundance" where appropriate.

Fourth, the use of lagged egg counts is biologically reasonable, but it also creates a forecasting distinction that should be clarified. When previous egg counts are available, the model partly uses recent entomological observations. In the spatial prediction maps, however, the authors appear to initialize the season using only climate variables and then iteratively use model-predicted previous values. The manuscript should more clearly separate hindcasting with observed lagged values from true forecasting using predicted lagged values, and should quantify performance under both conditions.

Fifth, the presence/absence label construction should be clarified. The manuscript indicates that presence labels were assigned to grid cells within positive administrative units only when key climatic indicators satisfied predefined criteria. Because the model itself uses climate variables as predictors, the authors should explain how this masking procedure avoids circularity or inflated performance. The criteria used for this masking should be clearly summarized in the main manuscript, not only in the Supporting Information.

Additional points:

1. The authors should provide the exact performance of the spatially blocked validation in the main text, not only state that performance was "slightly reduced but still high."
2. The external validation using VectorNet data from Lazio is valuable, but performance metrics for this independent dataset should be reported separately.
3. The manuscript should discuss limitations related to surveillance bias, heterogeneous absence certainty, and the exclusion of "introduced," "unknown," and "no data" categories.
4. The classifier reports very high accuracy, sensitivity, specificity, and CSI. These values are promising, but they should be interpreted cautiously given the potential effects of spatial autocorrelation and climate-based masking.
5. Please correct minor inconsistencies in terminology and reporting. For example, the manuscript refers to "three standard evaluation metrics" but lists four metrics. The number of independent runs also appears inconsistent in places, with the text referring to 20 runs and the table caption referring to the five best runs out of ten total runs.
6. Figure quality is generally acceptable, but some map panels and time-series panels would benefit from improved readability, larger labels, and clearer legends.

Overall, the manuscript has clear potential and addresses an important applied problem in vector ecology. However, the validation framework and interpretation of model performance need substantial clarification before the conclusions can be considered fully supported.


Reviewer #2: 1. The AIedes-Counter model relies on 90-day daily time-series climate inputs and autoregressive lagged oviposition variables, yet the manuscript does not systematically test the effects of different time window lengths (30d/60d/120d) and autoregressive lag orders on model performance. The hyperparameter screening of dropout regularization, network layers and neuron numbers only presents optimal results, without complete ablation experiments to support the rationality of the optimal configuration selection. In addition, for the zero-inflated feature that 80% of egg counts in the dataset equal zero, only WMSLE loss is adopted, while the performance differences of loss functions suitable for discrete count data such as zero-inflated negative binomial are not compared. It is suggested to supplement complete ablation experiments, time window sensitivity tests and comparative experiments of multiple loss functions to improve the basis for model stability and hyperparameter selection.
2. This study only completes the spatio-temporal simulation of mosquito abundance under current climatic conditions for a single year (2020), and fails to make full use of the RCP4.5 climate scenario mentioned in the manuscript to predict the expansion of Aedes albopictus under future climate change. Meanwhile, the model output results lack coupling analysis with the transmission risk of vector-borne infectious diseases, which cannot directly support practical application scenarios such as vector control and public health early warning. It is recommended to supplement the prediction results of distribution and abundance under medium and long-term climate scenarios, correlate the model's oviposition rate output with the local transmission risk of dengue and chikungunya, and elaborate on how this framework can serve practical CDC operations including targeted disinfection and monitoring site planning across European regions.
3. This paper only compares the classification performance between basic logistic regression and the optimal deep neural network, lacking comprehensive horizontal comparisons with mainstream species distribution models (MaxEnt, GAM) and classic ensemble machine learning algorithms (XGBoost, stacked machine learning) in this field. In addition, as a black-box model, the neural network fails to carry out interpretability analysis on variable importance, climate threshold effects and nonlinear interaction mechanisms, making it difficult to clarify how temperature, precipitation and dew point temperature drive the distribution and oviposition dynamics of *Aedes albopictus*. It is suggested to supplement complete control experiments of multiple benchmark models, add explainable AI analysis such as SHAP and permutation feature importance, and systematically interpret the ecological mechanisms driven by climate in the Results and Discussion sections.
4. The training dataset of the AIedes-Counter abundance prediction model established in this study heavily relies on the standardized AIMSurv surveillance dataset, while external validation only adopts a small number of VectorNet samples from a single mountainous area (Lazio, Italy). The verification of the model's generalization ability across regions and monitoring systems is insufficient. It is recommended to significantly expand independent observation datasets covering multiple countries, diverse landforms (high-altitude, coastal, inland urban areas) and different sampling protocols for stratified external validation. Meanwhile, the impact of geographical imbalance of surveillance data (enriched samples in Southern Europe, scarce samples in Northern and Eastern Europe) on prediction bias should be quantitatively analyzed, and the limitations of the model caused by data bias should be supplemented in the Discussion section.




More information and support 

FAQ: How do I revise my submission in Editorial Manager?
https://service.elsevier.com/app/answers/detail/a_id/28463/supporthub/publishing/

FAQ: How can I reset a forgotten password? 
https://service.elsevier.com/app/answers/detail/a_id/28452/supporthub/publishing/ 
For further assistance, please visit our customer service site: https://service.elsevier.com/app/home/supporthub/publishing/
Here you can search for solutions on a range of topics, find answers to frequently asked questions, and learn more about Editorial Manager via interactive tutorials. You can also talk 24/7 to our customer support team by phone and 24/7 by live chat and email

At Elsevier, we want to help all our authors to stay safe when publishing. Please be aware of fraudulent messages requesting money in return for the publication of your paper. If you are publishing open access with Elsevier, bear in mind that we will never request payment before the paper has been accepted. We have prepared some guidelines (https://www.elsevier.com/connect/authors-update/seven-top-tips-on-stopping-apc-scams ) that you may find helpful, including a short video on Identifying fake acceptance letters (https://www.youtube.com/watch?v=o5l8thD9XtE ). Please remember that you can contact Elsevier s Researcher Support team (https://service.elsevier.com/app/home/supporthub/publishing/) at any time if you have questions about your manuscript, and you can log into Editorial Manager to check the status of your manuscript (https://service.elsevier.com/app/answers/detail/a_id/29155/c/10530/supporthub/publishing/kw/status/).

#AU_ECOINF#

To ensure this email reaches the intended recipient, please do not delete the above code

________________________________________
In compliance with data protection regulations, you may request that we remove your personal registration details at any time. (Remove my information/details). Please contact the publication office if you have any questions.  
