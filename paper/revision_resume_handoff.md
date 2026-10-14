# Manuscript revision handoff

Updated: 9 October 2026

This document records the agreed scientific direction, editing policy, completed
work, and next actions so that the revision can be resumed in another thread. The
more detailed figure-and-table plan is in
`counter_revision_figures_tables_plan.md`.

## Scientific direction

The paper presents a quantitative framework for predicting ovitrap-derived weekly
egg-laying rates of *Aedes albopictus*. Its main contribution is a reproducible
dataset, explicit performance measures, and a reference modelling procedure that
allows future approaches, including statistical, machine-learning, and mechanistic
models, to be compared quantitatively.

The manuscript should focus on measured predictive performance. European
future-climate forecasting, operational deployment, and disease-risk prediction are
potential later applications rather than objectives of this paper. Egg-laying rate
can be discussed as a proxy for mosquito activity while stating that it is not a
direct estimate of adult abundance or disease risk. The same quantitative procedure
could be applied to adult-trap data when sufficiently extensive datasets become
available.

The revised Counter evaluation uses complete traps as groups. Five-fold grouped
cross-validation ensures that each evaluated observation belongs to a trap excluded
from training. The final selected configurations are trained with five independent
random initializations in each fold, giving 25 retained models. Screening results use
one seed per configuration and fold; every completed run is retained. Preprocessing
statistics are estimated from the training traps in each fold.

Observed-history prediction and autonomous forecasting must remain clearly
distinguished. The former uses measured previous egg-laying values. The latter uses
earlier model predictions recursively and is trained specifically for this use. The
dedicated autoregressive model is the relevant model for autonomous projections.

VectorNet is a separate external stress test and remains outside the principal
AIMSurv performance estimates. Its limited performance can be discussed in relation
to the small sample, protocol differences, and the mountainous valley setting, where
local microclimates may be poorly represented by gridded climate inputs. The
before-September/from-September split is a post hoc diagnostic motivated by the
missed second seasonal peak. One VectorNet scatter figure is planned for the SI.

## Agreed editing and implementation policy

1. Work through one figure or table at a time.
2. First propose its purpose, content, metrics, and layout; implement it only after
   approval.
3. After approval, update the generating code and figure/table, make the smallest
   consistent manuscript edits, and provide a short change summary.
4. Rebuild both `main.pdf` and `SI.pdf` after every LaTeX edit.
5. Inspect figures at their actual manuscript-page scale, with readable fonts and
   publication-quality vector output.
6. Keep the main paper focused by replacing existing figures and tables where
   possible. Put complete sensitivity analyses in the SI. Avoid adding unnecessary
   main-text figures or tables.
7. Scientific prose describes the final method directly. It should not read as a
   history of former analyses or revision rounds. Avoid negative constructions about
   procedures that are no longer part of the paper.
8. Use plain technical language. Define a standard procedure before explaining its
   implementation; avoid unexplained abbreviations such as OOF.
9. Reviewer responses must contain each editor/reviewer comment in its correct
   position and then state the corresponding final manuscript change concisely.
10. Preserve the frozen reference files `main_old.tex` and `SI_old.tex` unchanged.

## Agreed structure

The main paper retains its current number of central Counter figures and tables:

- Table 1: complete input-variable definitions.
- Figure 3: focused 90-day WMSLE input/architecture sensitivity.
- Table 2: matched-cohort comparison of observed-history and forecasting modes.
- Figure 5: held-out-trap observed-versus-predicted scatter on raw and log scales.
- Figure 6: observed and predicted time series for six long AIMSurv trap series.
- Figure 7: European output from the dedicated autoregressive ensemble.

The SI will contain the complete loss comparison, 30/60/90-day climate-window
sensitivity, leading configurations, lag comparison, autoregressive architecture
check, training variability, and VectorNet external evaluation. Detailed diagnostic
maps and exploratory VectorNet analyses remain in the experiment folders.

## Work completed in the current pass

### Main Table 1

The table and surrounding notation now describe:

- daily climate histories as `30d`, `60d`, and `90d`;
- weekly histories as `5w`, `9w`, and `13w`;
- climate variables with the common symbols `AT_W`, `TP_W`, and `DP_W`;
- measured one- and two-week egg-laying inputs as `ER_{1w}` and `ER_{2w}`;
- their availability indicators.

The matching notation was updated in the main paper and SI where required.

### Main Figure 3

Figure 3 was regenerated from the trap-grouped screening results. It now contains
three horizontal panels for dropout 0.0, 0.2, and 0.4; four architectures (`[32]`,
`[64]`, `[32,16]`, and `[64,32]`); WMSLE results; and 90-day histories represented
as either 90 daily values or 13 weekly averages. Error bars show the standard error
across five trap folds. The selected observed-history configuration is marked by a
star.

The x axis uses compact A--D codes, grouped under `Weekly (13w)` and `Daily (90d)`.
Their definitions are in the caption. The shared y axis is labelled simply `R²`, has
an upper limit of 0.60, and uses enlarged publication-scale fonts. The accepted
figure is a vector PDF.

Relevant files:

- plotting code: `../experiments/counter/img_for_paper/make_ablation.py`;
- input summary: `../experiments/counter/results/selection/configuration_summary.csv`;
- selected configuration: `../experiments/counter/results/selection/selected.json`;
- manuscript figure: `img/nn_results_by_dropout.pdf`;
- preview: `img/nn_results_by_dropout.png`;
- main manuscript inclusion and caption: `main.tex`, around Figure 3.

Direct regeneration command, run from the repository root:

```bash
python -m experiments.counter.img_for_paper.make_ablation \
  --output experiments/counter/results \
  --destination paper/img
```

During iterative editing, figures may first be rendered under `/tmp` for inspection
and copied to `paper/img` only after acceptance.

### Main Table 2

The former top-five table was replaced with the matched-cohort comparison of four
prediction modes on 5,036 observations from 375 traps. The table reports original-
scale R², log-R², RMSE, and zero/non-zero accuracy. Its compact layout uses input
codes A and B. Input set A is shared by the first three modes and contains `90d`
daily climate plus the one- and two-week egg-laying inputs. The observed-history
network is `[64]` with dropout 0.2; the dedicated autoregressive model uses the same
input structure with recursively generated values and the slightly different
`[64,32]` architecture. The focused architecture comparison is assigned to the SI.
Input set B is the best-performing WMSLE climate-only configuration by mean
held-out-trap R²: `13w` weekly climate, `[64,32]`, dropout 0.0.

The text explains why the dedicated autoregressive model is used for autonomous
projection: it is trained under the recursive conditions used during projection and
improves original-scale R² and RMSE relative to the climate-only model. The former
top-five comparison will later be represented by an updated SI table.

Principal metric sources are
`../experiments/counter/results/reports/performance_cohorts.csv` and
`../experiments/counter/results/autoregressive_jobs/comparison_64_32.json`.

### Main Figure 5

The scatter retains the original four-panel visual design: panels (a) and (b)
show in-sample and held-out-trap predictions on the original scale, and panels
(c) and (d) show the corresponding `log(1+x)` results. Held-out points are the
mean of the five seed models trained without the corresponding trap. All panels
show fitted regressions and one-to-one reference lines. Fonts, ticks, and legends
were enlarged for final manuscript-page readability; repeated vertical labels were
removed from panels (b) and (d) to retain clear spacing.

The figure uses the complete observed-history cohort of 5,129 observations from 387
traps and reports held-out R² = 0.566, held-out log-R² = 0.680, and held-out RMSE
= 26.12 (with corresponding in-sample R² = 0.712 and log-R² = 0.850). This is
distinct from Table 2's 5,036-row matched cohort, which is restricted to
observations usable by all four prediction modes. The Results text explains that the log scale
complements the original-scale assessment for right-skewed counts whose absolute
fluctuations increase with magnitude, limiting the influence of occasional high
values and showing proportional agreement across low and intermediate rates.

Relevant files are:

- plotting code: `../experiments/counter/img_for_paper/make_scatter.py`;
- frozen predictions: `../experiments/counter/results/evaluation/main_predictions.csv`;
- manuscript figure: `img/R2_scatter_plot.pdf`;
- preview: `img/R2_scatter_plot.png`.

Both `main.pdf` and `SI.pdf` currently compile successfully.

## Next action

Proceed to Main Figure 6, the time series for six long AIMSurv trap records. First
propose the trap-selection rule, panel layout, observation and prediction styling,
and display of variability across the five held-out seed models. After approval,
update the generating code and figure, make the minimum corresponding manuscript
edits, rebuild both PDFs, and provide a short change summary.
