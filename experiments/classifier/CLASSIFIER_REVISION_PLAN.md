# Classifier revision plan

## Objective

Produce one reproducible classifier workflow whose saved outputs support the
random-split comparison in the main manuscript and the NUTS3-grouped spatial
validation in the Supplementary Information.

## Analysis design

1. Retain the completed architecture and input-variable selection performed
   within the training portion of the fixed 80/20 random split.
2. Keep the selected input set (monthly temperature and precipitation) and
   selected architecture (`[50,50,50]`) fixed for subsequent evaluation.
3. For the full-coverage random comparison, use five stratified folds. In each
   fold train both the no-hidden-layer logistic-equivalent model and the
   selected neural network from five random seeds, then average the five
   predictions for every held-out observation.
4. Repeat the same two-model, five-seed comparison with five stratified grouped
   folds defined by complete NUTS3 regions. Every region must occur in the
   held-out portion exactly once and must never occur in both partitions of a
   fold.
5. Fit feature scaling using training observations only in every fit.
6. Save fold assignments, per-seed metrics, ensemble metrics, pooled metrics,
   and row-level ensemble predictions before generating figures.

The NUTS3 analysis evaluates the already selected models under geographic
separation. It is not a second architecture-selection exercise.

## Code and output changes

- Extend `run_paper_experiments.py` so both comparison modes produce the same
  result schema for logistic and AIedes models.
- Add per-fit checkpoints so interrupted random or NUTS3 stages can resume.
- Include coordinates in NUTS3 prediction tables.
- Keep all paper-figure scripts in `experiments/classifier/img_for_paper/`.
- Generate the main random-split map, the SI NUTS3 map, and the classifier
  model-selection figure from saved CSV files.
- Document commands, protocols, and result files in a classifier README.
- Add automated checks for training-only scaling, metrics, and grouped-fold
  separation.

## Manuscript updates after verification

- Report the final masked target composition (16,164 positive cells among
  45,724 cells).
- Make the learning rate, selected architecture, and validation terminology
  agree with the code.
- Use the verified random and NUTS3 metrics consistently in the text, captions,
  SI, and response to reviewers.
- Replace the archived SI NUTS3 figure with the newly generated figure.

## Version-control policy

Commit source code, documentation, compact metrics, fold manifests,
paper-facing prediction tables, and final figures. Exclude prepared input
copies, model checkpoints, temporary per-fit checkpoints, interrupted-run
files, timestamped result backups, Python caches, and LaTeX build files.

Keep the archived WorldClim analysis until it has its own reproducible
replacement. Remove the superseded Copernicus/NUTS3 notebooks and pickle
results only after the revised pipeline and PDFs have been verified.

## Proposed commit sequence

1. **Classifier utilities and tests**: changes under `src/AIedes/`, dependency
   declarations, and `tests/test_classifier_revision.py`.
2. **Reproducible classifier workflow**: data preparation, experiment launcher,
   figure scripts, `.gitignore`, and classifier documentation.
3. **Verified classifier outputs**: compact metrics, split manifests,
   paper-facing prediction tables, and the three final classifier figures.
4. **Paper and reviewer response**: `main.tex`, `SI.tex`, compiled PDFs, and the
   corresponding response-to-reviewers edits.
5. **Legacy cleanup (separate commit)**: remove only the superseded
   Copernicus/NUTS3 notebooks and pickle outputs after reviewing the deletion
   list; retain the WorldClim material and `estimated_risk` analysis.
