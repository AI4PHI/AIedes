# Counter revision results

Experiment `b8109027c7955b5ce41b5c6ece91baa6fb4ffe640426f059f8f8a9bd488a707c`. Scientific fixed-budget run.

All configurations and fits are retained. Configuration selection uses mean test-fold R².
Tables average seeds within each fold and report mean ± SE across the five folds.
Training-seed variability is reported separately. Bootstrap intervals resample whole traps.

- AIMSurv: 5129 out-of-fold observations, 387 traps.
- Main seed-ensemble OOF R²: 0.5657132441491273; log R²: 0.6796379173112566.
- Recursive matched subset: 5036 observations; R²: 0.4562149872621315.
- VectorNet: 280 observations, 40 traps; observed-lag R²: -0.004324748848650195.

`top_five_wmsle` and `best_per_loss` report seed-replicated results for configurations
ranked using the initial seed; they are not reranked using the replication scores.
`lag_order_sensitivity` fixes the best initial two-lag climate WMSLE configuration and
changes only the lag order (one versus two); the two-lag reference fits are reused. `early_stopping_summary` uses identical 3/1/1 groups
for all three checkpoints: final fixed-budget, best raw validation MSE, and log-MSE
checkpoint retained by a patience-200 early-stopping rule. The training loop
continues to the fixed horizon to obtain its paired comparator; test rows never select
these checkpoints. This supplementary 60/20/20 comparison is separate from the main 80/20 fits.

ZINB is applied to nonnegative weekly rates, which can be fractional after effort
standardization. Its gamma-extended objective is a count-inspired comparison, not an
exact discrete likelihood for these rates. Only exact zeros enter its zero-inflation branch.

Undefined metrics in constant or single-class subsets are left missing. VectorNet
metrics are reported separately from AIMSurv.

The five full-data deployment packages are in `../deploy/`; their training scores are
not validation evidence. Europe-wide map generation remains deferred for climate data.
