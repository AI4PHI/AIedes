"""Paper tables, external validation, and reviewer comparisons for the fixed protocol."""
from pathlib import Path

import numpy as np
import pandas as pd

from AIedes.data_loader.counter_prepare import synthetic_counter_data
from AIedes.evaluation.counter_eval import compute_metrics_from_arrays as scores
from AIedes.evaluation.counter_reconstruction import load_model, observed_prediction
from AIedes.evaluation.counter_reporting import evaluate, bootstrap
from AIedes.utils.counter_experiment import complete, mark_complete, read_json, write_json
from protocol import make_job, summary, supplementary_jobs
from figures.make_tables import table_export

METRICS = ['r2', 'log_r2', 'rmse', 'log_mse', 'accuracy', 'sensitivity', 'specificity', 'slope']


def read_jobs(jobs, eid):
    records = []
    for job in jobs:
        if not complete(job['folder'], eid):
            raise ValueError(f"Incomplete required job: {job['folder']}")
        records.append(read_json(Path(job['folder']) / 'result.json'))
    return records


def external_validation(output, config, selected, eid, smoke):
    if smoke:
        external, _ = synthetic_counter_data()
        external['id_trap'] += 1000
        external['end_date'] += pd.Timedelta(days=365)
    else:
        external = pd.read_pickle(config['external_data']).copy()
        external['country'] = external['countryCode']
    external = external.reset_index(drop=True)
    table = external[['id_trap', 'country', 'end_date']].rename(columns={'end_date': 'date'})
    table['row_id'] = np.arange(len(table))
    table['target'] = external.weeklyRates.to_numpy(float)
    for role in ('main', 'climate_only'):
        predictions = []
        for fold in range(1, 6):
            for seed in config['final_seeds']:
                job = make_job(output, selected[role], fold, seed, config, eid, 'replicate')
                if not complete(job['folder'], eid):
                    raise ValueError('External validation requires completed fold ensembles')
                model, package = load_model(Path(job['folder']) / 'model.pt')
                values = observed_prediction(model, package, external)
                table[f'{role}_fold{fold}_seed{seed}'] = values
                predictions.append(values)
        table[role] = np.mean(predictions, axis=0)
    table.to_csv(output / 'reports/vectornet_predictions.csv', index=False)
    report = dict(observations=len(table), traps=int(table.id_trap.nunique()),
                  first_date=str(table.date.min()), last_date=str(table.date.max()),
                  mode='observed-lag hindcast; independent 2021 data; no tuning on VectorNet',
                  metrics={role: scores(table.target, table[role]) for role in ('main', 'climate_only')},
                  uncertainty=bootstrap(table, ['main', 'climate_only'], config['bootstrap_samples'], config['bootstrap_seed']))
    write_json(output / 'reports/vectornet_metrics.json', report)
    return table, report


def evaluate_and_report(output, config, eid, selected, smoke=False):
    output = Path(output)
    destination = output / 'reports'
    if complete(destination, eid):
        return
    destination.mkdir(exist_ok=True)
    evaluate(output, config, eid)
    main = pd.read_csv(output / 'evaluation/main_predictions.csv', parse_dates=['date'])
    rec = pd.read_csv(output / 'evaluation/reconstruction_predictions.csv', parse_dates=['date'])
    replicated = [make_job(output, c, f, s, config, eid, 'replicate')
                  for c in selected['reported'] for f in range(1, 6) for s in config['final_seeds']]
    records = read_jobs(replicated, eid)
    runs = pd.DataFrame(records)
    runs.to_csv(destination / 'reported_runs.csv', index=False)
    table = summary(records)
    table['n_folds'] = 5
    table['seeds_per_fold'] = len(config['final_seeds'])
    table_export(table, destination / 'selected_configurations')
    for name, chosen in [('top_five_wmsle', selected['top_five_wmsle']), ('best_per_loss', selected['best_per_loss'])]:
        ids = [c['id'] for c in chosen]
        table_export(table.set_index('config_id').loc[ids].reset_index(), destination / name)
    # Training-seed variability conditional on each fixed fold, kept distinct from fold variability.
    variability = runs.groupby(['config_id', 'fold'])[[f'test_{m}' for m in METRICS]].agg(['mean', 'std'])
    variability.columns = ['_'.join(c) for c in variability.columns]
    variability.reset_index().to_csv(destination / 'training_seed_variability.csv', index=False)
    grid_runs = pd.read_csv(output / 'selection/grid_runs.csv')
    additional = runs.loc[runs.seed.ne(config['screen_seed'])]
    pd.concat([grid_runs, additional], ignore_index=True).to_csv(destination / 'all_runs.csv', index=False)
    supplemental = supplementary_jobs(output, selected, config, eid)
    extra = read_jobs(supplemental, eid)
    pd.DataFrame(extra).to_csv(destination / 'supplementary_runs.csv', index=False)
    anchor_runs = [r for r in records if r['config_id'] == selected['lag_anchor']['id']]
    one_lag_runs = [r for r in extra if r['kind'] == 'lag_order']
    lag = summary(anchor_runs + one_lag_runs)
    table_export(lag.sort_values('lag_order'), destination / 'lag_order_sensitivity')
    paired = pd.DataFrame(one_lag_runs).merge(pd.DataFrame(anchor_runs), on=['fold', 'seed'],
        suffixes=('_one_lag', '_two_lags'), validate='one_to_one')
    differences = paired[['fold', 'seed']].copy()
    for metric in METRICS:
        differences[metric + '_one_minus_two'] = paired[f'test_{metric}_one_lag'] - paired[f'test_{metric}_two_lags']
    table_export(differences, destination / 'lag_order_paired_differences')
    sensitivities = []
    for job in supplemental:
        if job['kind'] == 'sensitivity':
            data = pd.read_csv(Path(job['folder']) / 'sensitivity.csv')
            data['fold'], data['seed'] = job['outer'], job['seed']
            sensitivities.append(data)
    sensitivity = pd.concat(sensitivities)
    sensitivity.to_csv(destination / 'early_stopping_runs.csv', index=False)
    folded = sensitivity.groupby(['checkpoint', 'fold'])[METRICS + ['epoch']].mean()
    table_export(folded.groupby('checkpoint').mean().reset_index(), destination / 'early_stopping_summary')
    cohorts = []
    for name, data, columns in [('AIMSurv_all', main, ['prediction', 'climate_only']),
                               ('AIMSurv_matched_recursive', rec, ['observed_history', 'autonomous', 'climate_only'])]:
        for column in columns:
            cohorts.append(dict(cohort=name, mode=column, observations=len(data), traps=data.id_trap.nunique(),
                                **scores(data.target, data[column])))
    external, external_report = external_validation(output, config, selected, eid, smoke)
    for role in ('main', 'climate_only'):
        cohorts.append(dict(cohort='VectorNet_2021', mode=role, observations=len(external),
                            traps=external.id_trap.nunique(), **scores(external.target, external[role])))
    table_export(pd.DataFrame(cohorts), destination / 'performance_cohorts')
    coverage = dict(total_observations=len(main), total_traps=int(main.id_trap.nunique()),
                    recursive_observations=len(rec), recursive_traps=int(rec.id_trap.nunique()),
                    excluded_observations=len(main)-len(rec),
                    excluded_traps=int(main.id_trap.nunique()-rec.id_trap.nunique()))
    write_json(destination / 'recursive_coverage.json', coverage)
    main_scores = scores(main.target, main.prediction)
    recursive_scores = scores(rec.target, rec.autonomous)
    text = f'''# Counter revision results

Experiment `{eid}`. {'SYNTHETIC SMOKE ONLY — not scientific results.' if smoke else 'Scientific fixed-budget run.'}

All configurations and fits are retained. Configuration selection uses mean test-fold R².
Tables average seeds within each fold and report mean ± SE across the five folds.
Training-seed variability is reported separately. Bootstrap intervals resample whole traps.

- AIMSurv: {len(main)} out-of-fold observations, {main.id_trap.nunique()} traps.
- Main seed-ensemble OOF R²: {main_scores['r2']}; log R²: {main_scores['log_r2']}.
- Recursive matched subset: {len(rec)} observations; R²: {recursive_scores['r2']}.
- VectorNet: {len(external)} observations, {external.id_trap.nunique()} traps; observed-lag R²: {external_report['metrics']['main']['r2']}.

`top_five_wmsle` and `best_per_loss` report seed-replicated results for configurations
ranked using the initial seed; they are not reranked using the replication scores.
`lag_order_sensitivity` fixes the best initial two-lag climate WMSLE configuration and
changes only the lag order (one versus two); the two-lag reference fits are reused. `early_stopping_summary` uses identical 3/1/1 groups
for all three checkpoints: final fixed-budget, best raw validation MSE, and log-MSE
checkpoint retained by a patience-{config['patience']} early-stopping rule. The training loop
continues to the fixed horizon to obtain its paired comparator; test rows never select
these checkpoints. This supplementary 60/20/20 comparison is separate from the main 80/20 fits.

ZINB is applied to nonnegative weekly rates, which can be fractional after effort
standardization. Its gamma-extended objective is a count-inspired comparison, not an
exact discrete likelihood for these rates. Only exact zeros enter its zero-inflation branch.

Undefined metrics in constant or single-class subsets are left missing. VectorNet
metrics are reported separately from AIMSurv.

The five full-data deployment packages are in `../deploy/`; their training scores are
not validation evidence. Europe-wide map generation remains deferred for climate data.
'''
    (destination / 'RESULTS.md').write_text(text)
    files = [str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file() and p.name != 'done.json']
    mark_complete(destination, eid, files)
