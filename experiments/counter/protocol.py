"""Fixed-budget, five-fold Counter experiment agreed in PLAN.md."""
import argparse
import copy
import fcntl
import itertools
from importlib.metadata import version
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import pandas as pd
import torch

from AIedes.data_loader.counter_prepare import prepare_counter_data
from AIedes.data_loader.counter_data_loader import fit_counter_preprocessing, transform_counter_features
from AIedes.evaluation.counter_eval import compute_metrics_from_arrays as scores
from AIedes.evaluation.counter_reconstruction import load_model, observed_prediction
from AIedes.models.counter_models import counter_candidate_model, predict_counter_rates
from AIedes.train.counter_train import fit_counter_candidate
from AIedes.train import counter_jobs
from AIedes.utils.counter_losses import FrequencyWeightedLoss, ZINBLoss
from AIedes.utils.counter_experiment import (FIELDS, complete, digest, identity, mark_complete,
                                           read_json, source_hashes, write_json)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGES = ('all', 'prepare', 'paper', 'windows', 'select', 'replicate', 'supplement',
          'deploy', 'evaluate', 'figures')


def load_protocol(path):
    config = read_json(path)
    required = dict(schema_version=2, protocol='fixed_80_20', windows=[30, 60, 90],
        architectures=[[32], [64], [32, 16], [64, 32]], dropouts=[0., .2, .4],
        max_epochs=3000, scheduler_horizon=3000, scheduler_end_factor=.9, batch_size=64,
        final_seeds=[1, 2, 3, 4, 5], screen_seed=1, weight_bins=[1.],
        autonomous_lag_days=[7, 14], climate_endpoint='end_date',
        autonomous_startup='missing_flags_at_first_observation',
        deployment_choice='full_data_five_seeds', patience=200,
        loss_learning_rates={'WMSE': .0001, 'WMSLE': .001, 'ZINB': .0005})
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'{key} must be {value!r} for the agreed protocol')
    for key in ('bootstrap_samples', 'log_interval', 'plot_min_observations'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('weekly_data', 'full_data', 'split_manifest', 'external_data'):
        config[key] = str((Path(path).resolve().parent / config[key]).resolve())
    return config


def with_id(candidate):
    candidate = {k: v for k, v in candidate.items() if k != 'id'}
    return dict(candidate, id=identity(candidate)[:12])


def grid(config):
    candidates = []
    for loss, lr in config['loss_learning_rates'].items():
        windows = config['windows'] if loss == 'WMSLE' else [90]
        inputs = [dict(fields=fields, weekly=weekly, lags=lags, window=window)
                  for fields, weekly, lags, window in itertools.product(
                      [FIELDS, ['t2m_mean', 'tp_sum']], [False, True], [False, True], windows)]
        inputs += [dict(fields=[], weekly=False, lags=True, window=None)]
        for inp, widths, dropout in itertools.product(inputs, config['architectures'], config['dropouts']):
            candidates.append(with_id(dict(inp, widths=widths, dropout=dropout, loss=loss,
                                           learning_rate=lr)))
    assert len(candidates) == len({c['id'] for c in candidates}) == 516
    return sorted(candidates, key=lambda c: (c['window'] not in (None, 90), c['loss'], c['id']))


def metadata(candidate):
    fields = [f"{f}_znorm{'(mean)' if candidate['weekly'] else ''}" for f in candidate['fields']]
    order = candidate.get('lag_order', 2) if candidate['lags'] else 0
    fields += [f'p{i}zn' for i in range(1, order + 1)]
    return dict(config_id=candidate['id'], loss=candidate['loss'], window=candidate['window'],
                input_variables='; '.join(fields), input_family='+'.join(candidate['fields']) or 'lag-only',
                weekly=candidate['weekly'], lag_order=order, num_layers=len(candidate['widths']) + 1,
                hidden_dim=candidate['widths'][0], dropout=candidate['dropout'])


def make_job(output, candidate, fold, seed, config, experiment_id, kind='grid'):
    if kind == 'replicate':
        folder = output / 'final_models' / f'outer{fold}' / f"{candidate['id']}_seed{seed}"
    else:
        folder = output / kind / f"{candidate['id']}_fold{fold}_seed{seed}"
    return dict(folder=str(folder), candidate=candidate, outer=fold, seed=seed,
                config=config, experiment_id=experiment_id, kind=kind)


def partitions(frame, job):
    if job['kind'] == 'deploy':
        return np.arange(len(frame)), np.array([], dtype=int), None
    test = np.flatnonzero(frame.outer_group.eq(job['outer']))
    train = np.flatnonzero(frame.outer_group.ne(job['outer']))
    val = None
    if job['kind'] == 'sensitivity':
        validation = job['outer'] % 5 + 1
        val = np.flatnonzero(frame.outer_group.eq(validation))
        train = np.flatnonzero(~frame.outer_group.isin([job['outer'], validation]))
    if not len(train) or not len(test):
        raise ValueError('Empty training/test split')
    groups = [set(frame.iloc[rows].id_trap) for rows in (train, test) + ((val,) if val is not None else ())]
    if any(a & b for a, b in itertools.combinations(groups, 2)):
        raise ValueError('Traps cross a split boundary')
    return train, test, val


def execute_job(job, frame=None):
    started = time.monotonic()
    folder = Path(job['folder'])
    if complete(folder, job['experiment_id']):
        return str(folder), 'cached', time.monotonic() - started
    frame = counter_jobs._FRAME if frame is None else frame
    if frame is None:
        raise RuntimeError('Worker observations were not initialized')
    train, test, val = partitions(frame, job)
    config, candidate = job['config'], job['candidate']
    folder.mkdir(parents=True, exist_ok=True)
    monitored = {}
    callback = None
    if val is not None:
        stats = fit_counter_preprocessing(frame, train, candidate)
        vx = transform_counter_features(frame.iloc[val], candidate, stats)
        vy = frame.iloc[val].weeklyRates.to_numpy(float)
        monitored = {name: dict(value=float('inf'), epoch=0, state=None, stale=0, stopped=False)
                     for name in ('best_raw_validation', 'early_log_validation')}
        def callback(model, epoch):
            p = predict_counter_rates(model, vx)
            values = dict(best_raw_validation=float(np.mean((vy-p)**2)),
                          early_log_validation=float(np.mean((np.log1p(vy)-np.log1p(p))**2)))
            for name, result in monitored.items():
                if result['stopped']:
                    continue
                if values[name] < result['value']:
                    result.update(value=values[name], epoch=epoch,
                                  state=copy.deepcopy(model.state_dict()), stale=0)
                else:
                    result['stale'] += 1
                if name == 'early_log_validation' and result['stale'] >= config['patience']:
                    result['stopped'] = True
    package, history, _ = fit_counter_candidate(frame, train, None, candidate, config,
        job['seed'], epochs=config['max_epochs'], epoch_callback=callback)
    package.update(experiment_id=job['experiment_id'], outer_group=job['outer'],
                   validation_group=job['outer'] % 5 + 1 if val is not None else None,
                   training_row_ids=frame.iloc[train].row_id.tolist(), protocol='fixed_80_20')
    model = counter_candidate_model(candidate, package['input_dim'])
    model.load_state_dict(package['state_dict'])
    def objective(rows):
        x = transform_counter_features(frame.iloc[rows], candidate, package['stats'])
        y = torch.as_tensor(frame.iloc[rows].weeklyRates.to_numpy(), dtype=torch.float32)
        if candidate['loss'] == 'ZINB':
            criterion = ZINBLoss(zero_threshold=0.)
            criterion.load_state_dict(package['criterion_state_dict'])
        else:
            criterion = FrequencyWeightedLoss(config['weight_bins'], package['loss_weights'],
                'cpu', log_loss=candidate['loss'] == 'WMSLE', right=True)
        model.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32)
            prediction = model.forward_training(tensor) if candidate['loss'] == 'ZINB' else model(tensor)
            return float(criterion(prediction, y))
    train_prediction = observed_prediction(model, package, frame.iloc[train])
    result = dict(metadata(candidate), fold=job['outer'], seed=job['seed'], kind=job['kind'],
                  n_train=len(train), epochs=package['epochs_trained'],
                  train_loss=objective(train),
                  parameters=sum(p.numel() for p in model.parameters()),
                  **{f'train_{k}': v for k, v in scores(frame.iloc[train].weeklyRates, train_prediction).items()})
    files = ['model.pt', 'history.csv', 'job.json', 'result.json']
    if len(test):
        p = observed_prediction(model, package, frame.iloc[test])
        result.update({f'test_{k}': v for k, v in scores(frame.iloc[test].weeklyRates, p).items()})
        result['test_loss'] = objective(test)
        prediction = frame.iloc[test][['row_id', 'id_trap', 'country', 'end_date']].copy()
        prediction['target'], prediction['prediction'] = frame.iloc[test].weeklyRates.to_numpy(), p
        prediction.to_csv(folder / 'test.csv', index=False)
        files.append('test.csv')
    if monitored:
        records = [dict(checkpoint='fixed_final', epoch=config['max_epochs'],
                        **scores(frame.iloc[test].weeklyRates, p))]
        for name, state in monitored.items():
            model.load_state_dict(state['state'])
            p = observed_prediction(model, package, frame.iloc[test])
            records.append(dict(checkpoint=name, epoch=state['epoch'], **scores(frame.iloc[test].weeklyRates, p)))
            prediction[name] = p
            selected = dict(package, state_dict=state['state'], best_epoch=state['epoch'])
            torch.save(selected, folder / f'{name}.pt')
            files.append(f'{name}.pt')
        pd.DataFrame(records).to_csv(folder / 'sensitivity.csv', index=False)
        prediction.to_csv(folder / 'test.csv', index=False)
        files.append('sensitivity.csv')
    temporary = folder / f'model.{os.getpid()}.tmp'
    torch.save(package, temporary)
    os.replace(temporary, folder / 'model.pt')
    pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
    write_json(folder / 'job.json', job)
    result['seconds'] = time.monotonic() - started
    write_json(folder / 'result.json', result)
    mark_complete(folder, job['experiment_id'], files)
    return str(folder), 'trained', time.monotonic() - started


def grid_jobs(output, candidates, config, eid):
    return [make_job(output, c, fold, config['screen_seed'], config, eid)
            for c in candidates for fold in range(1, 6)]


def summary(records):
    frame = pd.DataFrame(records)
    metrics = [c for c in frame if c.startswith(('train_', 'test_'))]
    # Average training seeds within each fold first; folds are not independent repetitions.
    folded = frame.groupby(['config_id', 'fold'])[metrics].mean()
    avg = folded.groupby('config_id').mean().add_suffix('_mean')
    se = folded.groupby('config_id').sem().add_suffix('_se')
    info = frame.drop_duplicates('config_id').set_index('config_id')
    cols = ['loss', 'window', 'input_variables', 'input_family', 'weekly', 'lag_order',
            'num_layers', 'hidden_dim', 'dropout', 'parameters']
    return info[cols].join(avg).join(se).reset_index()


def select(output, candidates, config, eid):
    folder = output / 'selection'
    if complete(folder, eid):
        return read_json(folder / 'selected.json')
    records = []
    for job in grid_jobs(output, candidates, config, eid):
        if not complete(job['folder'], eid):
            raise ValueError(f"Incomplete grid job: {job['folder']}")
        records.append(read_json(Path(job['folder']) / 'result.json'))
    folder.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(output / 'runs.csv', index=False)
    table = summary(records).sort_values(['test_r2_mean', 'config_id'], ascending=[False, True])
    table.to_csv(folder / 'configuration_summary.csv', index=False)
    lookup = {c['id']: c for c in candidates}
    wmsle = table.loc[table.loss.eq('WMSLE')]
    top = wmsle.head(5).config_id.tolist()
    main = top[0]
    climate = wmsle.loc[wmsle.lag_order.eq(0)].iloc[0].config_id
    lag_anchor = wmsle.loc[wmsle.lag_order.eq(2) & wmsle.input_family.ne('lag-only')].iloc[0].config_id
    loss_best = table.groupby('loss', sort=False).head(1).config_id.tolist()
    reported = list(dict.fromkeys(top + loss_best + [climate, lag_anchor]))
    selected = dict(main=lookup[main], climate_only=lookup[climate],
                    top_five_wmsle=[lookup[i] for i in top], best_per_loss=[lookup[i] for i in loss_best],
                    lag_anchor=lookup[lag_anchor], reported=[lookup[i] for i in reported],
                    selection='highest mean five-fold test R2; ties by config ID')
    write_json(folder / 'selected.json', selected)
    winners = {str(fold): {role: dict(candidate=selected[role], epochs=config['max_epochs'])
                           for role in ('main', 'climate_only')} for fold in range(1, 6)}
    write_json(folder / 'winners.json', winners)
    pd.DataFrame(records).to_csv(folder / 'grid_runs.csv', index=False)
    mark_complete(folder, eid, ['configuration_summary.csv', 'selected.json', 'winners.json', 'grid_runs.csv'])
    return selected


def replicate_jobs(output, selected, config, eid):
    jobs = []
    for c, fold, seed in itertools.product(selected['reported'], range(1, 6), config['final_seeds']):
        job = make_job(output, c, fold, seed, config, eid, 'replicate')
        if seed == config['screen_seed'] and not complete(job['folder'], eid):
            source = Path(make_job(output, c, fold, seed, config, eid)['folder'])
            if not complete(source, eid):
                raise ValueError('Cannot reuse incomplete screening-seed fit')
            destination = Path(job['folder'])
            destination.mkdir(parents=True, exist_ok=True)
            # Publish the completion marker last so an interrupted copy resumes safely.
            marker = read_json(source / 'done.json')
            for name in marker['artifacts']:
                shutil.copy2(source / name, destination / name)
            mark_complete(destination, eid, list(marker['artifacts']))
        jobs.append(job)
    return jobs


def supplementary_jobs(output, selected, config, eid):
    # The two-lag reference is already among the replicated configurations.
    anchor = selected['lag_anchor']
    jobs = []
    candidate = with_id(dict(anchor, lag_order=1))
    for fold, seed in itertools.product(range(1, 6), config['final_seeds']):
        jobs.append(make_job(output, candidate, fold, seed, config, eid, 'lag_order'))
    for seed in config['final_seeds']:
        for fold in range(1, 6):
            jobs.append(make_job(output, selected['main'], fold, seed, config, eid, 'sensitivity'))
    return jobs


def sources():
    paths = source_hashes()
    for p in list(HERE.glob('*.py')) + list((HERE / 'figures').glob('*.py')) + list((HERE / 'tests').glob('*.py')):
        paths[str(p.relative_to(ROOT))] = digest(p)
    return paths


def validate_external(config):
    """Fail before the long training run if the independent input is unusable."""
    external = pd.read_pickle(config['external_data'])
    required = ['id_trap', 'countryCode', 'end_date', 'weeklyRates', 'prev1_rates', 'prev2_rates'] + FIELDS
    missing = set(required) - set(external.columns)
    if missing or external.empty:
        raise ValueError(f'Invalid external dataset; missing columns: {sorted(missing)}')
    if external.id_trap.isna().any() or external.duplicated(['id_trap', 'end_date']).any():
        raise ValueError('External trap/date identifiers are missing or duplicated')
    for field in FIELDS:
        values = np.stack(external[field].to_numpy()).astype(float)
        if values.shape != (len(external), 90) or not np.isfinite(values).all():
            raise ValueError(f'Invalid external climate arrays: {field}')
    y = external.weeklyRates.to_numpy(float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid external targets')
    for lag in ('prev1_rates', 'prev2_rates'):
        values = external[lag].dropna().to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'Invalid external lag values: {lag}')
    dates = pd.to_datetime(external.end_date, errors='raise')
    if dates.isna().any():
        raise ValueError('Missing external dates')
    print(f'External input verified: {len(external)} observations, {external.id_trap.nunique()} traps, '
          f'{dates.min().date()} to {dates.max().date()}', flush=True)


def main():
    from run import run_jobs
    from reporting import evaluate_and_report
    from figures.make_figures import make_figures
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--stage', choices=STAGES, default='all')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    torch.set_num_threads(1)
    config = load_protocol(args.config)
    candidates = grid(config)
    output = (args.output or HERE / ('smoke_results' if args.smoke else 'results')).resolve()
    if args.smoke:
        config.update(max_epochs=2, final_seeds=[1, 2], bootstrap_samples=10, patience=1)
        # Every loss, lag pathway and climate window; all five held-out groups.
        candidates = [c for c in candidates if c['widths'] == [32] and c['dropout'] == 0
                      and (c['fields'] == FIELDS or not c['fields'])
                      and (not c['weekly'] or (c['loss'] == 'WMSLE' and c['window'] == 60 and c['lags']))]
    print(f"{'SYNTHETIC SMOKE' if args.smoke else 'SCIENTIFIC EXPERIMENT'}: {len(candidates)} configurations, "
          f"{len(candidates)*5} fixed-budget trap-blocked fits; {config['max_epochs']} epochs", flush=True)
    print(f'Output: {output}; workers: {args.workers}; stage: {args.stage}', flush=True)
    if args.dry_run:
        print('Production plan: 1,620 paper fits + 960 window fits; reuse seed 1 for replication.')
        print('Then 140–180 extra seed fits (7–9 selected configurations), 25 one-lag fits,')
        print('25 checkpoint-sensitivity fits, 5 full-data deployment fits, evaluation and figures.')
        for key in ('weekly_data', 'full_data', 'split_manifest', 'external_data'):
            if not Path(config[key]).is_file():
                raise FileNotFoundError(config[key])
        print('All four input files exist. Dry-run only: no output or training created.')
        return
    if args.status:
        for kind in ('grid', 'final_models', 'lag_order', 'sensitivity', 'deploy', 'reports', 'figures'):
            print(kind, len(list((output / kind).rglob('done.json'))), 'completion markers (not verified)')
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        provenance = dict(config=config, code=sources(), smoke_only=args.smoke,
            inputs={} if args.smoke else {k: digest(config[k]) for k in
                ('weekly_data', 'full_data', 'split_manifest', 'external_data')},
            environment=dict(python=platform.python_version(), packages={name: version(name) for name in
                ('numpy', 'pandas', 'torch', 'scipy', 'scikit-learn', 'matplotlib', 'pycountry', 'jinja2')}))
        eid = identity(provenance)
        experiment = dict(provenance, experiment_id=eid)
        if (output / 'experiment.json').exists():
            if read_json(output / 'experiment.json') != experiment:
                raise ValueError('Code/config/data/environment changed; use a new --output directory')
        else:
            write_json(output / 'experiment.json', experiment)
            for relative in provenance['code']:
                dest = output / 'source_snapshot' / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, dest)
            write_json(output / 'grid.json', candidates)
        if not args.smoke:
            validate_external(config)
        prepare_counter_data(output, config, eid, args.smoke)
        if args.stage == 'prepare':
            return
        if args.stage in ('all', 'paper', 'windows'):
            chosen = candidates if args.stage == 'all' else [c for c in candidates
                if (c['window'] in (None, 90)) == (args.stage == 'paper')]
            run_jobs(grid_jobs(output, chosen, config, eid), output, args.workers, job_function=execute_job)
            if args.stage != 'all':
                return
        selected = select(output, candidates, config, eid)
        if args.stage == 'select':
            return
        if args.stage in ('all', 'replicate'):
            run_jobs(replicate_jobs(output, selected, config, eid), output, args.workers, job_function=execute_job)
            if args.stage == 'replicate':
                return
        if args.stage in ('all', 'supplement'):
            run_jobs(supplementary_jobs(output, selected, config, eid), output, args.workers, job_function=execute_job)
            if args.stage == 'supplement':
                return
        if args.stage in ('all', 'deploy'):
            jobs = [make_job(output, selected['main'], 0, s, config, eid, 'deploy') for s in config['final_seeds']]
            run_jobs(jobs, output, args.workers, job_function=execute_job)
            if args.stage == 'deploy':
                return
        if args.stage in ('all', 'evaluate'):
            evaluate_and_report(output, config, eid, selected, smoke=args.smoke)
            if args.stage == 'evaluate':
                return
        if args.stage in ('all', 'figures'):
            make_figures(output, eid)
        print('All requested non-map stages complete.', flush=True)
