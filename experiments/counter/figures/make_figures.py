"""Generate Counter-style figures; regenerate completed runs without retraining."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from AIedes.utils.counter_experiment import complete, mark_complete, write_json, read_json, digest, ROOT
from figures.make_ablation import make_ablation
from figures.make_scatter import make_scatter
from figures.make_timeseries import make_timeseries


def make_figures(output, experiment_id, regenerate=False):
    output = Path(output).resolve()
    experiment = read_json(output / 'experiment.json')
    if experiment['experiment_id'] != experiment_id:
        raise ValueError('Experiment identity differs')
    stages = ('reports', 'evaluation', 'selection', 'data_audit')
    for stage in stages:
        if not complete(output / stage, experiment_id):
            raise ValueError(f'Complete {stage} before figures')
    folder = output / 'figures'
    if folder.exists() and complete(folder, experiment_id) and not regenerate:
        return
    # Plotting may change independently; inference code and observed inputs may not.
    for relative, sha in experiment['code'].items():
        if relative.startswith('src/') and digest(ROOT / relative) != sha:
            raise ValueError(f'Inference source changed: {relative}')
    for key in ('weekly_data', 'external_data'):
        if digest(experiment['config'][key]) != experiment['inputs'][key]:
            raise ValueError(f'Observed input changed: {key}')
    selected = read_json(output / 'selection/selected.json')['main']
    if selected['loss'] != 'WMSLE':
        raise ValueError('Main prediction figures require the selected WMSLE model')
    with tempfile.TemporaryDirectory(prefix='.figures-', dir=output) as temporary:
        staging = Path(temporary)
        for generate in (make_ablation, make_scatter, make_timeseries):
            print(f'Generating {generate.__name__} with model {selected["id"]}', flush=True)
            with plt.rc_context(rc=matplotlib.rcParamsDefault):
                generate(output, staging)
        write_json(staging / 'figure_notes.json', {
            'selected_model': selected,
            'ablation': 'Original dropout columns, architecture colors, input order and selected-model star. Grid mean +/- SE across five folds; no performance filtering. Original axis limits expanded only when needed.',
            'R2_scatter_plot.pdf': 'Original train/test layout. Each training point averages the 20 models trained on that observation (four folds x five seeds). Each test point averages five held-out seed models. Test predictions recomputed from selected checkpoints and verified against saved predictions.',
            'R2_scatter_plot_with_log.pdf': 'Original four panels: train/test raw, then train/test log1p.',
            'most_populated_traps_time_series.pdf': 'Original nine AIMSurv and three VectorNet traps, 6x2 layout and cached original location basemap. AIMSurv uses held-out seed ensembles; VectorNet uses the 25 selected fold/seed models. As in the original, ISO weeks are aligned to a shared June-December axis, with centered three-week smoothing; AIMSurv observations are from 2020 and VectorNet from 2021. Original y limits expanded when needed to show every point.',
            'window_sensitivity.pdf': 'Architecture/dropout averages within each fold, then mean and fold SE; WMSLE only.',
            'bland_altman_test.pdf': 'Original single raw-scale panel. Descriptive mean difference +/- 1.96 population SD, matching the original notebook; no significance claim.',
            'weekly_rates_map.png': 'Deferred: European daily climate grid required.'})
        source_dir = Path(__file__).parent
        sources = list(source_dir.glob('*.py')) + list((source_dir / 'assets').glob('*'))
        for source in sources:
            target = staging / 'source_snapshot' / source.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_json(staging / 'figure_provenance.json', {
            'experiment_id': experiment_id,
            'generated_at_utc': datetime.now(timezone.utc).isoformat(),
            'selected_model': selected,
            'plotting_sources': {str(p.relative_to(source_dir)): digest(p) for p in sources},
            'verified_stage_manifests': {stage: digest(output / stage / 'done.json') for stage in stages},
            'training_provenance': 'Original experiment.json, checkpoints and source_snapshot retained unchanged.'})
        inventory = [str(p.relative_to(staging)) for p in staging.rglob('*') if p.is_file()]
        mark_complete(staging, experiment_id, inventory)
        if folder.exists():
            archive = output.parent / 'figure_versions' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            archive.parent.mkdir(exist_ok=True)
            folder.rename(archive)
            print(f'Previous figures preserved: {archive}', flush=True)
        shutil.move(str(staging), folder)
    print(f'Figures ready: {folder}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--regenerate', action='store_true')
    args = parser.parse_args()
    make_figures(args.output, read_json(args.output / 'experiment.json')['experiment_id'], args.regenerate)
