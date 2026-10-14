"""Publication scatter and agreement plots from frozen held-out-trap predictions."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from AIedes.utils.counter_experiment import write_json


def regression_metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual_ss = np.square(observed - predicted).sum()
    total_ss = np.square(observed - observed.mean()).sum()
    return {
        'r2': float(1 - residual_ss / total_ss),
        'rmse': float(np.sqrt(np.mean(np.square(observed - predicted)))),
    }


def prediction_table(output):
    path = Path(output) / 'evaluation' / 'main_predictions.csv'
    frame = pd.read_csv(path)
    seed_columns = sorted(
        column for column in frame.columns if column.startswith('prediction_seed')
    )
    if len(seed_columns) != 5:
        raise ValueError(f'Expected five held-out seed predictions, found {seed_columns}')
    required = {'row_id', 'id_trap', 'target', 'prediction'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'Missing prediction columns: {sorted(missing)}')
    seed_predictions = frame[seed_columns].to_numpy(dtype=float)
    np.testing.assert_allclose(
        seed_predictions.mean(axis=1), frame.prediction.to_numpy(dtype=float),
        rtol=2e-6, atol=1e-5,
    )
    return frame, seed_predictions


def make_heldout_scatter(output, destination=None):
    destination = Path(destination or Path(output) / 'figures')
    destination.mkdir(parents=True, exist_ok=True)
    frame, seed_predictions = prediction_table(output)

    training_path = Path(output) / 'figures' / 'scatter_training_predictions.csv'
    if not training_path.exists():
        raise FileNotFoundError(
            'Missing frozen in-sample predictions required for the four-panel '
            f'scatter figure: {training_path}'
        )
    training = pd.read_csv(training_path)
    required_training = {'row_id', 'target_data', 'predictions', 'split'}
    missing_training = required_training.difference(training.columns)
    if missing_training:
        raise ValueError(
            f'Missing training-prediction columns: {sorted(missing_training)}'
        )
    if not training['split'].eq('train').all():
        raise ValueError('Frozen in-sample prediction table contains non-training rows')
    if not np.array_equal(np.sort(training.row_id.to_numpy()), np.sort(frame.row_id.to_numpy())):
        raise ValueError('Training and held-out prediction tables use different observations')

    panels = [
        {
            'title': '(a) Training',
            'observed': training.target_data.to_numpy(dtype=float),
            'predicted': training.predictions.to_numpy(dtype=float),
            'color': '#1F5BFF',
            'points': 'Training points',
            'log': False,
        },
        {
            'title': '(b) Held-out',
            'observed': frame.target.to_numpy(dtype=float),
            'predicted': frame.prediction.to_numpy(dtype=float),
            'color': '#F5A623',
            'points': 'Held-out points',
            'log': False,
        },
        {
            'title': r'(c) Training, $\log(1+x)$',
            'observed': training.target_data.to_numpy(dtype=float),
            'predicted': training.predictions.to_numpy(dtype=float),
            'color': '#1F5BFF',
            'points': 'Training points',
            'log': True,
        },
        {
            'title': r'(d) Held-out, $\log(1+x)$',
            'observed': frame.target.to_numpy(dtype=float),
            'predicted': frame.prediction.to_numpy(dtype=float),
            'color': '#F5A623',
            'points': 'Held-out points',
            'log': True,
        },
    ]
    for panel in panels:
        if panel['log']:
            panel['observed'] = np.log1p(panel['observed'])
            panel['predicted'] = np.log1p(panel['predicted'])
        panel['metrics'] = regression_metrics(panel['observed'], panel['predicted'])

    raw_limit = np.ceil(
        max(panel['observed'].max() for panel in panels if not panel['log']) / 50
    ) * 50
    log_limit = max(
        max(panel['observed'].max(), panel['predicted'].max())
        for panel in panels if panel['log']
    ) * 1.035

    with plt.style.context('seaborn-v0_8-whitegrid'), plt.rc_context({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 17,
        'ytick.labelsize': 17,
        'legend.fontsize': 15,
    }):
        fig, axes = plt.subplots(1, 4, figsize=(19.0, 5.3))
        for index, (axis, panel) in enumerate(zip(axes, panels)):
            limit = log_limit if panel['log'] else raw_limit
            observed = panel['observed']
            predicted = panel['predicted']
            fit = np.polyfit(observed, predicted, 1)
            fit_x = np.array([0, limit])
            metric_label = 'log-$R^2$' if panel['log'] else '$R^2$'

            axis.scatter(
                observed, predicted, s=30, color=panel['color'], alpha=.56,
                edgecolors='none', rasterized=True, zorder=2,
                label=f"({chr(ord('a') + index)}) {panel['points']}",
            )
            axis.plot(fit_x, np.polyval(fit, fit_x), '--', color='#D62728',
                      linewidth=2.2, label='_nolegend_')
            axis.plot(
                [0, limit], [0, limit], '--', color='.15', linewidth=1.8,
                zorder=3, label='_nolegend_',
            )
            axis.plot([], [], ' ', label=rf'{metric_label}$={panel["metrics"]["r2"]:.3f}$')
            if panel['log']:
                axis.set(
                    xlabel=r'$\log(1+\mathrm{actual\ data})$',
                )
            else:
                axis.set(xlabel='Actual data')
            if index % 2 == 0:
                axis.set_ylabel(
                    r'$\log(1+\mathrm{predictions})$'
                    if panel['log'] else 'Predictions'
                )
            axis.set(xlim=(0, limit), ylim=(0, limit))
            axis.set_aspect('equal', adjustable='box')
            axis.grid(True, color='.88', linewidth=.8)
            axis.tick_params(length=5, width=1.0)
            axis.legend(loc='upper left', frameon=True, framealpha=.92,
                        borderpad=.35, labelspacing=.3, handlelength=1.6)

        fig.legend(
            handles=[
                Line2D([0], [0], color='#D62728', linestyle='--', linewidth=2.2,
                       label='Linear fit'),
                Line2D([0], [0], color='.15', linestyle='--', linewidth=1.8,
                       label='Equality line'),
            ],
            loc='upper center', bbox_to_anchor=(.5, .955), ncol=2,
            frameon=False, fontsize=16,
        )
        fig.tight_layout(rect=(0, 0, 1, .91), w_pad=.35)
        for name in ('R2_scatter_plot', 'R2_scatter_plot_with_log'):
            fig.savefig(destination / f'{name}.pdf', bbox_inches='tight', dpi=300)
        fig.savefig(destination / 'R2_scatter_plot.png', bbox_inches='tight', dpi=220)
        plt.close(fig)

    metrics = {
        'observations': int(len(frame)),
        'traps': int(frame.id_trap.nunique()),
        'training_r2': panels[0]['metrics']['r2'],
        'training_log_r2': panels[2]['metrics']['r2'],
        'heldout_r2': panels[1]['metrics']['r2'],
        'heldout_log_r2': panels[3]['metrics']['r2'],
        'heldout_rmse': panels[1]['metrics']['rmse'],
    }
    write_json(destination / 'scatter_metrics.json', metrics)
    return metrics


def make_bland_altman(output, destination=None):
    destination = Path(destination or Path(output) / 'figures')
    destination.mkdir(parents=True, exist_ok=True)
    frame, _ = prediction_table(output)
    observed = frame.target.to_numpy(dtype=float)
    predicted = frame.prediction.to_numpy(dtype=float)
    mean = (observed + predicted) / 2
    difference = predicted - observed
    bias = float(difference.mean())
    sd = float(difference.std(ddof=0))
    lower, upper = bias - 1.96 * sd, bias + 1.96 * sd

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(mean, difference, s=10, alpha=.25,
                 edgecolors='none', rasterized=True)
    axis.axhline(bias, color='.3', linestyle='--',
                 label=f'Mean difference = {bias:.2f}')
    axis.axhline(upper, color='#C44E52', linestyle='--',
                 label=f'+1.96 SD = {upper:.2f}')
    axis.axhline(lower, color='#C44E52', linestyle='--',
                 label=f'-1.96 SD = {lower:.2f}')
    axis.set(xlabel='Mean of predicted and observed rates',
             ylabel='Predicted minus observed rate',
             title='Held-out-trap agreement')
    axis.legend()
    fig.tight_layout()
    fig.savefig(destination / 'bland_altman_test.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    write_json(destination / 'bland_altman_statistics.json', {
        'raw': {'bias': bias, 'sd_ddof': 0,
                'descriptive_limits': [lower, upper]},
    })


def make_scatter(output, destination=None):
    """Pipeline entry point retained for the complete paper-figure workflow."""
    metrics = make_heldout_scatter(output, destination)
    make_bland_altman(output, destination)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Completed Counter results directory')
    parser.add_argument('--destination', type=Path, required=True,
                        help='Directory for regenerated figures')
    args = parser.parse_args()
    make_scatter(args.output.resolve(), args.destination.resolve())
