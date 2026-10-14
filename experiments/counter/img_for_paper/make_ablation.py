"""Counter notebook layouts with fixed-fold revision statistics."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from AIedes.utils.counter_experiment import read_json


def input_label(row):
    if row.input_family == 'lag-only':
        return r'$(E_{1w},E_{2w},I_{1w},I_{2w})$'
    period = '13w' if row.weekly else '90d'
    variables = 'T,P,D' if 'd2m' in row.input_family else 'T,P'
    label = rf'$({variables})_{{{period}}}$'
    if row.lag_order:
        label += '\n' + r'$(E_{1w},E_{2w},I_{1w},I_{2w})$'
    return label


def architecture_label(layers, width):
    """Display the actual hidden-layer widths rather than counting the output layer."""
    labels = {
        (2, 32): r'$[32]$',
        (2, 64): r'$[64]$',
        (3, 32): r'$[32,16]$',
        (3, 64): r'$[64,32]$',
    }
    return labels.get((layers, width), f'{layers - 1} hidden layers, first width {width}')


def direct_input_label(row):
    """Vertical variable labels shared by the main and SI figures."""
    if row.input_family == 'lag-only':
        return r'$(E,I)$'
    label = r'$(T,P,D)$' if 'd2m' in row.input_family else r'$(T,P)$'
    return label + r'$+(E,I)$' if row.lag_order else label


def draw_points(ax, rows, labels, architectures, colors, metric, offset, marker_size,
                selected=None, star=False):
    """Plot one architecture series per colour at the given label positions."""
    for i, (layers, width) in enumerate(architectures):
        group = (rows.loc[rows.num_layers.eq(layers) & rows.hidden_dim.eq(width)]
                 .set_index('label').reindex(labels))
        x = np.arange(len(labels)) + (i - 1.5) * offset
        ax.errorbar(x, group[f'test_{metric}_mean'], yerr=group[f'test_{metric}_se'],
                    fmt='o', color=colors[i], capsize=2, capthick=1.5, linewidth=1.5,
                    markersize=marker_size, alpha=.9,
                    label=architecture_label(layers, width))
        if star and selected is not None:
            chosen = group.config_id.eq(selected)
            if chosen.any():
                ax.errorbar(x[chosen], group.loc[chosen, f'test_{metric}_mean'],
                            yerr=group.loc[chosen, f'test_{metric}_se'], fmt='*', markersize=18,
                            linewidth=0, zorder=6, mfc=colors[i], mec='black', mew=1)


def plot_grid(table, losses, metric, destination, name, selected):
    single = len(losses) == 1
    paper = table.loc[(table.window.eq(90) | table.window.isna()) & table.loss.isin(losses)].copy()
    paper['label'] = paper.apply(input_label, axis=1)
    # The main figure shows the eight climate-based input sets; the SI figures
    # add the no-climate set in front of them. lag_order is 0 or 2, so the lag
    # term uses its presence rather than its value.
    if single:
        paper = paper.loc[paper.input_family.ne('lag-only')]
    paper['order'] = (
        1
        + 4 * (~paper.weekly).astype(int)
        + 2 * paper.lag_order.gt(0).astype(int)
        + paper.input_family.str.contains('d2m').astype(int)
    )
    paper.loc[paper.input_family.eq('lag-only'), 'order'] = 0
    ordered_inputs = paper.sort_values('order').drop_duplicates('label')
    labels = ordered_inputs.label.tolist()
    tick_labels = ordered_inputs.apply(direct_input_label, axis=1).tolist()
    # The y range is set by the climate-based configurations. Some no-climate
    # points fall outside it and are therefore not drawn.
    climate = paper.loc[paper.input_family.ne('lag-only')]
    architectures = sorted(set(zip(climate.num_layers, climate.hidden_dim)))
    colors = sns.color_palette('Set1', n_colors=4)
    dropouts = [0., .2, .4]
    # Offsets of the weekly/daily group boundaries, which shift by one position
    # when the no-climate set is present.
    shift = 0 if single else 1
    if single:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.1), sharey=True, squeeze=False)
    else:
        # Same width as the main figure so that, printed at \linewidth, both
        # figures shrink by the same factor and share effective font sizes.
        fig, axes = plt.subplots(len(losses), 3, figsize=(13.5, 12.0),
                                 sharey=True, squeeze=False)
    for r, loss in enumerate(losses):
        for c, dropout in enumerate(dropouts):
            ax = axes[r, c]
            rows = paper.loc[paper.loss.eq(loss) & paper.dropout.eq(dropout)]
            draw_points(ax, rows, labels, architectures, colors, metric,
                        .1 if single else .12, 9 if single else 7,
                        selected=selected, star=single)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(
                tick_labels if r == len(losses) - 1 else [],
                rotation=90, ha='center', va='top',
                fontsize=12.5 if single else 12.5,
            )
            ax.set_xlim(-.6, len(labels) - .4)
            if not single:
                ax.axvline(.5, color='.65', linestyle='--', linewidth=1)
            ax.axvline(3.5 + shift, color='.65', linestyle='--', linewidth=1)
            title = f'({chr(97+c)})  Dropout = {dropout:.1f}'
            if single:
                ax.set_title(title, loc='left', fontsize=17, pad=9)
            elif r == 0:
                ax.set_title(title, loc='left', fontsize=16.5, pad=9)
            if single or r == 0:
                axis_transform = ax.get_xaxis_transform()
                if single:
                    ax.text(1.5, .965, 'Weekly climate (13w)', transform=axis_transform,
                            ha='center', va='top', fontsize=14, color='.25')
                    ax.text(5.5, .965, 'Daily climate (90d)', transform=axis_transform,
                            ha='center', va='top', fontsize=14, color='.25')
                else:
                    # Short labels keep the three group names apart once the
                    # no-climate set occupies the first position.
                    ax.text(0, .965, 'No\nclimate', transform=axis_transform,
                            ha='center', va='top', fontsize=12, color='.25',
                            linespacing=.95)
                    ax.text(2.5, .965, 'Weekly (13w)', transform=axis_transform,
                            ha='center', va='top', fontsize=13, color='.25')
                    ax.text(6.5, .965, 'Daily (90d)', transform=axis_transform,
                            ha='center', va='top', fontsize=13, color='.25')
            ax.grid(True, alpha=.3)
            ax.tick_params(axis='y', labelsize=15 if single else 13)
            if c == 0:
                ax.set_ylabel(
                    r'$R^2$' if single else (
                        f'{loss}\n' + (r'Held-out-trap $R^2$' if metric == 'r2'
                                       else 'Zero/non-zero accuracy')
                    ),
                    fontsize=18 if single else 15,
                )
    if single:
        handles, names = axes[0, 0].get_legend_handles_labels()
        handles.append(Line2D([], [], marker='*', linestyle='None', markersize=12, markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.2))
        fig.legend(
            handles,
            names + ['Selected configuration'],
            loc='upper center',
            ncol=5,
            frameon=True,
            fancybox=True,
            bbox_to_anchor=(.5, .995),
            fontsize=15,
            columnspacing=1.25,
            handletextpad=.45,
        )
    else:
        handles = [Line2D([0], [0], marker='o', linestyle='-', linewidth=2,
                          color=colors[i], label=architecture_label(l, w))
                   for i, (l, w) in enumerate(architectures)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.995),
                   ncol=4, frameon=True, fancybox=True,
                   title='Hidden-layer architecture', fontsize=14, title_fontsize=14)
    pad = .01 if metric == 'r2' else .003
    means, errors = climate[f'test_{metric}_mean'], climate[f'test_{metric}_se']
    low = float((means - errors).min()) - pad
    high = float((means + errors).max()) + pad
    if single:
        # The main figure keeps its approved scale: a fixed R² ceiling that
        # leaves room between the highest error bars and the group labels.
        axes[0, 0].set_ylim(min(.45, low), .62 if metric == 'r2' else max(.94, high))
    else:
        # Climate-driven range with headroom for the group labels. No-climate
        # points outside this range are clipped; the captions say so.
        axes[0, 0].set_ylim(low, high + .17 * (high - low))
    fig.tight_layout(rect=(0, .025 if single else .02, 1, .89 if single else .94))
    fig.subplots_adjust(wspace=.1, hspace=.14)
    fig.savefig(destination / f'{name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(destination / f'{name}.png', bbox_inches='tight', dpi=180)
    plt.close(fig)


def make_ablation(output, destination=None):
    destination = destination or output / 'figures'
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(output / 'selection/configuration_summary.csv')
    selected = read_json(output / 'selection/selected.json')['main']['id']
    with plt.style.context('seaborn-v0_8-whitegrid'):
        plot_grid(table, ['WMSLE'], 'r2', destination, 'nn_results_by_dropout', selected)
        plot_grid(table, ['WMSE', 'WMSLE', 'ZINB'], 'r2', destination, 'nn_results_by_dropout_3loss', selected)
        plot_grid(table, ['WMSE', 'WMSLE', 'ZINB'], 'accuracy', destination, 'nn_results_by_dropout_3loss_acc', selected)
    runs = pd.read_csv(output / 'selection/grid_runs.csv')
    rows = runs.loc[runs.loss.eq('WMSLE') & runs.window.notna()]
    # Equal-weight architecture/dropout means within each fixed fold before averaging folds.
    folded = rows.groupby(['input_family', 'lag_order', 'weekly', 'window', 'fold']).test_r2.mean()
    summary = folded.groupby(level=[0, 1, 2, 3]).agg(['mean', 'sem']).reset_index()
    summary.to_csv(destination / 'window_sensitivity.csv', index=False)
    families = sorted(
        summary.input_family.unique(),
        key=lambda value: ('d2m' in value, value),
    )
    orders = sorted(summary.lag_order.unique())
    low = float((summary['mean'] - summary['sem']).min())
    high = float((summary['mean'] + summary['sem']).max())
    span = high - low
    fig, axes = plt.subplots(
        len(families), len(orders),
        figsize=(11.5, 7.6), sharex=True, sharey=True, squeeze=False,
    )
    series_style = {
        False: ('Daily climate values', 'tab:blue'),
        True: ('Weekly climate averages', 'tab:orange'),
    }
    panel_index = 0
    for i, family in enumerate(families):
        for j, order in enumerate(orders):
            ax = axes[i, j]
            subset = summary.loc[summary.input_family.eq(family) & summary.lag_order.eq(order)]
            for weekly, group in subset.groupby('weekly'):
                name, color = series_style[bool(weekly)]
                ax.errorbar(group.window, group['mean'], yerr=group['sem'], marker='o', capsize=3,
                            linewidth=2, markersize=7, color=color, label=name)
            variables = r'$(T,P,D)$' if 'd2m' in family else r'$(T,P)$'
            if order:
                variables += r'$+(E,I)$'
            ax.set_title(f'({chr(97 + panel_index)})  {variables}', loc='left', fontsize=15)
            panel_index += 1
            ax.set_xticks([30, 60, 90], ['30d / 5w', '60d / 9w', '90d / 13w'])
            ax.set_xlim(20, 100)
            ax.set_ylim(low - .04 * span, high + .06 * span)
            ax.grid(alpha=.25)
            ax.tick_params(labelsize=12)
            if i == len(families) - 1:
                ax.set_xlabel('Climate-history representation', fontsize=13)
            if j == 0:
                ax.set_ylabel(r'Mean held-out-trap $R^2$', fontsize=13)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc='upper center', ncol=2,
        frameon=True, fancybox=True, fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.subplots_adjust(wspace=.10, hspace=.24)
    fig.savefig(destination / 'window_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(destination / 'window_sensitivity.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Completed Counter results directory')
    parser.add_argument('--destination', type=Path, required=True,
                        help='Directory for regenerated ablation figures')
    args = parser.parse_args()
    make_ablation(args.output.resolve(), args.destination.resolve())
