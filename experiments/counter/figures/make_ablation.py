"""Counter notebook layouts with fixed-fold revision statistics."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from AIedes.utils.counter_experiment import read_json


def input_label(row):
    if row.input_family == 'lag-only':
        return r'$ER_{2w}$'
    period = '13w' if row.weekly else '90d'
    label = rf'$AT_{{{period}}}$ + $TP_{{{period}}}$'
    if 'd2m' in row.input_family:
        label += rf' + $DP_{{{period}}}$'
    if row.lag_order:
        label += r' + $ER_{2w}$'
    return label


def plot_grid(table, losses, metric, destination, name, selected):
    single = len(losses) == 1
    paper = table.loc[(table.window.eq(90) | table.window.isna()) & table.loss.isin(losses)].copy()
    paper['label'] = paper.apply(input_label, axis=1)
    # Match the input orders in the two original notebooks exactly.
    if single:
        paper = paper.loc[paper.input_family.ne('lag-only')]
        paper['order'] = (~paper.weekly).astype(int)*4 + paper.lag_order + paper.input_family.str.contains('d2m').astype(int)
    else:
        paper['order'] = 1 + 4*paper.input_family.str.contains('d2m').astype(int) + paper.lag_order + paper.weekly.astype(int)
        paper.loc[paper.input_family.eq('lag-only'), 'order'] = 0
    labels = paper.sort_values('order').label.drop_duplicates().tolist()
    architectures = sorted(set(zip(paper.num_layers, paper.hidden_dim)))
    colors = sns.color_palette('Set1', n_colors=4)
    fig, axes = plt.subplots(len(losses), 3, figsize=(12, 5 if single else 9), sharey=True, squeeze=False)
    for r, loss in enumerate(losses):
        for c, dropout in enumerate([0., .2, .4]):
            ax = axes[r, c]
            rows = paper.loc[paper.loss.eq(loss) & paper.dropout.eq(dropout)]
            for i, (layers, width) in enumerate(architectures):
                group = rows.loc[rows.num_layers.eq(layers) & rows.hidden_dim.eq(width)].set_index('label').reindex(labels)
                x = np.arange(len(labels)) + (i-1.5)*(.1 if single else .12)
                ax.errorbar(x, group[f'test_{metric}_mean'], yerr=group[f'test_{metric}_se'],
                            fmt='o', color=colors[i], capsize=2, capthick=1.2 if single else 1.5,
                            linewidth=1.8 if single else 1.5, markersize=6, alpha=.85 if single else .9,
                            label=f'{layers}-Layers, {width}-units')
                chosen = group.config_id.eq(selected)
                if single and chosen.any():
                    ax.errorbar(x[chosen], group.loc[chosen, f'test_{metric}_mean'],
                                yerr=group.loc[chosen, f'test_{metric}_se'], fmt='*', markersize=16,
                                linewidth=0, zorder=6, mfc=colors[i], mec='black', mew=1)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels if r == len(losses)-1 else [], rotation=45, ha='right', fontsize=9)
            title = f'({chr(97+c)}) - Dropout = {dropout:.1f}' if single else f'Dropout = {dropout:.1f}'
            ax.text(.2 if single else .3, .9 if single else .92, title,
                    transform=ax.transAxes, fontsize=11, ha='center' if single else 'right',
                    va='bottom' if single else 'center', bbox=dict(boxstyle='round,pad=0.25', edgecolor='black', facecolor='white', alpha=.85))
            ax.grid(True, alpha=.3)
            if c == 0:
                ax.set_ylabel(r'Test $R^2$ Score' if single else (r'Test $R^2$' if metric == 'r2' else 'Test Accuracy'), fontsize=11)
                if not single:
                    ax.text(-.25, .5, f'Loss: {loss}', rotation=90, transform=ax.transAxes, ha='center', va='center', fontweight='bold')
    if single:
        handles, names = axes[0,0].get_legend_handles_labels()
        handles.append(Line2D([], [], marker='*', linestyle='None', markersize=12, markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.2))
        fig.legend(handles, names+['Selected AIedes model'], loc='upper center', ncol=5, frameon=True, fancybox=True, bbox_to_anchor=(.5,.9))
    else:
        handles = [Line2D([0],[0],marker='o',linestyle='-',linewidth=2,color=colors[i],label=f'{l}-Layers, {w}-units') for i,(l,w) in enumerate(architectures)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5,.98), ncol=4, frameon=True, fancybox=True, title='Architecture')
    # Expand the original limits only where the revised means/error bars need room.
    lower, upper = (.45,.65) if metric == 'r2' else (.88,.94)
    means, errors = paper[f'test_{metric}_mean'], paper[f'test_{metric}_se']
    pad = .01 if metric == 'r2' else .003
    axes[0,0].set_ylim(min(lower, float((means-errors).min())-pad), max(upper, float((means+errors).max())+pad))
    fig.tight_layout()
    fig.subplots_adjust(top=.80 if single else .90, wspace=.1, hspace=.10)
    fig.savefig(destination / f'{name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(destination / f'{name}.png', bbox_inches='tight', dpi=180)
    plt.close(fig)


def make_ablation(output, destination=None):
    destination = destination or output / 'figures'
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
    families = sorted(summary.input_family.unique())
    orders = sorted(summary.lag_order.unique())
    fig, axes = plt.subplots(len(families), len(orders), figsize=(6*len(orders), 4*len(families)), squeeze=False)
    for i, family in enumerate(families):
        for j, order in enumerate(orders):
            ax = axes[i, j]
            subset = summary.loc[summary.input_family.eq(family) & summary.lag_order.eq(order)]
            for weekly, group in subset.groupby('weekly'):
                ax.errorbar(group.window, group['mean'], yerr=group['sem'], marker='o', capsize=3,
                            label='Weekly means' if weekly else 'Daily inputs')
            ax.set(title=f"{'T+D+P' if 'd2m' in family else 'T+P'} · {order} lags",
                   xlabel='Climate window (days)', ylabel='Mean test R² across configurations')
            ax.set_xticks([30, 60, 90]); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination / 'window_sensitivity.pdf')
    plt.close(fig)
