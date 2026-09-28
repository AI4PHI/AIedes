"""Export canonical numerical tables and publication-friendly TeX copies."""

def table_export(frame, path):
    frame.to_csv(path.with_suffix('.csv'), index=False)
    if 'test_r2_mean' in frame:
        compact = frame[['loss', 'window', 'input_variables', 'num_layers', 'hidden_dim', 'dropout']].copy()
        for metric, label in [('r2', 'R²'), ('log_r2', 'Log R²'), ('accuracy', 'Accuracy')]:
            compact[label] = [f'{mean:.3f} ± {se:.3f}' for mean, se in
                              zip(frame[f'test_{metric}_mean'], frame[f'test_{metric}_se'])]
    elif 'cohort' in frame:
        compact = frame[['cohort', 'mode', 'observations', 'traps', 'r2', 'log_r2', 'rmse', 'accuracy']]
    else:
        compact = frame
    compact.to_latex(path.with_suffix('.tex'), index=False, float_format='%.4f', escape=True)
