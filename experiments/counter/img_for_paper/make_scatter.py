"""Original Counter scatter/Bland–Altman layouts; selected-model ensembles."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from AIedes.evaluation.counter_reconstruction import load_model, observed_prediction
from AIedes.utils.counter_experiment import read_json, write_json, complete


def make_scatter(output, destination=None):
    import numpy as np
    import matplotlib.pyplot as plt
    destination = destination or output / 'figures'
    selected = read_json(output / 'selection/selected.json')['main']
    eid = read_json(output / 'experiment.json')['experiment_id']
    frame = pd.read_pickle(output / 'data_audit/observations.pkl')
    test = pd.read_csv(output / 'evaluation/main_predictions.csv').set_index('row_id').reindex(frame.row_id)
    total, counts = np.zeros(len(frame)), np.zeros(len(frame), dtype=int)
    heldout, heldout_counts = np.zeros(len(frame)), np.zeros(len(frame), dtype=int)
    torch.set_num_threads(1)
    for fold in range(1,6):
        for seed in range(1,6):
            folder = output / 'final_models' / f'outer{fold}' / f"{selected['id']}_seed{seed}"
            if not complete(folder, eid):
                raise ValueError(f'Missing checkpoint: {folder}')
            model, package = load_model(folder / 'model.pt')
            if package['candidate'] != selected:
                raise ValueError('Checkpoint differs from the selected model')
            mask = frame.row_id.isin(package['training_row_ids']).to_numpy()
            if not np.array_equal(~mask, frame.outer_group.eq(fold).to_numpy()):
                raise ValueError('Training and held-out trap membership disagree')
            prediction = observed_prediction(model, package, frame)
            total[mask] += prediction[mask]; counts[mask] += 1
            heldout[~mask] += prediction[~mask]; heldout_counts[~mask] += 1
    if not (np.all(counts == 20) and np.all(heldout_counts == 5)):
        raise ValueError('Incomplete training/test ensembles')
    np.testing.assert_allclose(heldout/heldout_counts, test.prediction, rtol=2e-6, atol=1e-5)
    np.testing.assert_allclose(frame.weeklyRates, test.target)
    training = pd.DataFrame({'row_id': frame.row_id, 'target_data': frame.weeklyRates, 'predictions': total/counts, 'split': 'train'})
    testing = pd.DataFrame({'row_id': frame.row_id.to_numpy(), 'target_data': test.target.to_numpy(), 'predictions': test.prediction.to_numpy(), 'split': 'test'})
    data_egg = pd.concat([training, testing], ignore_index=True)
    training.to_csv(destination / 'scatter_training_predictions.csv', index=False)
    from sklearn.metrics import r2_score, mean_squared_error
    import seaborn as sns
    import matplotlib.pyplot as plt

    # scatter plot of targets vs predictions with seaborn
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharex=True, sharey=True)
    data_egg_train = data_egg[data_egg["split"] == "train"]
    data_egg_test = data_egg[data_egg["split"] == "test"]

    # Train plot
    sns.scatterplot(x=data_egg_train["target_data"], y=data_egg_train["predictions"], ax=axs[0], color='blue', alpha=0.6, label='Train Data Points (in-sample)')
    r2_train = r2_score(data_egg_train["target_data"], data_egg_train["predictions"])
    rmse_train = np.sqrt(mean_squared_error(data_egg_train["target_data"], data_egg_train["predictions"]))
    # Add linear fit
    z_train = np.polyfit(data_egg_train["target_data"], data_egg_train["predictions"], 1)
    p_train = np.poly1d(z_train)
    fit_label_train = f'Linear Fit (y={z_train[0]:.2f}x + {z_train[1]:.2f})'
    axs[0].plot(data_egg_train["target_data"], p_train(data_egg_train["target_data"]), "r--", alpha=0.8, label=fit_label_train)
    #axs[0].set_title('Train')
    axs[0].set_xlabel("Weekly Egg-laying Rate")
    axs[0].set_ylabel("AIedes-Counter Predictions")

    # Test plot
    sns.scatterplot(x=data_egg_test["target_data"], y=data_egg_test["predictions"], ax=axs[1], color='orange', alpha=0.6, label='Test Data Points (out-of-fold)')
    r2_test = r2_score(data_egg_test["target_data"], data_egg_test["predictions"])
    rmse_test = np.sqrt(mean_squared_error(data_egg_test["target_data"], data_egg_test["predictions"]))
    # Add linear fit
    z_test = np.polyfit(data_egg_test["target_data"], data_egg_test["predictions"], 1)
    p_test = np.poly1d(z_test)
    fit_label_test = f'Linear Fit (y={z_test[0]:.2f}x + {z_test[1]:.2f})'
    axs[1].plot(data_egg_test["target_data"], p_test(data_egg_test["target_data"]), "r--", alpha=0.8, label=fit_label_test)
    #axs[1].set_title('Test')
    axs[1].set_xlabel("Weekly Egg-laying Rate")
    axs[1].set_ylabel("AIedes-Counter Predictions")

    # Determine the overall range for x and y axes
    x_max = max(data_egg_train["target_data"].max(), data_egg_test["target_data"].max())
    y_max = max(data_egg_train["predictions"].max(), data_egg_test["predictions"].max())
    max_limit = max(x_max, y_max) * 1.05 # Use the greater of the two for a square plot, with a 5% buffer

    # Set the same limits for both plots
    axs[0].set_xlim(0, max_limit)
    axs[0].set_ylim(0, max_limit)
    axs[1].set_xlim(0, max_limit)
    axs[1].set_ylim(0, max_limit)

    # Add a diagonal line for reference and legend
    axs[0].plot([0, max_limit], [0, max_limit], 'k--', lw=1, label='Bisector (y=x)')
    axs[1].plot([0, max_limit], [0, max_limit], 'k--', lw=1, label='Bisector (y=x)')

    # Add metrics as legend entries (invisible plots for legend only)
    axs[0].plot([], [], ' ', label=f'R²: {r2_train:.2f}')
    axs[0].plot([], [], ' ', label=f'RMSE: {rmse_train:.1f}')
    axs[1].plot([], [], ' ', label=f'R²: {r2_test:.2f}')
    axs[1].plot([], [], ' ', label=f'RMSE: {rmse_test:.1f}')

    # Add legends
    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    plt.subplots_adjust(wspace=.1)  # Increase horizontal space between plots

    #plt.tight_layout()
    plt.savefig(destination / "R2_scatter_plot.pdf")
    plt.savefig(destination / "R2_scatter_plot.png", dpi=180)
    plt.close(fig)
    from sklearn.metrics import r2_score, mean_squared_error
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    # scatter plot of targets vs predictions with seaborn
    sns.set(style="whitegrid")
    fig, axs = plt.subplots(1, 4, figsize=(22, 6))
    data_egg_train = data_egg[data_egg["split"] == "train"]
    data_egg_test = data_egg[data_egg["split"] == "test"]

    # --- Plot 1: Original Data Train ---
    sns.scatterplot(x=data_egg_train["target_data"], y=data_egg_train["predictions"], ax=axs[0], color='blue', alpha=0.6, label='Points')
    r2_train = r2_score(data_egg_train["target_data"], data_egg_train["predictions"])
    rmse_train = np.sqrt(mean_squared_error(data_egg_train["target_data"], data_egg_train["predictions"]))
    # Add linear fit
    z_train = np.polyfit(data_egg_train["target_data"], data_egg_train["predictions"], 1)
    p_train = np.poly1d(z_train)
    fit_label_train = f'Linear Fit (y={z_train[0]:.2f}x + {z_train[1]:.2f})'
    axs[0].plot(data_egg_train["target_data"], p_train(data_egg_train["target_data"]), "r--", alpha=0.8, label=fit_label_train)
    axs[0].set_title(f'Train, in-sample (R²: {r2_train:.3f}, RMSE: {rmse_train:.1f})')
    axs[0].set_xlabel("Actual data")
    axs[0].set_ylabel("Predictions")

    # --- Plot 2: Original Data Test ---
    sns.scatterplot(x=data_egg_test["target_data"], y=data_egg_test["predictions"], ax=axs[1], color='orange', alpha=0.6, label='Points')
    r2_test = r2_score(data_egg_test["target_data"], data_egg_test["predictions"])
    rmse_test = np.sqrt(mean_squared_error(data_egg_test["target_data"], data_egg_test["predictions"]))
    # Add linear fit
    z_test = np.polyfit(data_egg_test["target_data"], data_egg_test["predictions"], 1)
    p_test = np.poly1d(z_test)
    fit_label_test = f'Linear Fit (y={z_test[0]:.2f}x + {z_test[1]:.2f})'
    axs[1].plot(data_egg_test["target_data"], p_test(data_egg_test["target_data"]), "r--", alpha=0.8, label=fit_label_test)
    axs[1].set_title(f'Test, out-of-fold (R²: {r2_test:.3f}, RMSE: {rmse_test:.1f})')
    axs[1].set_xlabel("Actual data")
    axs[1].set_ylabel("Predictions")

    # Determine the overall range for x and y axes for the original data plots
    x_max = max(data_egg_train["target_data"].max(), data_egg_test["target_data"].max())
    y_max = max(data_egg_train["predictions"].max(), data_egg_test["predictions"].max())
    max_limit = max(x_max, y_max) * 1.05 # Use the greater of the two for a square plot, with a 5% buffer

    # Set the same limits for the first two plots
    for ax in axs[:2]:
        ax.set_xlim(0, max_limit)
        ax.set_ylim(0, max_limit)
        ax.plot([0, max_limit], [0, max_limit], 'k--', lw=1, label='Bisector (y=x)')
        ax.legend(loc='upper left')

    # --- Plot 3: Log-transformed Data Train ---
    # Prepare log-transformed data
    log_target_train = np.log1p(data_egg_train["target_data"])
    log_pred_train = np.log1p(data_egg_train["predictions"])
    log_target_test = np.log1p(data_egg_test["target_data"])
    log_pred_test = np.log1p(data_egg_test["predictions"])

    sns.scatterplot(x=log_target_train, y=log_pred_train, ax=axs[2], color='blue', alpha=0.6, label='Points')
    r2_log_train = r2_score(log_target_train, log_pred_train)
    rmse_log_train = np.sqrt(mean_squared_error(log_target_train, log_pred_train))
    # Add linear fit
    z_log_train = np.polyfit(log_target_train, log_pred_train, 1)
    p_log_train = np.poly1d(z_log_train)
    fit_label_log_train = f'Linear Fit (y={z_log_train[0]:.2f}x + {z_log_train[1]:.2f})'
    axs[2].plot(log_target_train, p_log_train(log_target_train), "r--", alpha=0.8, label=fit_label_log_train)
    axs[2].set_title(f'Log-Transformed Train, in-sample (R²: {r2_log_train:.3f}, RMSE: {rmse_log_train:.1f})')
    axs[2].set_xlabel("log(1 + Actual data)")
    axs[2].set_ylabel("log(1 + Predictions)")

    # --- Plot 4: Log-transformed Data Test ---
    sns.scatterplot(x=log_target_test, y=log_pred_test, ax=axs[3], color='orange', alpha=0.6, label='Points')
    r2_log_test = r2_score(log_target_test, log_pred_test)
    rmse_log_test = np.sqrt(mean_squared_error(log_target_test, log_pred_test))
    # Add linear fit
    z_log_test = np.polyfit(log_target_test, log_pred_test, 1)
    p_log_test = np.poly1d(z_log_test)
    fit_label_log_test = f'Linear Fit (y={z_log_test[0]:.2f}x + {z_log_test[1]:.2f})'
    axs[3].plot(log_target_test, p_log_test(log_target_test), "r--", alpha=0.8, label=fit_label_log_test)
    axs[3].set_title(f'Log-Transformed Test, out-of-fold (R²: {r2_log_test:.3f}, RMSE: {rmse_log_test:.1f})')
    axs[3].set_xlabel("log(1 + Actual data)")
    axs[3].set_ylabel("log(1 + Predictions)")

    # Determine the overall range for x and y axes for the log-transformed plots
    log_x_max = max(log_target_train.max(), log_target_test.max())
    log_y_max = max(log_pred_train.max(), log_pred_test.max())
    log_max_limit = max(log_x_max, log_y_max) * 1.05

    # Set the same limits for the last two plots
    for ax in axs[2:]:
        ax.set_xlim(0, log_max_limit)
        ax.set_ylim(0, log_max_limit)
        ax.plot([0, log_max_limit], [0, log_max_limit], 'k--', lw=1, label='Bisector (y=x)')
        ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(destination / "R2_scatter_plot_with_log.pdf")
    plt.close(fig)

    import numpy as np
    import matplotlib.pyplot as plt

    def bland_altman_plot(actual, predicted, ax=None, title='Bland-Altman Plot'):
        mean = np.mean([actual, predicted], axis=0)
        diff = predicted - actual
        md = np.mean(diff)
        sd = np.std(diff)
        loa_upper = md + 1.96*sd
        loa_lower = md - 1.96*sd

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.scatter(mean, diff, alpha=0.5)
        ax.axhline(md, color='gray', linestyle='--', label=f'Mean diff = {md:.2f}')
        ax.axhline(loa_upper, color='red', linestyle='--', label=f'+1.96 SD = {loa_upper:.2f}')
        ax.axhline(loa_lower, color='red', linestyle='--', label=f'-1.96 SD = {loa_lower:.2f}')
        ax.set_xlabel('Mean of predicted and actual')
        ax.set_ylabel('Predicted − Actual')
        ax.set_title(title)
        ax.legend()

        return ax

    # Example for test set:
    fig, ax = plt.subplots(figsize=(8,6))
    bland_altman_plot(
        data_egg_test["target_data"].values,
        data_egg_test["predictions"].values,
        ax=ax,
        title="Bland-Altman Plot (out-of-fold test set)"
    )
    plt.tight_layout()
    plt.savefig(destination / "bland_altman_test.pdf")
    plt.close(fig)
    difference = data_egg_test.predictions - data_egg_test.target_data
    bias, sd = float(difference.mean()), float(np.std(difference))
    write_json(destination / 'bland_altman_statistics.json', {
        'raw': {'bias': bias, 'sd_ddof': 0,
                'descriptive_limits': [bias-1.96*sd, bias+1.96*sd]}})
