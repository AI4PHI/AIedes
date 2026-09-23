"""Strict outer-held-out scores; no model/configuration selection here."""
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd

from AIedes.utils.counter_experiment import complete, mark_complete, read_json, write_json
from AIedes.evaluation.counter_reconstruction import load_model, observed_prediction, trajectory
from AIedes.evaluation.counter_eval import compute_metrics_from_arrays as scores


def model_folder(output, outer, candidate_id, seed):
    return Path(output) / "final_models" / f"outer{outer}" / f"{candidate_id}_seed{seed}"


def summarize(frame, columns, config):
    output = {"traps": int(frame.id_trap.nunique()), "observations": len(frame),
              "metrics": {c: scores(frame.target, frame[c]) for c in columns}, "history_strata": {}}
    for name, part in frame.groupby("history_class"):
        output["history_strata"][name] = {"traps": int(part.id_trap.nunique()),
                                        "metrics": {c: scores(part.target, part[c]) for c in columns}}
    if "startup" in frame:
        output["startup_strata"] = {
            str(name): {"traps": int(part.id_trap.nunique()), "metrics": {c: scores(part.target, part[c]) for c in columns}}
            for name, part in frame.groupby("startup")}
    output["bootstrap"] = bootstrap(frame, columns, config["bootstrap_samples"], config["bootstrap_seed"])
    output["seed_metrics"] = {c: scores(frame.target, frame[c]) for c in frame if "_seed" in c}
    return output


def evaluate(output, config, experiment_id):
    output = Path(output)
    folder = output / "evaluation"
    if complete(folder, experiment_id) and complete(output / "export", experiment_id):
        return
    folder.mkdir(parents=True, exist_ok=True)
    if not complete(output / "selection", experiment_id):
        raise ValueError("Selection must complete first")
    selections = read_json(output / "selection/winners.json")
    frame = pd.read_pickle(output / "data_audit/observations.pkl")
    with (output / "data_audit/weather.pkl").open("rb") as f:
        weather = pickle.load(f)
    main_parts, reconstruction_parts, daily_parts, reports = [], [], [], {}
    export_manifest = {"deployment_choice": "deferred", "outer_ensembles": {},
                       "experiment_id": experiment_id, "climate_alignment": config["climate_alignment_evidence"],
                       "warning": "Never score existing observations using models that trained or selected on their trap."}
    export = output / "export"
    export.mkdir(exist_ok=True)
    for outer in range(1, 6):
        chosen = selections[str(outer)]
        test = frame.loc[frame.outer_group.eq(outer)].copy()
        metadata = ["row_id", "id_trap", "country", "end_date", "outer_group", "history_class", "plot_eligible", "reconstruction_eligible"]
        table = test[metadata].copy().rename(columns={"end_date": "date"})
        table["target"] = test.weeklyRates.to_numpy(float)
        loaded = {}
        paths = {}
        for role in ("main", "climate_only"):
            candidate = chosen[role]["candidate"]
            ensemble = []
            paths[role] = []
            for seed in config["final_seeds"]:
                job = model_folder(output, outer, candidate["id"], seed)
                if not complete(job, experiment_id):
                    raise ValueError(f"Missing refit {job}")
                model, package = load_model(job / "model.pt")
                if set(package["training_row_ids"]) & set(test.row_id):
                    raise ValueError("Test row entered refit")
                if package["outer_group"] != outer or package["experiment_id"] != experiment_id:
                    raise ValueError("Wrong model provenance")
                ensemble.append((model, package))
                column = ("prediction" if role == "main" else role) + f"_seed{seed}"
                table[column] = observed_prediction(model, package, test)
                relative = Path(f"outer{outer}") / f"{candidate['id']}_seed{seed}.pt"
                (export / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(job / "model.pt", export / relative)
                paths[role].append(str(relative))
            prefix = "prediction" if role == "main" else role
            table[prefix] = table[[prefix + f"_seed{s}" for s in config["final_seeds"]]].mean(axis=1)
            loaded[role] = ensemble
        export_manifest["outer_ensembles"][str(outer)] = paths
        main_parts.append(table)
        rec = table.loc[table.reconstruction_eligible].copy().rename(columns={"prediction": "observed_history"})
        rec["autonomous"] = np.nan
        rec["startup"] = False
        for seed in config["final_seeds"]:
            rec[f"autonomous_seed{seed}"] = np.nan
        for trap, part in rec.groupby("id_trap"):
            trajectories = []
            for (model, package), seed in zip(loaded["main"], config["final_seeds"]):
                daily = trajectory(model, package, weather[int(trap)], part.date.min(), part.date.max())
                aligned = daily.set_index("date").loc[pd.to_datetime(part.date)]
                rec.loc[part.index, f"autonomous_seed{seed}"] = aligned.prediction.to_numpy()
                trajectories.append(daily.prediction.to_numpy())
            daily["prediction"] = np.mean(trajectories, axis=0)
            daily["id_trap"], daily["outer_group"] = int(trap), outer
            for seed, values in zip(config["final_seeds"], trajectories):
                daily[f"prediction_seed{seed}"] = values
            rec.loc[part.index, "autonomous"] = daily.set_index("date").loc[pd.to_datetime(part.date)].prediction.to_numpy()
            rec.loc[part.index, "startup"] = daily.set_index("date").loc[pd.to_datetime(part.date)].startup.to_numpy()
            daily_parts.append(daily)
        if rec.autonomous.isna().any():
            raise ValueError("Incomplete autonomous predictions")
        reconstruction_parts.append(rec)
        reports[str(outer)] = {"main_test": summarize(table, ["prediction", "climate_only"], config),
                              "reconstruction": summarize(rec, ["autonomous", "observed_history", "climate_only"], config)}
        print(f"Evaluated outer {outer}: {len(table)} main / {len(rec)} reconstruction rows", flush=True)
    main = pd.concat(main_parts).sort_values("row_id").reset_index(drop=True)
    rec = pd.concat(reconstruction_parts).sort_values("row_id").reset_index(drop=True)
    if main.row_id.tolist() != sorted(frame.row_id.tolist()) or main.row_id.duplicated().any():
        raise ValueError("Main OOF coverage mismatch")
    if rec.row_id.tolist() != sorted(frame.loc[frame.reconstruction_eligible, "row_id"].tolist()):
        raise ValueError("Reconstruction OOF coverage mismatch")
    main.to_csv(folder / "main_predictions.csv", index=False)
    rec.to_csv(folder / "reconstruction_predictions.csv", index=False)
    pd.concat(daily_parts).to_csv(folder / "daily_trajectories.csv", index=False)
    write_json(folder / "metrics_by_outer.json", reports)
    write_json(folder / "pooled_metrics.json", {
        "main_test": summarize(main, ["prediction", "climate_only"], config),
        "reconstruction": summarize(rec, ["autonomous", "observed_history", "climate_only"], config),
        "climate_alignment": config["climate_alignment_evidence"],
        "interpretation": ("Configuration selected by mean five-fold test R2. Seed ensembles are averaged within each held-out fold; every prediction uses models trained on the other four folds."
                           if config.get('schema_version') == 2 else
                           "Outer-fold models can differ. Pooled scores evaluate the selection/training procedure, not a deployed 25-model ensemble.")})
    variability = []
    for cohort in ("main_test", "reconstruction"):
        roles = reports["1"][cohort]["metrics"]
        for role in roles:
            for metric in ("r2", "log_r2", "accuracy"):
                values = [reports[str(o)][cohort]["metrics"][role][metric] for o in range(1, 6)]
                defined = np.asarray([v for v in values if v is not None])
                variability.append({"cohort": cohort, "prediction": role, "metric": metric,
                    "mean": float(defined.mean()) if len(defined) else None,
                    "sd_across_outer_groups": float(defined.std(ddof=1)) if len(defined) > 1 else None,
                    "min": float(defined.min()) if len(defined) else None, "max": float(defined.max()) if len(defined) else None,
                    "defined_groups": len(defined)})
    pd.DataFrame(variability).to_csv(folder / "split_variability.csv", index=False)
    write_json(export / "ensembles.json", export_manifest)
    shutil.copy2(output / "experiment.json", export / "experiment.json")
    write_json(export / "winners.json", selections)
    mark_complete(export, experiment_id, [str(p.relative_to(export)) for p in export.rglob("*") if p.is_file() and p.name != "done.json"])
    mark_complete(folder, experiment_id, ["main_predictions.csv", "reconstruction_predictions.csv", "daily_trajectories.csv",
                  "metrics_by_outer.json", "pooled_metrics.json", "split_variability.csv"])


def bootstrap(frame, columns, repetitions, seed):
    """Matched whole-trap resampling, conditional on already fitted predictions."""
    groups = [g.index.to_numpy() for _, g in frame.reset_index(drop=True).groupby("id_trap")]
    frame = frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    values = {c: {m: [] for m in ("r2", "log_r2", "accuracy")} for c in columns}
    differences = {c: {m: [] for m in ("r2", "log_r2", "accuracy")} for c in columns[1:]}
    for _ in range(repetitions if groups else 0):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        sample = frame.iloc[idx]
        measured = {c: scores(sample.target, sample[c]) for c in columns}
        for c in columns:
            for m in values[c]:
                if measured[c][m] is not None:
                    values[c][m].append(measured[c][m])
        for c in differences:
            for m in differences[c]:
                a, b = measured[c][m], measured[columns[0]][m]
                if a is not None and b is not None:
                    differences[c][m].append(a-b)
    def summarize(collection):
        return {c: {m: {"interval": np.quantile(v, [.025, .975]).tolist() if v else None,
                        "defined_resamples": len(v), "undefined_resamples": repetitions-len(v)}
                    for m, v in metrics.items()} for c, metrics in collection.items()}
    return {"conditional_on_fitted_predictions": True, "resamples": repetitions,
            "scores": summarize(values), "differences_vs_" + columns[0]: summarize(differences)}


def plot_results(output, experiment_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    if not complete(output / "evaluation", experiment_id):
        raise ValueError("Evaluation must finish before plotting")
    figures = output / "figures"
    if complete(figures, experiment_id):
        return
    figures.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(output / "evaluation/main_predictions.csv", parse_dates=["date"])
    rec = pd.read_csv(output / "evaluation/reconstruction_predictions.csv", parse_dates=["date"])
    daily = pd.read_csv(output / "evaluation/daily_trajectories.csv", parse_dates=["date"])
    for name, frame, columns in [("main", main, ["prediction", "climate_only"]),
                                  ("reconstruction", rec, ["autonomous", "observed_history", "climate_only"])]:
        for column in columns:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            metric = scores(frame.target, frame[column])
            for ax, log in zip(axes, [False, True]):
                y = np.log1p(frame.target) if log else frame.target
                p = np.log1p(frame[column]) if log else frame[column]
                ax.scatter(y, p, s=7, alpha=.3)
                maximum = max(float(y.max()), float(p.max()), 1.)
                ax.plot([0, maximum], [0, maximum], color="black", linestyle="--")
                ax.set(xlabel="Observed log(1 + eggs/week)" if log else "Observed eggs/week",
                       ylabel="Predicted log(1 + eggs/week)" if log else "Predicted eggs/week",
                       title=f"{'Log' if log else 'Raw'} R² = {metric['log_r2' if log else 'r2']}")
            fig.suptitle(f"{name}: {column}; {len(frame)} held-out observations")
            fig.tight_layout()
            fig.savefig(figures / f"{name}_{column}_scatter.png", dpi=140)
            fig.savefig(figures / f"{name}_{column}_scatter.pdf")
            plt.close(fig)
    gallery = figures / "trap_curves"
    gallery.mkdir(exist_ok=True)
    plot_rows = []
    for trap, observed in rec.groupby("id_trap"):
        if len(observed) <= 10:
            continue
        simulated = daily.loc[daily.id_trap.eq(trap)].sort_values("date")
        observed = observed.sort_values("date")
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for ax, log in zip(axes, [False, True]):
            convert = np.log1p if log else np.asarray
            ax.plot(simulated.date, convert(simulated.prediction), label="Autonomous", color="tab:blue")
            ax.scatter(observed.date, convert(observed.target), label="Observed", color="black", s=18)
            ax.plot(observed.date, convert(observed.observed_history), label="Observed-history prediction", alpha=.7)
            ax.plot(observed.date, convert(observed.climate_only), label="Climate-only", alpha=.7)
            ax.set_ylabel("log(1 + eggs/week)" if log else "Eggs/week")
        axes[0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"Trap {trap} — {observed.country.iloc[0]} — held-out group {observed.outer_group.iloc[0]}")
        fig.autofmt_xdate()
        fig.tight_layout()
        filename = f"trap_{trap}.png"
        fig.savefig(gallery / filename, dpi=140)
        plt.close(fig)
        plot_rows.append(dict(id_trap=int(trap), observations=len(observed), country=observed.country.iloc[0],
                              outer_group=int(observed.outer_group.iloc[0]), figure=filename))
    pd.DataFrame(plot_rows).to_csv(figures / "plot_inventory.csv", index=False)
    records = pd.read_csv(output / "selection/comparison.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    for outer, part in records.groupby("outer_group"):
        ax.plot(np.arange(len(part)), np.sort(part.log_mse), label=f"Outer {outer}")
    ax.set(xlabel="Configuration rank within outer development set", ylabel="Pooled validation log error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "configuration_comparison.png", dpi=140)
    plt.close(fig)
    inventory = [str(p.relative_to(figures)) for p in figures.rglob("*") if p.is_file() and p.name != "done.json"]
    mark_complete(figures, experiment_id, inventory)
