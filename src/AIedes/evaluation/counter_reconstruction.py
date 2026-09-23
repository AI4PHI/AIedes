"""Reloadable prediction; autonomous API accepts climate and dates, never egg targets."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from AIedes.data_loader.counter_data_loader import (
    counter_feature_names as feature_names,
    transform_counter_features as transform,
)
from AIedes.models.counter_models import (
    counter_candidate_model as model_for,
    predict_counter_rates as predict,
)
from AIedes.utils.counter_experiment import FIELDS


def load_model(path):
    package = torch.load(Path(path), map_location="cpu", weights_only=False)
    if package["schema_version"] != 1 or feature_names(package["candidate"]) != package["feature_names"]:
        raise ValueError("Unsupported model/feature schema")
    model = model_for(package["candidate"], package["input_dim"])
    model.load_state_dict(package["state_dict"])
    model.eval()
    return model, package


def observed_prediction(model, package, frame):
    return predict(model, transform(frame, package["candidate"], package["stats"]))


def trajectory(model, package, weather, start, end):
    """One seed, daily weekly-rate predictions; independent state and exact day lags."""
    candidate, stats = package["candidate"], package["stats"]
    dates = pd.date_range(start, end, freq="D")
    history_dates = pd.date_range(pd.Timestamp(start)-pd.Timedelta(days=89), end)
    if not weather.index.is_unique or not history_dates.isin(weather.index).all():
        raise ValueError("Missing/duplicate dated climate: uninterrupted reconstruction unavailable")
    if not np.isfinite(weather.reindex(history_dates).to_numpy(float)).all():
        raise ValueError("Nonfinite dated climate")
    records = {}
    for field in candidate["fields"]:
        series = weather.reindex(history_dates)[field].to_numpy(float)
        records[field] = list(np.lib.stride_tricks.sliding_window_view(series, 90))
    records["prev1_rates"] = np.full(len(dates), np.nan)
    records["prev2_rates"] = np.full(len(dates), np.nan)
    x = transform(pd.DataFrame(records), candidate, stats)
    predictions = np.empty(len(dates), float)
    if not candidate["lags"]:
        predictions[:] = predict(model, x)
    else:
        model.eval()
        with torch.no_grad():
            for i in range(len(dates)):
                lag_days = package["autonomous_lag_days"][:candidate.get('lag_order', 2)]
                for k, lag in enumerate(lag_days):
                    if i >= lag:
                        field = f"prev{k+1}_rates"
                        x[i, -2*len(lag_days)+2*k] = (predictions[i-lag]-stats[field]["mean"]) / stats[field]["std"]
                        x[i, -2*len(lag_days)+2*k+1] = 1.
                predictions[i] = model(torch.from_numpy(x[i:i+1])).item()
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise ValueError("Invalid autonomous trajectory")
    active_lags = package["autonomous_lag_days"][:candidate.get('lag_order', 2)] if candidate['lags'] else []
    return pd.DataFrame({"date": dates, "prediction": predictions,
                         "startup": np.arange(len(dates)) < max(active_lags, default=0)})


def build_weather(frame, full):
    """Assemble one dated daily climate series per trap from overlapping caches."""
    weather, inventory, gaps = {}, [], []
    for trap, obs in frame.groupby("id_trap"):
        source = full.loc[full.id_trap == trap]
        chunks = []
        for row in source.itertuples(index=False):
            values = np.stack([np.asarray(getattr(row, c), dtype=float) for c in FIELDS], axis=1)
            if values.shape != (90, 3) or not np.isfinite(values).all():
                raise ValueError(f"Invalid climate arrays at trap {trap}")
            dates = pd.date_range(pd.Timestamp(row.end_date)-pd.Timedelta(days=89), periods=90)
            chunks.append(pd.DataFrame(values, index=dates, columns=FIELDS))
        merged = pd.concat(chunks).groupby(level=0)
        lo, hi = merged.min(), merged.max()
        if not np.isclose(lo.to_numpy(), hi.to_numpy(), rtol=1e-8, atol=1e-6).all():
            raise ValueError(f"Overlapping climate disagrees at trap {trap}")
        weather[int(trap)] = lo.sort_index()
        first, last = pd.Timestamp(obs.end_date.min()), pd.Timestamp(obs.end_date.max())
        required = pd.date_range(first-pd.Timedelta(days=89), last)
        missing = required.difference(lo.index)
        for date in missing:
            gaps.append({"id_trap": int(trap), "date": str(date.date())})
        inventory.append({"id_trap": int(trap), "outer_group": int(obs.outer_group.iloc[0]),
                          "country": str(obs.country.iloc[0]), "observations": len(obs),
                          "first_date": str(first.date()), "last_date": str(last.date()),
                          "missing_climate_days": len(missing),
                          "reconstruction_eligible": not len(missing),
                          "plot_eligible": not len(missing) and len(obs) > 10})
    return weather, pd.DataFrame(inventory), pd.DataFrame(gaps, columns=["id_trap", "date"])
