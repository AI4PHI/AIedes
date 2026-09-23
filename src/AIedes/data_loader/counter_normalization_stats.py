import numpy as np
import pandas as pd


normalization_stats = {'d2m_max': {'mean': np.float64(286.61156059883706), 'std': np.float64(6.04840804667878)},
 'd2m_max_monthly': {'mean': np.float64(286.61156059883706), 'std': np.float64(5.391567503226255)},
 'd2m_mean': {'mean': np.float64(284.2504400114741), 'std': np.float64(6.349310465362731)},
 'd2m_mean_monthly': {'mean': np.float64(284.25044001147415), 'std': np.float64(5.7206755625559635)},
 'd2m_min': {'mean': np.float64(281.9250471178535), 'std': np.float64(6.717682596268078)},
 'd2m_min_monthly': {'mean': np.float64(281.9250471178536), 'std': np.float64(6.032559739529602)},
 'prev1_rates': {'mean': np.float64(10.79785956173768), 'std': np.float64(43.112008287636684)},
 'prev2_rates': {'mean': np.float64(11.612144350888778), 'std': np.float64(44.37542122251766)},
 'swvl1_max': {'mean': np.float64(0.29610410984927854), 'std': np.float64(0.10422228128682841)},
 'swvl1_max_monthly': {'mean': np.float64(0.2961041098492786), 'std': np.float64(0.09293334959527232)},
 'swvl1_mean': {'mean': np.float64(0.2832375571494801), 'std': np.float64(0.09783093945518208)},
 'swvl1_mean_monthly': {'mean': np.float64(0.2832375571494802), 'std': np.float64(0.08904317627458175)},
 'swvl1_min': {'mean': np.float64(0.27235715195482263), 'std': np.float64(0.09463954334481062)},
 'swvl1_min_monthly': {'mean': np.float64(0.27235715195482263), 'std': np.float64(0.08698777240957146)},
 't2m_max': {'mean': np.float64(294.93095167563115), 'std': np.float64(7.01464467698358)},
 't2m_max_monthly': {'mean': np.float64(294.93095167563115), 'std': np.float64(6.133294140768315)},
 't2m_mean': {'mean': np.float64(290.35135916734384), 'std': np.float64(6.597812061078441)},
 't2m_mean_monthly': {'mean': np.float64(290.3513591673438), 'std': np.float64(6.069297236694833)},
 't2m_min': {'mean': np.float64(285.5049033248692), 'std': np.float64(6.641708638130881)},
 't2m_min_monthly': {'mean': np.float64(285.50490332486913), 'std': np.float64(6.228358454240875)},
 'tp_sum': {'mean': np.float64(0.03414772080722409), 'std': np.float64(0.07681575209861834)},
 'tp_sum_monthly': {'mean': np.float64(0.03414772080722409), 'std': np.float64(0.02819076641683726)},
 'u10_max': {'mean': np.float64(1.8433350926085077), 'std': np.float64(1.659576397958717)},
 'u10_max_monthly': {'mean': np.float64(1.8433350926085077), 'std': np.float64(0.9960918171811289)},
 'u10_mean': {'mean': np.float64(0.4067713737682111), 'std': np.float64(1.3322854121262353)},
 'u10_mean_monthly': {'mean': np.float64(0.40677137376821115), 'std': np.float64(0.6543109194592288)},
 'u10_min': {'mean': np.float64(-0.9275042631711495), 'std': np.float64(1.3958233897232277)},
 'u10_min_monthly': {'mean': np.float64(-0.9275042631711495), 'std': np.float64(0.6644806893471061)},
 'v10_max': {'mean': np.float64(1.2721509976660461), 'std': np.float64(1.882947526850702)},
 'v10_max_monthly': {'mean': np.float64(1.2721509976660461), 'std': np.float64(0.9861707675443273)},
 'v10_mean': {'mean': np.float64(-0.2866936532448027), 'std': np.float64(1.6414639980214825)},
 'v10_mean_monthly': {'mean': np.float64(-0.28669365324480267), 'std': np.float64(0.8078615301095902)},
 'v10_min': {'mean': np.float64(-1.7881945010268903), 'std': np.float64(1.7081765946881557)},
 'v10_min_monthly': {'mean': np.float64(-1.78819450102689), 'std': np.float64(0.9733566515781314)}}


def fit_normalization_stats(df, row_indices, fields, field_windows=None, allow_constant=False):
    """Fit z-score parameters using only the supplied training rows."""
    subset = df.iloc[np.asarray(row_indices)]
    fitted = {}
    for field in fields:
        if field not in subset:
            raise ValueError(f"Cannot fit normalization: raw field '{field}' is missing.")
        series = subset[field].dropna()
        if series.empty:
            if allow_constant:
                fitted[field] = {"mean": 0., "std": 1., "n_values": 0}
                continue
            raise ValueError(f"Cannot fit normalization: '{field}' has no training values.")
        sample = series.iloc[0]
        if isinstance(sample, (np.ndarray, list, tuple)):
            window = (field_windows or {}).get(field)
            arrays = [np.asarray(value, dtype=float).ravel() for value in series]
            if window and any(len(value) < window for value in arrays):
                raise ValueError(f"Insufficient training history for '{field}'")
            values = np.concatenate([value[-window:] if window else value for value in arrays])
        else:
            values = series.to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"Cannot fit normalization: '{field}' has no finite training values.")
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0 and allow_constant:
            std = 1.0
        if not np.isfinite(std) or std == 0:
            raise ValueError(f"Cannot fit normalization: '{field}' has zero/invalid training SD.")
        fitted[field] = {"mean": mean, "std": std, "n_values": int(values.size)}
    return fitted


def normalize_climate_data(
    df_: pd.DataFrame,
    field: str,
    normalization_stats: dict
) -> pd.DataFrame:
    """
    Z-score normalize a single climate field using precomputed mean & std.
    
    Adds a new column: f"{field}_znorm".
    
    Parameters
    ----------
    df_ : pd.DataFrame
        Input DataFrame containing the raw field values.
    field : str
        The exact column name to normalize (e.g. 'u10_min', 'tp_sum_monthly', …).
    normalization_stats : dict
        Mapping field → {'mean': float, 'std': float}.
    
    Returns
    -------
    pd.DataFrame
        The same DataFrame with an extra column '{field}_znorm'.
    """
    if field not in normalization_stats:
        raise ValueError(f"No normalization stats found for field '{field}'.")
    
    μ = normalization_stats[field]['mean']
    σ = normalization_stats[field]['std']
    if σ == 0 or np.isnan(σ):
        raise ValueError(f"Standard deviation is zero or NaN for field '{field}'.")
    
    df_ = df_.copy()
    df_[f"{field}_znorm"] = (df_[field] - μ) / σ
    return df_


def normalize_dataframe(
    df: pd.DataFrame,
    normalization_stats: dict = normalization_stats,
    fields_to_normalize: list[str] | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Apply Z-score normalization to a batch of climate fields.
    
    For each field in `fields_to_normalize` (or a default list), if that
    column exists in `df` and has a valid mean/std in normalization_stats,
    adds a '{field}_znorm' column to the output DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with raw climate fields.
    normalization_stats : dict
        Mapping each field name to its {'mean','std'}.
    fields_to_normalize : list[str], optional
        Exact list of column names to normalize. If None, uses the same
        default list you had before:
            [
                'tp_sum',
                'u10_min','u10_max','u10_mean',
                'v10_min','v10_max','v10_mean',
                'd2m_min','d2m_max','d2m_mean',
                't2m_min','t2m_max','t2m_mean',
                'swvl1_min','swvl1_max','swvl1_mean',
                'tp_sum_monthly',
                'u10_min_monthly','u10_max_monthly','u10_mean_monthly',
                'v10_min_monthly','v10_max_monthly','v10_mean_monthly',
                'd2m_min_monthly','d2m_max_monthly','d2m_mean_monthly',
                't2m_min_monthly','t2m_max_monthly','t2m_mean_monthly',
                'swvl1_min_monthly','swvl1_max_monthly','swvl1_mean_monthly',
                'prev_weeklyRate','prev2_weeklyRate'
            ]
    verbose : bool, default True
        If True, prints which fields were normalized and which were skipped.
    
    Returns
    -------
    pd.DataFrame
        A copy of `df` with added '{field}_znorm' columns for each applied field.
    """
    default_fields = [
        'tp_sum',
        'u10_min','u10_max','u10_mean',
        'v10_min','v10_max','v10_mean',
        'd2m_min','d2m_max','d2m_mean',
        't2m_min','t2m_max','t2m_mean',
        'swvl1_min','swvl1_max','swvl1_mean',
        'tp_sum_monthly',
        'u10_min_monthly','u10_max_monthly','u10_mean_monthly',
        'v10_min_monthly','v10_max_monthly','v10_mean_monthly',
        'd2m_min_monthly','d2m_max_monthly','d2m_mean_monthly',
        't2m_min_monthly','t2m_max_monthly','t2m_mean_monthly',
        'swvl1_min_monthly','swvl1_max_monthly','swvl1_mean_monthly',
        'prev1_rates','prev2_rates'
    ]
    fields = fields_to_normalize if fields_to_normalize is not None else default_fields

    df_norm = df.copy()
    applied, skipped = [], []

    for field in fields:
        if field not in df_norm.columns:
            skipped.append(f"{field} (missing)")
            continue
        if field not in normalization_stats:
            skipped.append(f"{field} (no stats)")
            continue

        try:
            df_norm = normalize_climate_data(df_norm, field, normalization_stats)
            applied.append(field)
        except ValueError as e:
            skipped.append(f"{field} (error: {e})")

    if verbose:
        print(f"Normalized {len(applied)} fields: {applied}")
        if skipped:
            print(f"Skipped {len(skipped)} fields: {skipped}")

    return df_norm



# Option A: with .apply()
def make_pair(x):
    if pd.notna(x):
        return np.array([x, 1.])
    else:
        return np.array([0, 0])

def create_flagged_weekly_rates(data_egg: pd.DataFrame, fields = ["prev1_rates", "prev2_rates", "prev1_rates_norm", "prev2_rates_norm", "prev1_rates_znorm", "prev2_rates_znorm",]) -> pd.DataFrame:
    """
    Create a DataFrame with flagged weekly rates for specified fields.
    
    Parameters
    ----------
    data_egg : pd.DataFrame
        Input DataFrame containing the weekly rates.
    fields : list of str, optional
        List of fields to process. Defaults to ["prev1_rates", "prev2_rates", "prev1_rates_norm", "prev2_rates_norm"].
    
    Returns
    -------
    pd.DataFrame
        DataFrame with new columns for each field, containing pairs of values.
    """
    maps_name = {
        "prev1_rates": "p1",
        "prev2_rates": "p2",
        "prev1_rates_norm": "p1n",
        "prev2_rates_norm": "p2n",
        "prev1_rates_znorm": "p1zn",
        "prev2_rates_znorm": "p2zn"
    }
    for field in fields:
        if field in data_egg.columns:
            data_egg[maps_name[field]] = data_egg[field].apply(make_pair)
    
    return data_egg
