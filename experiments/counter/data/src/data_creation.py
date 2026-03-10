import numpy as np
import pandas as pd

def compute_previous_rates(
    df: pd.DataFrame,
    id_col: str = "id_trap",
    date_col: str = "end_date",
    rate_col: str = "weeklyRates",
    effort_col: str = "samplingEffort Days",
    verbose: bool = True,
    delta_days: int = 1
) -> pd.DataFrame:
    """
    For each row in `df`, finds the 1-period-ago and 2-periods-ago measurements
    of `rate_col` (within +/-1 day of the implied start date), and stores them
    in new columns 'prev_<rate_col>' and 'prev2_<rate_col>'.  If multiple
    candidates are found, takes their mean.

    If verbose=True, prints counts of how many were found/missing and how many
    times a mean was used.

    Returns a new DataFrame with the added columns.
    """
    df = df.copy()
    prev_col  = f"prev_{rate_col}"
    prev2_col = f"prev2_{rate_col}"
    df[prev_col]  = np.nan
    df[prev2_col] = np.nan

    multiple_prev  = 0
    multiple_prev2 = 0

    for i, row in df.iterrows():
        trap = row[id_col]
        end  = row[date_col]
        eff  = row[effort_col]
        start = end - pd.Timedelta(days=eff)

        # 1-period-ago window
        w0 = start - pd.Timedelta(days=delta_days)
        w1 = start + pd.Timedelta(days=delta_days)
        prev_rows = df[
            (df[id_col] == trap) &
            (df.index != i) &
            (df[date_col] > w0) &
            (df[date_col] < w1)
        ][rate_col]

        if len(prev_rows) == 1:
            df.at[i, prev_col] = prev_rows.iloc[0]
        elif len(prev_rows) > 1:
            df.at[i, prev_col] = prev_rows.mean()
            multiple_prev += 1

        # 2-periods-ago window (2*eff days before start, ±1 day)
        prev2_center = end - pd.Timedelta(days=2*eff)
        w0 = prev2_center - pd.Timedelta(days=delta_days)
        w1 = prev2_center + pd.Timedelta(days=delta_days)
        prev2_rows = df[
            (df[id_col] == trap) &
            (df.index != i) &
            (df[date_col] > w0) &
            (df[date_col] < w1)
        ][rate_col]

        if len(prev2_rows) == 1:
            df.at[i, prev2_col] = prev2_rows.iloc[0]
        elif len(prev2_rows) > 1:
            df.at[i, prev2_col] = prev2_rows.mean()
            multiple_prev2 += 1

    if verbose:
        total = len(df)
        found1 = df[prev_col].notna().sum()
        miss1  = df[prev_col].isna().sum()
        found2 = df[prev2_col].notna().sum()
        miss2  = df[prev2_col].isna().sum()

        print(f"Total rows:                                  {total}")
        print(f"Entries with prev       measurements found: {found1}")
        print(f"Entries with prev       measurements missing: {miss1}")
        print(f"Used mean for >1 prev  candidates:           {multiple_prev}")
        print(f"Entries with prev2      measurements found: {found2}")
        print(f"Entries with prev2      measurements missing: {miss2}")
        print(f"Used mean for >1 prev2 candidates:           {multiple_prev2}")

    return df


def merge_dataframe(
    df: pd.DataFrame,
    group_keys: list[str] = ["id_trap", "start_date", "end_date"],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Groups the DataFrame by `group_keys`, merging array‐like columns by element‐wise mean,
    numeric columns by mean, and all others by taking the first value.
    
    Prints summary statistics if verbose=True, including how many columns used each
    aggregation (“mean”, “first”, or the array‐merging helper).
    
    Returns the merged DataFrame.
    """
    # 1) helper for array/list columns
    def _merge_arrays(arrays):
        try:
            return np.mean(np.stack(arrays), axis=0)
        except Exception:
            # fallback if something goes wrong
            return arrays.iloc[0]
    
    # 2) build aggregation rules
    agg_rules: dict[str, object] = {}
    mean_count = 0
    first_count = 0
    array_count = 0

    for col in df.columns:
        if col in group_keys:
            continue
        non_null = df[col].dropna()
        sample = non_null.iloc[0] if not non_null.empty else None
        
        if isinstance(sample, (np.ndarray, list)):
            agg_rules[col] = _merge_arrays
            array_count += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            agg_rules[col] = "mean"
            mean_count += 1
        else:
            agg_rules[col] = "first"
            first_count += 1
    
    # 3) compute stats before merge
    initial_rows = len(df)
    group_size = df.groupby(group_keys).size()
    
    # 4) perform groupby + aggregation
    merged = (
        df
        .groupby(group_keys)
        .agg(agg_rules)
        .reset_index()
    )
    
    # 5) compute and (optionally) print summary
    final_rows = len(merged)
    total_merged = initial_rows - final_rows
    multi_input_groups = (group_size > 1).sum()
    
    if verbose:
        print(f"Initial rows:                     {initial_rows}")
        print(f"Final rows after merge:           {final_rows}")
        print(f"Total entries merged:             {total_merged}")
        print(f"Groups with >1 records:           {multi_input_groups}")
        if multi_input_groups:
            print("\nGroup‐size distribution for groups merged (>1 record):")
            print(group_size[group_size > 1].describe(), "\n")
        
        # new summary of agg‐rules
        print("Aggregation rules applied per column:")
        print(f"  • array‐merge helper (_merge_arrays): {array_count}")
        print(f"  • numeric mean:                       {mean_count}")
        print(f"  • first‐value fallback:               {first_count}")
    
    return merged


normalization_values = {
 'prev_weeklyRates': {'min': 0.0, 'max': 994.0},
 'prev2_weeklyRates': {'min': 0.0, 'max': 994.0},
 'v10': {'min': -16.140758733663443, 'max': 14.163530544568607},
 'u10': {'min': -10.843626718824694, 'max': 12.500118531742228},
 't2m': {'min': 255.58061066606894, 'max': 315.1352098562941},
 'd2m': {'min': 252.21284308891344, 'max': 299.0503128245249},
 'swvl1': {'min': 0.04799544618504019, 'max': 0.515732518641216},
 'tp': {'min': -1.332267676062776e-15, 'max': 1.7602644567198311},
 'prev': {'min': 0.0, 'max': 634.0},
 'prev2': {'min': 0.0, 'max': 634.0}}

def normalize_climate_data(df_, field, normalization_values = normalization_values):
    """
    Normalize a given climate data field using precomputed min and max values.
    
    Parameters:
    df_ (pd.DataFrame): The input dataframe containing the climate data.
    field (str): The field to normalize.
    normalization_values (dict): A dictionary with min and max values for each base field.
    
    Returns:
    pd.DataFrame: The dataframe with additional normalized columns.
    """
    base_field = field.split('_')[0]  # Extract base field name before min, max, mean, or sum
    
    if base_field not in normalization_values:
        raise ValueError(f"Unknown base field: {base_field}")
    
    min_, max_ = normalization_values[base_field]['min'], normalization_values[base_field]['max']
    
    if min_ == max_:
        raise ValueError(f"Min and max values are the same for field: {base_field}, normalization not possible.")
    
    df_[f'{field}_norm'] = (df_[field] - min_) / (max_ - min_)
    return df_


def normalize_dataframe(
    df: pd.DataFrame,
    normalization_values: dict = normalization_values,
    fields_to_normalize: list[str] | None = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Applies `normalize_climate_data(df, field, normalization_values)` to each
    field in `fields_to_normalize` (or a sensible default list) if that column
    exists in `df`. Returns a new DataFrame with all requested normalizations applied.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    normalization_values : dict
        Mapping from field names to whatever parameters your
        `normalize_climate_data` needs (e.g. mean/std dict).
    fields_to_normalize : list[str], optional
        Exactly which columns to normalize. If None, uses the default list:
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
                'prev_weeklyRates','prev2_weeklyRates'
            ]
    verbose : bool, default True
        If True, prints how many fields were normalized and which were missing.

    Returns
    -------
    pd.DataFrame
        A copy of `df` with each requested column passed through
        `normalize_climate_data`.
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
        'prev_weeklyRates','prev2_weeklyRates'
    ]
    fields = fields_to_normalize if fields_to_normalize is not None else default_fields

    df_norm = df.copy()
    applied = []
    skipped = []

    for field in fields:
        if field in df_norm.columns:
            df_norm = normalize_climate_data(df_norm, field, normalization_values)
            applied.append(field)
        else:
            skipped.append(field)

    if verbose:
        print(f"Normalized {len(applied)} fields: {applied}")
        if skipped:
            print(f"Skipped {len(skipped)} missing fields: {skipped}")

    return df_norm
