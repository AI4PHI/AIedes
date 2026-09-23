import sys
from pathlib import Path

# Ensure src/ helpers and local data_creation are importable
TESTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(TESTS_DIR.parent / "src"))

import pandas as pd
import numpy as np
from data_creation import merge_dataframe, compute_previous_rates, normalize_dataframe

def make_pair(x):
    """Convert single value to [value, 1.0] pair, or [0, 0] for NaN."""
    if pd.notna(x):
        return np.array([x, 1.])
    else:
        return np.array([0, 0])

def create_eggs_dataset(input_path: str, output_dir: str = "./"):
    """
    Complete pipeline to create normalized egg dataset.
    
    Parameters:
    -----------
    input_path : str
        Path to input pickle file (e.g., "AIMSurv_albopictus_2020_era5_land.pkl")
    output_dir : str
        Directory to save output files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    data = pd.read_pickle(input_path)
    
    # Filter data
    print("Filtering data...")
    data_filtered = data[(data["keep"] == True) & (data["climate_nan"] == "no")]
    print(f"Filtered: {data_filtered.shape[0]} of {data.shape[0]} rows")
    
    # Select eggs only
    data_egg = data_filtered[data_filtered["lifeStage"] == "Egg"].copy()
    data_egg = data_egg.rename(columns={"weeklyRate": "weeklyRates"})
    
    # Column renaming mapping
    mapping = {
        "10m_v_component_of_wind": "v10",
        "10m_u_component_of_wind": "u10",
        "2m_temperature": "t2m",
        "2m_dewpoint_temperature": "d2m",
        "volumetric_soil_water_layer_1": "swvl1",
        "total_precipitation": "tp",
    }
    
    def rename_columns(col_name):
        postfixes = ['_mean', '_min', '_max', '_sum', 
                     '_mean_monthly', '_min_monthly', '_max_monthly', '_sum_monthly']
        for postfix in postfixes:
            if col_name.endswith(postfix):
                base_name = col_name.replace(postfix, '')
                if base_name in mapping:
                    return mapping[base_name] + postfix
        return col_name
    
    data_egg.columns = [rename_columns(col) for col in data_egg.columns]
    
    # Merge duplicates
    print("Merging duplicates...")
    data_egg = merge_dataframe(data_egg)
    
    # Compute previous rates
    print("Computing previous rates...")
    data_egg = compute_previous_rates(data_egg, verbose=True, delta_days=3)
    
    # Normalize
    print("Normalizing data...")
    data_egg = normalize_dataframe(data_egg)
    
    # Convert prev columns to [value, mask] pairs
    print("Creating prev value pairs...")
    # Keep prev1_rates and prev2_rates as raw values - they get normalized at runtime by counter_data_loader
    data_egg['prev1_rates'] = data_egg['prev_weeklyRates'].copy()
    data_egg['prev2_rates'] = data_egg['prev2_weeklyRates'].copy()
    data_egg['prev_weeklyRates'] = data_egg['prev_weeklyRates'].apply(make_pair)
    data_egg['prev2_weeklyRates'] = data_egg['prev2_weeklyRates'].apply(make_pair)
    data_egg['prev_weeklyRates_norm'] = data_egg['prev_weeklyRates_norm'].apply(make_pair)
    data_egg['prev2_weeklyRates_norm'] = data_egg['prev2_weeklyRates_norm'].apply(make_pair)
    
    # Save main dataset
    print("Saving datasets...")
    data_egg.to_pickle(str(output_dir / "eggs_y_norm.pkl"))
    
    # Create and save 7-day subset
    data_egg_7days = data_egg[
        (data_egg["samplingEffort Days"] >= 6) & 
        (data_egg["samplingEffort Days"] <= 8)
    ]
    data_egg_7days.to_pickle(str(output_dir / "eggs_y_norm_7days.pkl"))
    
    # Create and save 14-day subset
    data_egg_14days = data_egg[
        (data_egg["samplingEffort Days"] >= 13) & 
        (data_egg["samplingEffort Days"] <= 15)
    ]
    data_egg_14days.to_pickle(str(output_dir / "eggs_y_norm_14days.pkl"))
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total: {data_egg.shape[0]}")
    print(f"7-day subset: {data_egg_7days.shape[0]}")
    print(f"14-day subset: {data_egg_14days.shape[0]}")
    if len(data_egg_7days) > 0:
        print(f"Zeros in 7-day: {(data_egg_7days['weeklyRates'] == 0).sum() / len(data_egg_7days):.2%}")
    if len(data_egg_14days) > 0:
        print(f"Zeros in 14-day: {(data_egg_14days['weeklyRates'] == 0).sum() / len(data_egg_14days):.2%}")
    print(f"Zeros in total: {(data_egg['weeklyRates'] == 0).sum() / len(data_egg):.2%}")
    
    return data_egg, data_egg_7days, data_egg_14days

if __name__ == "__main__":
    import argparse

    _SRC_DIR = Path(__file__).parent.resolve()   # …/data/src
    _DATA_DIR = _SRC_DIR.parent                  # …/data

    parser = argparse.ArgumentParser(description="Create eggs normalized dataset")
    parser.add_argument(
        "--input", "-i",
        default=str(_DATA_DIR / "AIMSurv_albopictus_2020_era5_land.pkl"),
        help="Path to climate-enriched pickle file",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(_DATA_DIR),
        help="Directory to save output files (default: data/)",
    )
    args = parser.parse_args()

    create_eggs_dataset(input_path=args.input, output_dir=args.output_dir)