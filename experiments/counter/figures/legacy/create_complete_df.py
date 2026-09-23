"""
Aggregate and Filter Experiment Results
=======================================

This script recursively searches a given folder for experiment result files
stored as Pandas pickles (*.pkl), aggregates them into a single DataFrame,
computes summary statistics from training histories, filters out the lower-
performing half of models within each hyperparameter group, and saves the
filtered results for further analysis.

Main Features
-------------
1. Recursive file search:
   - Finds all `.pkl` files under the specified folder and subfolders.
   - Skips files whose name contains the substring "filtered".

2. Data aggregation:
   - Reads each pickle into a DataFrame.
   - Adds a `source_file` column to trace origin.
   - Concatenates all files into one combined DataFrame.

3. Metric summarization:
   - From each `*_history` column (e.g., train_loss_history),
     compute the mean of the last 10 values.
   - Produces new `*_end` columns such as `train_loss_end`,
     `test_r2_end`, `train_rmse_end`, etc.
   - Drops all raw `*_history` columns to save memory.

4. Filtering by performance:
   - Groups rows by: `input_variables`, `num_layers`,
     `hidden_dim`, and `dropout`.
   - Within each group, ranks models by `test_r2` (last epoch value).
   - Keeps only the top 25% of rows (rounded up).

5. Output:
   - Writes the filtered DataFrame to:
     `<results-folder>/filtered_results.pkl`
   - Prints summary statistics about group sizes and
     number of rows before/after filtering.

Command Line Usage
------------------
    python script.py --results-folder /path/to/results

Arguments
---------
--results-folder, -f : str or Path
    Path to the folder containing experiment `.pkl` files.
    Defaults to "./MSE".
    The same folder will also be used to write the output file.

Requirements
------------
- Python 3.8+
- pandas
- numpy

Example
-------
Suppose you have experiment pickles in `./MSE/` and subfolders:

    ./MSE/run1/results.pkl
    ./MSE/run2/results.pkl
    ./MSE/nested/run3/results.pkl

Run:

    python script.py -f ./MSE

This will:
    • Load all three files
    • Compute end-of-training metrics
    • Keep only the top half models in each group
    • Save results to `./MSE/filtered_results.pkl`
"""


import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import gc

def compute_last_ten_mean(arr, num=10, best=False):
    if isinstance(arr, (list, np.ndarray)) and len(arr) >= num:
        if best:
            sorted_arr = sorted(arr, reverse=True)
            return np.mean(sorted_arr[:num])
        else:
            return np.mean(arr[-num:])
    elif isinstance(arr, (list, np.ndarray)):
        return np.mean(arr)
    else:
        return np.nan

def main(results_folder: Path):
    """
    Process machine learning experiment results by combining pickle files and filtering top performers.
    This function:
    1. Recursively finds all .pkl files in the results folder (excluding filtered files)
    2. Loads and combines all DataFrames with source file tracking
    3. Computes end-of-training metrics from history columns using last 10 epochs
    4. Drops history columns to save memory
    5. Groups results by model configuration and keeps top 50% by test R² score
    6. Saves filtered results to 'filtered_results.pkl'
    Args:
        results_folder (Path): Directory containing experiment result pickle files
    Returns:
        None: Saves filtered DataFrame to disk and prints processing statistics
    Note:
        Requires compute_last_ten_mean function and pandas, numpy, gc imports.
        Creates various end metrics for loss, R², RMSE, accuracy, and classification rates.
    """
    
    results_folder.mkdir(parents=True, exist_ok=True)

    # find every *.pkl in this folder AND all subfolders, excluding "*filtered*"
    pkl_files = [p for p in results_folder.rglob("*.pkl") if "filtered" not in p.name]
    pkl_files.sort()

    if not pkl_files:
        print(f"No .pkl files found under: {results_folder.resolve()}")
        return

    dataframes = []
    for file in pkl_files:
        df = pd.read_pickle(file).reset_index(drop=True)
        df["source_file"] = file.name  # or str(file.relative_to(results_folder))
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    total_rows = combined_df.shape[0]
    print(f"Total number of rows in the combined DataFrame: {total_rows}")

    # create “_end” metrics from histories
    combined_df["train_loss_end"] = combined_df["train_loss_history"].apply(compute_last_ten_mean)
    combined_df["test_loss_end"] = combined_df["test_loss_history"].apply(compute_last_ten_mean)
    combined_df["train_r2_end"] = combined_df["train_r2_history"].apply(compute_last_ten_mean)
    combined_df["test_r2_end"] = combined_df["test_r2_history"].apply(compute_last_ten_mean)
    combined_df["test_rmse_end"] = combined_df["test_rmse_history"].apply(compute_last_ten_mean)
    combined_df["train_rmse_end"] = combined_df["train_rmse_history"].apply(compute_last_ten_mean)
    combined_df["train_log_r2_end"] = combined_df["train_log_r2_history"].apply(compute_last_ten_mean)
    combined_df["test_log_r2_end"] = combined_df["test_log_r2_history"].apply(compute_last_ten_mean)
    combined_df["train_nonzero_r2_end"] = combined_df["train_nonzero_r2_history"].apply(compute_last_ten_mean)
    combined_df["test_nonzero_r2_end"] = combined_df["test_nonzero_r2_history"].apply(compute_last_ten_mean)
    combined_df["train_binary_accuracy_end"] = combined_df["train_binary_accuracy_history"].apply(compute_last_ten_mean)
    combined_df["test_binary_accuracy_end"] = combined_df["test_binary_accuracy_history"].apply(compute_last_ten_mean)
    combined_df["train_FPR_end"] = combined_df["train_FPR_history"].apply(compute_last_ten_mean)
    combined_df["test_FPR_end"] = combined_df["test_FPR_history"].apply(compute_last_ten_mean)
    combined_df["train_FNR_end"] = combined_df["train_FNR_history"].apply(compute_last_ten_mean)
    combined_df["test_FNR_end"] = combined_df["test_FNR_history"].apply(compute_last_ten_mean)
    combined_df["train_TPR_end"] = combined_df["train_TPR_history"].apply(compute_last_ten_mean)
    combined_df["test_TPR_end"] = combined_df["test_TPR_history"].apply(compute_last_ten_mean)
    combined_df["train_TNR_end"] = combined_df["train_TNR_history"].apply(compute_last_ten_mean)
    combined_df["test_TNR_end"] = combined_df["test_TNR_history"].apply(compute_last_ten_mean)

    # drop history columns to save memory
    history_cols = [c for c in combined_df.columns if "history" in c]
    combined_df.drop(columns=history_cols, inplace=True)
    del history_cols
    gc.collect()

    # group + keep top half by test_r2_end
    grp_cols = ["input_variables", "num_layers", "hidden_dim", "dropout"]
    original_group_sizes = combined_df.groupby(grp_cols).size()
    print("Original group sizes:")
    print(original_group_sizes.value_counts().sort_index())

    unique_original_sizes = original_group_sizes.unique()
    if len(unique_original_sizes) == 1:
        print(f"\n✓ All original groups have exactly {unique_original_sizes[0]} rows")
        print(f"After filtering (top 25%), each group will have {int(np.ceil(unique_original_sizes[0] / 4))} rows")
    else:
        print(f"\n✗ Original groups have different sizes: {sorted(unique_original_sizes)}")
        print("\nDetailed breakdown:")
        for size in sorted(unique_original_sizes):
            groups_with_size = original_group_sizes[original_group_sizes == size]
            print(f"  {len(groups_with_size)} group(s) with {size} rows each")
            if len(groups_with_size) <= 5:
                print(f"    Examples: {groups_with_size.index.tolist()[:3]}")

    total_combinations = len(original_group_sizes)
    print(f"\nTotal number of unique combinations: {total_combinations}")

    mask = (
        combined_df
        .groupby(grp_cols)["test_r2"]
        .transform(lambda x: x.rank(method="first", ascending=False) <= np.ceil(len(x) / 4))
    )

    filtered_df = combined_df[mask].reset_index(drop=True)
    total_filtered_rows = filtered_df.shape[0]
    print(f"Total number of rows in the filtered DataFrame: {total_filtered_rows} from the total of {total_rows}")

    # Save alongside inputs (same folder acts as output)
    out_path = results_folder / "filtered_results.pkl"
    filtered_df.to_pickle(out_path)
    print(f"Filtered results saved to: {out_path.resolve()}")
    print(f"Filtered DataFrame shape: {filtered_df.shape}")
    print(f"Filtered DataFrame columns: {filtered_df.columns.tolist()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate and filter experiment results.")
    parser.add_argument(
        "--results-folder",
        "-f",
        type=Path,
        default=Path("./MSE"),
        help="Folder to search recursively for .pkl files (also used as output folder).",
    )
    args = parser.parse_args()
    main(args.results_folder)
