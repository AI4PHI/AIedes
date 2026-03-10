import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import argparse
import ast


from AIedes.data_loader.classificator_data_loader import normalize_temperature_df, normalize_precipitation_df, get_data_loaders

from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from AIedes.train.classificator_train    import run_model
from AIedes.evaluation.classificator_evaluate import evaluate_model_nuts3, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train AIedes classificator")
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=10)
    # input variables only two possible state temp or tp, can be both
    parser.add_argument("--input_variables", type=str, choices=["temp", "tp", "temp_tp"], default="temp_tp",
                        help="Input variables to use for the model. 'temp' for temperature, 'tp' for precipitation, 'both' for both.")  
    
    parser.add_argument("--use_nuts3_split", type=bool, default=False,
                        help="Use NUTS3 region-based splitting instead of random splitting")

    parser.add_argument(
        "--layers",
        type=ast.literal_eval,   # parses one Python literal
        help=(
            "List of layers sizes, e.g. `--layers 64 32` creates a model with [64, 32] hidden layers. "
        ),
        default=[]  # default is no hidden layers
    )

    args = parser.parse_args()
    params = vars(args)
    print(f"Parameters: {params}")

    torch.set_num_threads(params["num_threads"])


    # read the data

    #parent_dir = "/home/biazzin/git/mimesis/data/ourdata/copernicus_ecdc/"
    parent_dir = "./data/"
    year = "2020"
    df = pd.read_csv(parent_dir + f'ecdc_albopictus_cordex_{year}.zip')
    df_suitable = df
    df_Presence_Absence = (df_suitable[(df_suitable["presence_numeric"] == 1) | (df_suitable["presence_numeric"] == 0)]).copy()
    df_Presence_Absence["NUTS3_Code"] = df_Presence_Absence["LocationCode"]


    # set presence absence to 1 if suitable is 1, otherwise set to 0
    df_Presence_Absence["Presence_Absence"] = ((df_Presence_Absence['presence_numeric'] == 1) & (df_Presence_Absence['Suitable'] == 1)).astype(int)

    #normalize the data

    df_Presence_Absence = normalize_temperature_df(df_Presence_Absence)
    df_Presence_Absence = normalize_precipitation_df(df_Presence_Absence)


    temperature_features = ['temp_Jan_C', 'temp_Feb_C', 'temp_Mar_C', 'temp_Apr_C',
                            'temp_May_C', 'temp_Jun_C', 'temp_Jul_C', 'temp_Aug_C',
                            'temp_Sep_C', 'temp_Oct_C', 'temp_Nov_C', 'temp_Dec_C']
    precipitation_features = ['precip_Jan_mm', 'precip_Feb_mm', 'precip_Mar_mm', 
                            'precip_Apr_mm', 'precip_May_mm', 'precip_Jun_mm', 
                            'precip_Jul_mm', 'precip_Aug_mm', 'precip_Sep_mm', 
                            'precip_Oct_mm', 'precip_Nov_mm', 'precip_Dec_mm']
    # Select features for the model
    if params["input_variables"] == "temp":
        features = temperature_features
    elif params["input_variables"] == "tp":
        features = precipitation_features
    elif params["input_variables"] == "temp_tp":
        # Use both temperature and precipitation features
        # Note: This is the default behavior, so no need to check for it explicitly
        # If both are selected, combine the two lists
        features = temperature_features + precipitation_features

    data = df_Presence_Absence[features + ["Presence_Absence"]].to_numpy()



    # Settings
    batch_size = params["batch_size"]
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    test_ratio = 0.15
    val_ratio = 0.15
    random_seed = 42
    
    # Determine NUTS3 codes if using NUTS3-based split
    nuts3_codes = df_Presence_Absence["NUTS3_Code"].values if params["use_nuts3_split"] else None
    split_type = "nuts3" if params["use_nuts3_split"] else "random"


    # ── Hyperparameters ────────────────────────────────────────────────
    n_folds       = 5

    # ── 1) Get full train+val loader (no built-in val split) & test loader
    train_loader_full, test_loader, _, input_size = get_data_loaders(
        data=data,
        NUTS3_codes=nuts3_codes,
        batch_size=batch_size,
        test_ratio=0.2,  # ← CHANGE FROM 0.0 TO 0.15
        val_ratio=0.0,
        random_seed=random_seed,
        shuffle=True,
        normalize=False
    )

    # ── 2) Prepare for stratified k-fold on the full train dataset
    full_dataset = train_loader_full.dataset
    labels = [full_dataset[i][1].item() for i in range(len(full_dataset))]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    hidden_layers = params["layers"]

    # ── 3) perform k-fold CV ────────────

    print(f"\nArchitecture: {hidden_layers}")
    fold_records = []

    # 3a) K-fold splits
    for fold, (train_idx, test_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels), start=1):

        # build per-fold train/val loaders
        train_loader = DataLoader(
            Subset(full_dataset, train_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        test_loader = DataLoader(
            Subset(full_dataset, test_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

        # 3b) Train & validate this fold
        metrics_dict_cv = {}
        model_cv = run_model(
            input_size,
            learning_rate,
            train_loader,
            test_loader,
            test_loader,
            df_Presence_Absence,
            hidden_layers=hidden_layers,
            num_epochs=num_epochs,
            features=features,
            metrics_dict=metrics_dict_cv
        )

        # compute validation metrics for this fold
        acc_test, tpr_test, tnr_test, fpr_test, fnr_test = evaluate_model(model_cv, test_loader)
        fold_records.append({
            "fold":    fold,
            "ACC_test": acc_test,
            "TPR_test": tpr_test,
            "TNR_test": tnr_test,
            "FPR_test": fpr_test,
            "FNR_test": fnr_test,
        })

    # ── 4) Aggregate per-fold metrics ────────────────────────────────
    df_folds  = pd.DataFrame(fold_records)
    # raw lists of validation metrics
    acc_test_list = df_folds["ACC_test"].tolist()
    tpr_test_list = df_folds["TPR_test"].tolist()
    tnr_test_list = df_folds["TNR_test"].tolist()
    fpr_test_list = df_folds["FPR_test"].tolist()
    fnr_test_list = df_folds["FNR_test"].tolist()
    # summary stats
    mean_tests = df_folds.mean().add_prefix("MEAN_")
    std_tests  = df_folds.std(). add_prefix("STD_")

    # ── 5) Retrain on the full train+val set ────────────────────────
    metrics_dict_full = {}
    model_full = run_model(
        input_size,
        learning_rate,
        train_loader_full,
        train_loader_full,   # dummy val_loader to satisfy signature
        test_loader,
        df_Presence_Absence,
        hidden_layers=hidden_layers,
        num_epochs=num_epochs,
        features=features,
        metrics_dict=metrics_dict_full
    )

    # ── 6) Region‐level evaluation ─────────────────────────────────
    res_region = evaluate_model_nuts3(
        model_full,
        df_Presence_Absence,
        normalize=True,
        features=features
    )

    # ── 7) Train/Test classification performance ───────────────────
    acc_train, tpr_train, tnr_train, fpr_train, fnr_train = evaluate_model(model_full, train_loader_full)
    acc_test,  tpr_test,  tnr_test,  fpr_test,  fnr_test  = evaluate_model(model_full, test_loader)

    # ── 8) Compile results for this architecture ───────────────────
    res = {
        "hidden_layers":    hidden_layers,
        # CV raw validation lists
        "ACC_test_list":     acc_test_list,
        "TPR_test_list":     tpr_test_list,
        "TNR_test_list":     tnr_test_list,
        "FPR_test_list":     fpr_test_list,
        "FNR_test_list":     fnr_test_list,
        # CV summary stats
        **mean_tests.to_dict(),
        **std_tests.to_dict(),
        # region-level metrics
        **res_region,
        # classification metrics on full train+val & test
        "ACC":    acc_train,
        "TPR":    tpr_train,
        "TNR":    tnr_train,
        "FPR":    fpr_train,
        "FNR":    fnr_train,
        "ACC_test": acc_test,
        "TPR_test": tpr_test,
        "TNR_test": tnr_test,
        "FPR_test": fpr_test,
        "FNR_test": fnr_test,
        # losses from full training
        "train_losses": metrics_dict_full["train_losses"],
        "val_losses":   metrics_dict_full.get("val_losses", []),
        "test_losses":  metrics_dict_full["test_losses"],
        # metadata
        "type":            "NN",
        "num_params":      sum(p.numel() for p in model_full.parameters()),
        "num_epochs":      num_epochs,
        "test_ratio":      test_ratio,
        "val_ratio":       0.0,
        "learning_rate":   learning_rate,
        "random_seed":     random_seed,
        "input_variables": features,
        "input_size":      input_size,
        "split_type":      split_type,
        "model":           [model_full],
    }


    res_df = pd.DataFrame([res])

    import pickle
    with open(f"./results/results_and_models_{year}_{hidden_layers}_{params['input_variables']}_{split_type}.pkl", "wb") as f:
        pickle.dump(res_df, f)

if __name__ == "__main__":
    main()