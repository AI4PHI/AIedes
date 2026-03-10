import numpy as np
import pandas as pd
import geopandas as gpd
import torch
import argparse
import ast
import os

from AIedes.data_loader.classificator_data_loader import normalize_temperature_df, normalize_precipitation_df, get_data_loaders

from AIedes.train.classificator_train    import run_model
from AIedes.evaluation.classificator_evaluate import evaluate_model_nuts3, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train AIedes classificator")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=4*2048)
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=10)
    # input variables only two possible state temp or tp, can be both
    parser.add_argument("--input_variables", type=str, choices=["temp", "temp_tp", "4Bio"], default="temp_tp",
                        help="Input variables to use for the model. 'temp' for temperature, 'tp' for precipitation, 'both' for both.")  

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
    torch.set_num_interop_threads(params["num_threads"])  # Add this line
    
    print(f"PyTorch intra-op threads: {torch.get_num_threads()}")
    print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")

    # read the data

    #parent_dir = "/home/biazzin/git/mimesis/data/ourdata/copernicus_ecdc/"
    parent_dir = "./data/"
    year = "2020"
    df = pd.read_csv(parent_dir + f'albopictus_presence_absence_ecdc_world_clim_all_bio.zip')
    df_suitable = df
    
    # Filter data based on presence/absence and suitability
    df_Presence_Absence = df_suitable[(df_suitable["Presence_Absence"] == 1) | (df_suitable["Presence_Absence"] == 0)].copy()

    # Set presence/absence to 1 if suitable is 1, otherwise set to 0
    df_Presence_Absence["Presence_Absence"] = ((df_Presence_Absence['Presence_Absence'] == 1) & (df_Presence_Absence['Suitable'] == 1)).astype(int)

    # Convert to GeoDataFrame and filter absence points near presence
    df_gdf = gpd.GeoDataFrame(
        df_Presence_Absence,
        geometry=gpd.points_from_xy(df_Presence_Absence['Longitude'], df_Presence_Absence['Latitude'])
    )

    # Assuming `gdf` is your GeoDataFrame with a 'record' column that indicates 'presence' or 'absence'
    df_gdf.set_crs(epsg=4326, inplace=True)

    df_gdf = df_gdf.to_crs(epsg=3857)

    # Step 1: Split into presence and absence records
    presence_gdf = df_gdf[df_gdf['Presence_Absence'] == 1]
    absence_gdf = df_gdf[df_gdf['Presence_Absence'] == 0]

    # Step 2: Buffer presence points by 1 km (distance in meters; assumes CRS in meters)
    presence_buffered = presence_gdf.copy()
    presence_buffered['geometry'] = presence_buffered.geometry.buffer(1000)  # 1 km buffer
    print("Buffered presence points")
    # Step 3: Spatial join to find absence points within 1 km of any presence point
    near_presence = gpd.sjoin(absence_gdf, presence_buffered, how='inner', predicate='within')
    print("Absence points near presence points")
    # Step 4: Remove these absence points from the original GeoDataFrame
    filtered_gdf = df_gdf[~df_gdf.index.isin(near_presence.index)]


    df_Presence_Absence = filtered_gdf

    # Feature selection

    #features = ['annual_temp', 'annual_prec', 'max_temp_warmest_month', 'prec_warmest_quarter', 'TAVG_01', 'TAVG_02', 'TAVG_03', 'TAVG_04', 'TAVG_05', 'TAVG_06', 'TAVG_07', 'TAVG_08', 'TAVG_09', 'TAVG_10', 'TAVG_11', 'TAVG_12']
    features = ['annual_temp', 'annual_prec', 'max_temp_warmest_month', 'prec_warmest_quarter']

    if params["input_variables"] == "temp":
        features = ['TAVG_01', 'TAVG_02', 'TAVG_03', 'TAVG_04', 'TAVG_05', 'TAVG_06', 'TAVG_07', 
                    'TAVG_08', 'TAVG_09', 'TAVG_10', 'TAVG_11', 'TAVG_12']
    elif params["input_variables"] == "temp_tp":
        features = ['TAVG_01', 'TAVG_02', 'TAVG_03', 'TAVG_04', 'TAVG_05', 'TAVG_06', 'TAVG_07', 
                    'TAVG_08', 'TAVG_09', 'TAVG_10', 'TAVG_11', 'TAVG_12',
                    'annual_temp', 'annual_prec']
    elif params["input_variables"] == "4Bio":
        features = ['annual_temp', 'annual_prec', 'max_temp_warmest_month', 'prec_warmest_quarter']
    else:
        print(f"Invalid input_variables: {params['input_variables']}")
    
    print(f"Using features: {features}")


    # Settings (single split: 80% train, 20% test; no validation)
    batch_size = params["batch_size"]
    learning_rate = params["lr"]
    num_epochs = params["num_epochs"]
    test_ratio = 0.20
    val_ratio = 0.0
    random_seed = 42


    data = df_Presence_Absence[features + ["Presence_Absence"]].to_numpy()

    # Single split loaders (no validation)
    train_loader, test_loader, _, input_size = get_data_loaders(
        data=data,
        batch_size=batch_size,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed,
        shuffle=True,
        normalize=True
    )

    hidden_layers = params["layers"]

    # Train on TRAIN only; pass train_loader as dummy val_loader to satisfy signature
    print(f"\nArchitecture: {hidden_layers}")
    metrics_dict = {}
    model = run_model(
        input_size,
        learning_rate,
        train_loader,
        train_loader,      # dummy val (no validation phase)
        test_loader,
        df_Presence_Absence,
        hidden_layers=hidden_layers,
        num_epochs=num_epochs,
        features=features,
        metrics_dict=metrics_dict,
        
    )

    # Skip region-level evaluation during training experiments (it's slow!)
    # Uncomment only when you need regional metrics for final results
    # res_region = evaluate_model_nuts3(
    #     model,
    #     df_Presence_Absence,
    #     normalize=True,
    #     features=features
    # )
    res_region = {}  # Empty dict for now

    # Train/Test classification performance
    acc_train, tpr_train, tnr_train, fpr_train, fnr_train = evaluate_model(model, train_loader)
    acc_test,  tpr_test,  tnr_test,  fpr_test,  fnr_test  = evaluate_model(model, test_loader)

    # Save results; keep MEAN_/STD_ keys for compatibility (single split => STD=0)
    res = {
        "hidden_layers": hidden_layers,

        # Single-split lists
        "ACC_test_list": [acc_test],
        "TPR_test_list": [tpr_test],
        "TNR_test_list": [tnr_test],
        "FPR_test_list": [fpr_test],
        "FNR_test_list": [fnr_test],

        # Summary stats
        "MEAN_ACC_test": acc_test,
        "MEAN_TPR_test": tpr_test,
        "MEAN_TNR_test": tnr_test,
        "MEAN_FPR_test": fpr_test,
        "MEAN_FNR_test": fnr_test,
        "STD_ACC_test":  0.0,
        "STD_TPR_test":  0.0,
        "STD_TNR_test":  0.0,
        "STD_FPR_test":  0.0,
        "STD_FNR_test":  0.0,

        # Region-level metrics
        **res_region,

        # Train/Test metrics
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

        # Losses
        "train_losses": metrics_dict.get("train_losses", []),
        "val_losses":   metrics_dict.get("val_losses", []),  # may be empty
        "test_losses":  metrics_dict.get("test_losses", []),

        # metadata
        "type":            "NN",
        "num_params":      sum(p.numel() for p in model.parameters()),
        "num_epochs":      num_epochs,
        "test_ratio":      test_ratio,
        "val_ratio":       val_ratio,
        "learning_rate":   learning_rate,
        "random_seed":     random_seed,
        "input_variables": features,
        "input_size":      input_size,
        "model":           [model],
    }

    res_df = pd.DataFrame([res])

    import pickle
    with open(f"./results/results_and_models_{year}_{hidden_layers}_{params['input_variables']}.pkl", "wb") as f:
        pickle.dump(res_df, f)

if __name__ == "__main__":
    main()