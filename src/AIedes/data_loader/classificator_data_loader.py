# data_loader.py

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

def normalize_temperature_df(df, mean_temp=9.3578, std_temp=8.02477):
    """
    Normalize the temperature columns in the DataFrame using mean and standard deviation.
    """
    temp_cols = [col for col in df.columns if 'temp_' in col]
    print("temp_cols", temp_cols)
    if not temp_cols:
        print("No temperature columns found to normalize.")
        return df
    
    # Calculate mean and std if not provided
    if mean_temp is None or std_temp is None:
        temp_values = df[temp_cols].values.flatten()
        mean_temp = temp_values.mean()
        std_temp = temp_values.std()
        print(f"Calculated mean_temp: {mean_temp}, std_temp: {std_temp}")
    
    # Normalize each temperature column
    for col in temp_cols:
        df[col] = (df[col] - mean_temp) / std_temp
    print("Temperature columns normalized with mean and std.")
    return df

def normalize_precipitation_df(df, mean_precip=96.20395, std_precip=79.6608):
    """
    Normalize the precipitation columns in the DataFrame using mean and standard deviation.
    """
    precip_cols = [col for col in df.columns if 'precip_' in col]
    print("precip_cols", precip_cols)
    if not precip_cols:
        print("No precipitation columns found to normalize.")
        return df
    
    # Calculate mean and std if not provided
    if mean_precip is None or std_precip is None:
        precip_values = df[precip_cols].values.flatten()
        mean_precip = precip_values.mean()
        std_precip = precip_values.std()
        print(f"Calculated mean_precip: {mean_precip}, std_precip: {std_precip}")
    
    # Normalize each precipitation column
    for col in precip_cols:
        df[col] = (df[col] - mean_precip) / std_precip
    print("Precipitation columns normalized with mean and std.")
    return df

class MosquitoDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X  # Torch tensor, already standardized
        self.y = torch.tensor(y, dtype=torch.float)  # Labels for classification
        self.transform = transform  # Transform to apply (if any)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.transform:
            sample = self.transform(sample)
        return self.X[idx], self.y[idx]

def load_data(filename):
    # Load data from compressed numpy file
    data = np.load(filename)['data']
    
    # Features and labels
    X = data[:, :-1]  # First N-1 columns: features
    y = data[:, -1]   # Last column: labels
    
    return X, y

def split_data(X, y, test_ratio=0.2,random_seed=42, shuffle=True):
    # Set the seed for reproducibility
    np.random.seed(random_seed)
    
    # Shuffle the data
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    # Split the data
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test):
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_train_std = X_train
    X_test_std = X_test

    # Compute mean and std from training data
    mean = torch.mean(X_train, dim=0)
    std = torch.std(X_train, dim=0)
    
    # Avoid division by zero
    std[std == 0] = 1.0
    
    # Standardize the data
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std
    
    return X_train_std, X_test_std

def split_data_by_nuts3_regions(NUTS3_codes, test_ratio=0.2, val_ratio=0.0, random_seed=42):
    """
    Split data by NUTS3 regions instead of randomly.
    
    First splits NUTS3 regions into train/test (and optionally val).
    Then collects all data point indices belonging to each region group.
    
    Args:
        NUTS3_codes: array of NUTS3 region codes for each data point
        test_ratio: proportion of regions to assign to test set
        val_ratio: proportion of regions to assign to validation set
        random_seed: for reproducibility
    
    Returns:
        train_idx, test_idx, val_idx: indices of points belonging to each region group
    """
    np.random.seed(random_seed)
    
    # Get unique NUTS3 regions
    unique_regions = np.unique(NUTS3_codes)
    n_regions = len(unique_regions)
    
    # Shuffle regions
    shuffled_regions = unique_regions.copy()
    np.random.shuffle(shuffled_regions)
    
    # Split regions by ratios
    n_test_regions = max(1, int(n_regions * test_ratio))
    n_val_regions = max(1, int(n_regions * val_ratio)) if val_ratio > 0 else 0
    n_train_regions = n_regions - n_test_regions - n_val_regions
    
    train_regions = shuffled_regions[:n_train_regions]
    test_regions = shuffled_regions[n_train_regions:n_train_regions + n_test_regions]
    val_regions = shuffled_regions[n_train_regions + n_test_regions:] if n_val_regions > 0 else np.array([])
    
    print(f"Total regions: {n_regions}")
    print(f"Train regions: {n_train_regions}, Test regions: {n_test_regions}, Val regions: {n_val_regions}")
    
    # Get indices for each region group
    train_idx = np.where(np.isin(NUTS3_codes, train_regions))[0]
    test_idx = np.where(np.isin(NUTS3_codes, test_regions))[0]
    val_idx = np.where(np.isin(NUTS3_codes, val_regions))[0] if n_val_regions > 0 else None
    
    print(f"Train points: {len(train_idx)}, Test points: {len(test_idx)}, Val points: {len(val_idx) if val_idx is not None else 0}")
    
    return train_idx, test_idx, val_idx


def get_data_loaders(
    data=None,
    NUTS3_codes=None,
    filename=None,
    batch_size=1024,
    test_ratio=0.2,
    val_ratio=0.0,  # NEW
    random_seed=42,
    num_workers=4,
    shuffle=True,
    normalize=False
):
    from sklearn.model_selection import train_test_split

    # Load and preprocess data
    if data is not None:
        X = data[:, :-1]  # First N-1 columns: features
        y = data[:, -1]   # Last column: labels
    elif filename is not None:
        X, y = load_data(filename)
    else:
        raise ValueError('No data provided. Please provide either a NumPy array or a filename.')

    if data is not None and filename is not None:
        raise ValueError('Both data and filename provided. Please provide only one source of data.')

    # Special case: test_ratio == 1 → full dataset as both train & test
    if test_ratio == 1:
        # normalize if requested
        if normalize:
            _, X_std = standardize_data(X, X)
        else:
            X_std = torch.tensor(X, dtype=torch.float32)

        y_tensor = torch.tensor(y, dtype=torch.float32)
        full_dataset = MosquitoDataset(X_std, y_tensor)

        train_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=shuffle,       # respects the caller's shuffle flag
            num_workers=num_workers
        )
        test_loader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            shuffle=False,         # always in-order for "test"
            num_workers=num_workers
        )

        # return train, test, no-val, and input size
        return train_loader, test_loader, None, X_std.shape[1]


    if test_ratio + val_ratio >= 1.0:
        raise ValueError("The sum of test_ratio and val_ratio must be less than 1.")

    # ── NEW: Split by NUTS3 regions if provided ──
    if NUTS3_codes is not None:
        train_idx, test_idx, val_idx = split_data_by_nuts3_regions(
            NUTS3_codes, 
            test_ratio=test_ratio, 
            val_ratio=val_ratio, 
            random_seed=random_seed
        )
        
        # Extract data for each split using region-based indices
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_val, y_val = (X[val_idx], y[val_idx]) if val_idx is not None and len(val_idx) > 0 else (None, None)
    else:
        # ── EXISTING: Random split logic ──
        # Split data
        X_temp, X_test, y_temp, y_test = split_data(X, y, test_ratio=test_ratio, random_seed=random_seed, shuffle=shuffle)
        if val_ratio > 0:
            val_ratio_adjusted = val_ratio / (1 - test_ratio)  # proportion of temp to allocate to val
            X_train, X_val, y_train, y_val = split_data(X_temp, y_temp, test_ratio=val_ratio_adjusted, random_seed=random_seed, shuffle=shuffle)
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None

    # Normalize
    if normalize:
        X_train_std, X_test_std = standardize_data(X_train, X_test)
        if X_val is not None:
            _, X_val_std = standardize_data(X_train, X_val)
        else:
            X_val_std = None
    else:
        X_train_std = torch.tensor(X_train, dtype=torch.float32)
        X_test_std = torch.tensor(X_test, dtype=torch.float32)
        X_val_std = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None

    # Convert labels
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32) if y_val is not None else None

    # Datasets and loaders
    train_dataset = MosquitoDataset(X_train_std, y_train)
    test_dataset = MosquitoDataset(X_test_std, y_test)
    val_dataset = MosquitoDataset(X_val_std, y_val) if X_val is not None else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers) if val_dataset else None

    return train_loader, test_loader, val_loader, X_train_std.shape[1]
