import os
import pandas as pd
import torch
import numpy as np

from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_utils import get_original_feature_labels, get_renamed_feature_labels, get_delete_feature_labels


def _preprocess_data(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    original_feature_labels = get_original_feature_labels()
    renamed_feature_labels = get_renamed_feature_labels()
    delete_feature_labels = get_delete_feature_labels()

    df = df.drop(columns=delete_feature_labels)
    
    rename_dict = {
        k: v for k, v in zip(original_feature_labels, renamed_feature_labels)
    }
    df = df.rename(columns=rename_dict)

    labels = df[label_column].unique()
    
    for label in labels:
        if "Attempted" in label:
            df.loc[df[label_column] == label, label_column] = "BENIGN"
        if "Web Attack" in label:
            df.loc[df[label_column] == label, label_column] = "Web Attack"
        if "Infiltration" in label:
            df.loc[df[label_column] == label, label_column] = "Infiltration"
        if "DoS" in label and label != "DDoS":
            df.loc[df[label_column] == label, label_column] = "DoS"
    
    return df


def _load_data(data_path: str, label_column: str) -> pd.DataFrame:
    print(f"Loading data from {data_path}...")
    files = glob(os.path.join(data_path, "*.csv.gz"))
    print(f"Found {len(files)} files.")
    dfs = [pd.read_csv(file) for file in tqdm(files, desc="Loading data")]
    df = pd.concat(dfs)
    # df = _preprocess_data(df, label_column)
    return df


class CICIDS2017ImprovedPresent(Dataset):

    def __init__(self, data_path: str, label_column: str, transform=None):
        self.data_path = data_path
        self.label_column = label_column
        self.transform = transform

        self.data = _load_data(data_path, label_column)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data.drop(columns=[self.label_column]).values[idx], self.data[self.label_column].values[idx]


class CICIDS2017Improved(Dataset):
    base_folder = "CICIDS2017_improved"
    
    # Define class mapping from string labels to numeric IDs
    CLASS_MAPPING = {
        'BENIGN': 0,
        'Botnet': 1,
        'DDoS': 2,
        'DoS': 3,
        'FTP-Patator': 4,
        'Heartbleed': 5,
        'Infiltration': 6,
        'Portscan': 7,
        'SSH-Patator': 8,
        'Web Attack': 9
    }

    def __init__(
        self, root, train=True, transform=None, target_transform=None,
        download=False, index=None, base_sess=None, autoaug=1, max_samples=None
    ):
        super(CICIDS2017Improved, self).__init__()
        print(f"index: {index}")

        self.root = os.path.expanduser(root)
        self.train = train

        # ignore download, autoaug parameter
        data_path = os.path.join(self.root, self.base_folder)
        train_data_path = os.path.join(data_path, "train")
        test_data_path = os.path.join(data_path, "test")

        if self.train:
            self.data = _load_data(train_data_path, "Label")
        else:
            self.data = _load_data(test_data_path, "Label")
        
        # Debug: Check original labels before mapping
        print(f"Original label distribution (before mapping):")
        print(self.data["Label"].value_counts())
        
        # Save original labels for debugging
        original_labels = self.data["Label"].copy()
        
        # Check if labels are already numeric or need mapping
        # Check if labels are numeric by trying to convert and checking if any are NaN
        numeric_labels = pd.to_numeric(self.data["Label"], errors='coerce')
        num_numeric = (~numeric_labels.isna()).sum()
        num_total = len(self.data["Label"])
        
        # If more than 95% of labels can be converted to numeric, treat as numeric
        is_numeric = (num_numeric / num_total) > 0.95 if num_total > 0 else False
        
        if is_numeric:
            # Labels are already numeric, just ensure they are integers
            print(f"Labels are already numeric ({num_numeric}/{num_total} are numeric), skipping mapping")
            self.data["Label"] = pd.to_numeric(self.data["Label"], errors='coerce')
            # Drop any rows that couldn't be converted (should be very few)
            self.data = self.data.dropna(subset=["Label"]).reset_index(drop=True)
            # Now convert to int after dropping NaN
            self.data["Label"] = self.data["Label"].astype(int)
        else:
            # Labels are strings, need to map them to numeric IDs
            print(f"Labels are strings ({num_numeric}/{num_total} are numeric), applying CLASS_MAPPING")
            self.data["Label"] = self.data["Label"].map(self.CLASS_MAPPING)
            
            # Debug: Check for NaN values after mapping
            nan_count = self.data["Label"].isna().sum()
            if nan_count > 0:
                print(f"WARNING: {nan_count} labels could not be mapped to numeric IDs")
                unmapped_labels = original_labels[self.data["Label"].isna()].unique()
                print(f"Unmapped labels: {unmapped_labels}")
                # Drop rows with unmapped labels
                self.data = self.data.dropna(subset=["Label"]).reset_index(drop=True)
        
        # Debug: Check label distribution after mapping
        print(f"Label distribution after processing:")
        print(self.data["Label"].value_counts().sort_index())
        
        # Filter classes based on index (for incremental learning) - do this BEFORE max_samples
        if index is not None:
            if isinstance(index, list):
                # index is a list of class IDs (integers)
                class_filter = self.data["Label"].isin(index)
                self.data = self.data[class_filter].reset_index(drop=True)
            elif isinstance(index, str):
                # index is a file path
                with open(index, 'r') as f:
                    class_ids = [int(line.strip()) for line in f.readlines()]
                print(f"Filtering for classes: {class_ids}")
                class_filter = self.data["Label"].isin(class_ids)
                print(f"Found {class_filter.sum()} samples matching the filter")
                self.data = self.data[class_filter].reset_index(drop=True)
        
        # Debug: Check final data size
        print(f"Final dataset size: {len(self.data)} samples")
        if len(self.data) > 0:
            print(f"Final label distribution: {self.data['Label'].value_counts().sort_index().to_dict()}")
        
        # Limit dataset size if max_samples is specified - do this AFTER class filtering
        if max_samples is not None and len(self.data) > max_samples:
            # Use stratified sampling to maintain class balance
            from sklearn.model_selection import train_test_split
            X = self.data.drop(columns=["Label"])
            y = self.data["Label"]
            
            # Check if stratified sampling is possible (each class needs at least 2 samples)
            unique_classes = y.unique()
            min_samples_per_class = y.value_counts().min()
            use_stratify = len(unique_classes) > 1 and min_samples_per_class >= 2
            
            if use_stratify:
                X_sampled, _, y_sampled, _ = train_test_split(
                    X, y, train_size=max_samples, stratify=y, random_state=42
                )
            else:
                # Use simple random sampling if stratification is not possible
                print(f"WARNING: Cannot use stratified sampling (classes: {len(unique_classes)}, min samples per class: {min_samples_per_class})")
                sampled_indices = np.random.choice(len(self.data), size=max_samples, replace=False)
                X_sampled = X.iloc[sampled_indices]
                y_sampled = y.iloc[sampled_indices]
            
            self.data = pd.concat([X_sampled, y_sampled], axis=1).reset_index(drop=True)
            print(f"After max_samples filtering: {len(self.data)} samples")
        
        # Check if dataset is empty
        if len(self.data) == 0:
            raise ValueError(
                f"Dataset is empty after filtering! "
                f"This may be due to:\n"
                f"1. No data matching the specified classes in the index file\n"
                f"2. Label mapping issues (check CLASS_MAPPING)\n"
                f"3. Data path issues\n"
                f"Train mode: {self.train}, Index: {index}"
            )
        
        # Pre-compute features and labels for efficiency
        self.features = self.data.drop(columns=["Label"]).values.astype(np.float32)
        self.targets = self.data["Label"].values.astype(np.int64)
        
        # Store the number of features for model initialization
        self.num_features = self.features.shape[1]
        print(f"Dataset initialized with {self.num_features} features")
        
        # Set transform for compatibility
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get pre-computed features and labels
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = self.targets[idx]
        return features, label

