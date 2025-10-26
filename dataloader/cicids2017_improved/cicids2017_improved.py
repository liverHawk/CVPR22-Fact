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
    files = glob(os.path.join(data_path, "*.csv.gz"))
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
        
        # Convert string labels to numeric IDs
        self.data["Label"] = self.data["Label"].map(self.CLASS_MAPPING)
        
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
                class_filter = self.data["Label"].isin(class_ids)
                self.data = self.data[class_filter].reset_index(drop=True)
        
        # Limit dataset size if max_samples is specified - do this AFTER class filtering
        if max_samples is not None and len(self.data) > max_samples:
            # Use stratified sampling to maintain class balance
            from sklearn.model_selection import train_test_split
            X = self.data.drop(columns=["Label"])
            y = self.data["Label"]
            X_sampled, _, y_sampled, _ = train_test_split(
                X, y, train_size=max_samples, stratify=y, random_state=42
            )
            self.data = pd.concat([X_sampled, y_sampled], axis=1).reset_index(drop=True)
        
        # Pre-compute features and labels for efficiency
        self.features = self.data.drop(columns=["Label"]).values.astype(np.float32)
        self.targets = self.data["Label"].values.astype(np.int64)
        
        # Set transform for compatibility
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get pre-computed features and labels
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = self.targets[idx]
        return features, label

