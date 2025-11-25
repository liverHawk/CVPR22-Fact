import os
# import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as preprocessing
# from sklearn.model_selection import train_test_split


class CICIDS2017_improved(Dataset):
    """
    CICIDS2017_improved dataset loader for few-shot class-incremental learning.
    This dataset contains network traffic features for intrusion detection.
    """

    def __init__(
        self,
        root="data/",
        train=True,
        index_path=None,
        index=None,
        base_sess=None,
        autoaug=1,
        normalize_method="standard",
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.autoaug = autoaug
        self.normalize_method = normalize_method

        # Load and preprocess data
        self._pre_operate(self.root)

        # Select data based on index or index_path
        if base_sess:
            self.data, self.targets = self.SelectfromClasses(
                self.data, self.targets, index
            )
        else:
            if index_path is not None:
                # For new sessions, index_path contains data indices
                self.data, self.targets = self.SelectfromTxt(
                    self.data2label, index_path
                )
            elif index is not None:
                # If index is provided directly (list of class indices or data indices)
                if isinstance(index, list) and len(index) > 0:
                    # Check if index contains class indices (integers matching class labels)
                    # or data indices (larger numbers)
                    if all(isinstance(i, (int, str)) for i in index):
                        # Try to interpret as class indices first
                        try:
                            class_indices = [int(i) for i in index]
                            # If all indices are within valid class range, treat as class indices
                            if all(
                                0 <= idx < len(np.unique(self.targets))
                                for idx in class_indices
                            ):
                                self.data, self.targets = self.SelectfromClasses(
                                    self.data, self.targets, class_indices
                                )
                            else:
                                # Otherwise treat as data indices
                                self.data, self.targets = self.SelectfromDataIndices(
                                    self.data, self.targets, class_indices
                                )
                        except Exception:
                            self.data, self.targets = self.SelectfromClasses(
                                self.data, self.targets, index
                            )
                    else:
                        self.data, self.targets = self.SelectfromClasses(
                            self.data, self.targets, index
                        )
                else:
                    self.data, self.targets = self.SelectfromClasses(
                        self.data, self.targets, index
                    )
        
        nplist = self.targets.tolist()
        print(np.unique(nplist))

        # Convert to numpy arrays if needed
        if isinstance(self.data, list):
            self.data = np.array(self.data)
        if isinstance(self.targets, list):
            self.targets = np.array(self.targets)

    def _get_delete_columns(self):
        """
        Columns to delete according to research_data_drl preprocessing.
        """
        return [
            "id",
            "Flow ID",
            "Src IP",
            "Src Port",
            "Dst IP",
            "Timestamp",
            "Bwd PSH Flags",
            "Fwd URG Flags",
            "Bwd URG Flags",
            "Fwd RST Flags",
            "Bwd RST Flags",
            "FIN Flag Count",
            "RST Flag Count",
            "URG Flag Count",
            "CWR Flag Count",
            "ECE Flag Count",
            "Fwd Bytes/Bulk Avg",
            "Fwd Packet/Bulk Avg",
            "Fwd Bulk Rate Avg",
            "Bwd Bytes/Bulk Avg",
            "ICMP Code",
            "ICMP Type",
            "Total TCP Flow Time",
        ]

    def _get_column_rename_dict(self):
        """
        Column rename mapping from CICIDS2017 to BASE format.
        """
        cicids2017_cols = [
            "Dst Port",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packet",
            "Total Bwd packets",
            "Total Length of Fwd Packet",
            "Total Length of Bwd Packet",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Packet Length Min",
            "Packet Length Max",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "SYN Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Fwd Segment Size Avg",
            "Bwd Segment Size Avg",
            "Bwd Packet/Bulk Avg",
            "Bwd Bulk Rate Avg",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "FWD Init Win Bytes",
            "Bwd Init Win Bytes",
            "Fwd Act Data Pkts",
            "Fwd Seg Size Min",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ]

        base_cols = [
            "Destination Port",
            "Protocol",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Fwd Packet Length Max",
            "Fwd Packet Length Min",
            "Fwd Packet Length Mean",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
            "Bwd Packet Length Min",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Std",
            "Flow Bytes/s",
            "Flow Packets/s",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Total",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Total",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd Header Length",
            "Bwd Header Length",
            "Fwd Packets/s",
            "Bwd Packets/s",
            "Min Packet Length",
            "Max Packet Length",
            "Packet Length Mean",
            "Packet Length Std",
            "Packet Length Variance",
            "SYN Flag Count",
            "PSH Flag Count",
            "ACK Flag Count",
            "Down/Up Ratio",
            "Average Packet Size",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Bwd Avg Packets/Bulk",
            "Bwd Avg Bulk Rate",
            "Subflow Fwd Packets",
            "Subflow Fwd Bytes",
            "Subflow Bwd Packets",
            "Subflow Bwd Bytes",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ]

        return dict(zip(cicids2017_cols, base_cols))

    def _fast_process(self, df):
        """
        Fast preprocessing: drop columns and handle missing/infinite values.
        Following research_data_drl preprocessing pattern.
        """
        # Drop specified columns
        delete_cols = self._get_delete_columns()
        existing_delete_cols = [col for col in delete_cols if col in df.columns]
        df = df.drop(existing_delete_cols, axis=1)

        # Drop 'Attempted Category' column if exists
        if "Attempted Category" in df.columns:
            df = df.drop(columns=["Attempted Category"])

        # Replace infinite values with NaN and drop NaN rows
        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        return df

    def _column_adjustment(self, df):
        """
        Adjust labels and rename columns according to research_data_drl preprocessing.
        """
        # Label consolidation
        labels = df["Label"].unique()

        for label in labels:
            if "Attempted" in label:
                df.loc[df["Label"] == label, "Label"] = "BENIGN"
            if "Web Attack" in label:
                df.loc[df["Label"] == label, "Label"] = "Web Attack"
            if "Infiltration" in label:
                df.loc[df["Label"] == label, "Label"] = "Infiltration"
            if "DoS" in label and label != "DDoS":
                df.loc[df["Label"] == label, "Label"] = "DoS"

        # Rename columns
        rename_dict = self._get_column_rename_dict()
        # Only rename columns that exist in the dataframe
        existing_rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
        df = df.rename(columns=existing_rename_dict)

        return df

    def _pre_operate(self, root):
        """
        Load and preprocess CICIDS2017_improved dataset.
        Following preprocessing pattern from research_data_drl repository.
        Expected structure:
        - root/CICIDS2017_improved/train.csv (or train.parquet)
        - root/CICIDS2017_improved/test.csv (or test.parquet)
        """
        # data_dir = os.path.join(root, 'CICIDS2017_improved')
        data_dir = root

        # Try to load CSV or parquet files
        train_file = os.path.join(data_dir, "train.csv")
        test_file = os.path.join(data_dir, "test.csv")

        if not os.path.exists(train_file):
            train_file = os.path.join(data_dir, "train.parquet")
            test_file = os.path.join(data_dir, "test.parquet")

        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"CICIDS2017_improved dataset not found in {data_dir}. "
                "Please ensure train.csv/test.csv or train.parquet/test.parquet exist."
            )

        # Load data
        if train_file.endswith(".csv"):
            train_df = pd.read_csv(train_file)
            test_df = pd.read_csv(test_file)
        else:
            train_df = pd.read_parquet(train_file)
            test_df = pd.read_parquet(test_file)

        # Fast preprocessing: drop columns and handle missing/infinite values
        train_df = self._fast_process(train_df)
        test_df = self._fast_process(test_df)

        # Column adjustment: label consolidation and column renaming
        train_df: pd.DataFrame = self._column_adjustment(train_df)
        test_df: pd.DataFrame = self._column_adjustment(test_df)

        # Separate features and labels
        if "Label" not in train_df.columns:
            raise ValueError("'Label' column not found in dataset")
        label_col = "Label"

        # Encode labels
        le = LabelEncoder()
        all_labels = pd.concat([train_df[label_col], test_df[label_col]], axis=0)
        le.fit(all_labels)

        # Split features and labels
        feature_cols = [col for col in train_df.columns if col != label_col]

        train_features = train_df[feature_cols].values
        train_labels = le.transform(train_df[label_col].values)

        test_features = test_df[feature_cols].values
        test_labels = le.transform(test_df[label_col].values)

        # Normalize features
        # scaler = StandardScaler()
        train_features = self.normalize_features(train_features)
        test_features = self.normalize_features(test_features)
        # train_features = scaler.fit_transform(train_features)
        # test_features = scaler.transform(test_features)

        # Store scaler and label encoder for later use
        # self.scaler = scaler
        # self.label_encoder = le

        # Create data2label mapping for index-based selection
        self.data2label = {}

        if self.train:
            self.data = train_features
            self.targets = train_labels
            for i, label in enumerate(train_labels):
                self.data2label[i] = label
        else:
            self.data = test_features
            self.targets = test_labels
            for i, label in enumerate(test_labels):
                self.data2label[i] = label

    def SelectfromTxt(self, data2label, index_path):
        """
        Select data based on index file (list of indices).
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        with open(index_path, "r") as f:
            indices = [int(line.strip()) for line in f.readlines()]

        data_tmp = []
        targets_tmp = []

        for idx in indices:
            if idx < len(self.data):
                data_tmp.append(self.data[idx])
                targets_tmp.append(self.targets[idx])
        
        targets_list = targets_tmp
        print(np.unique(targets_list))

        return np.array(data_tmp), np.array(targets_tmp)

    def SelectfromClasses(self, data, targets, index):
        """
        Select data based on class indices.
        """
        data_tmp = []
        targets_tmp = []

        for class_idx in index:
            ind_cl = np.where(class_idx == targets)[0]
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))

        targets_list = targets_tmp
        print(np.unique(targets_list))

        return data_tmp, targets_tmp

    def SelectfromDataIndices(self, data, targets, indices):
        """
        Select data based on data sample indices.
        """
        data_tmp = []
        targets_tmp = []

        for idx in indices:
            if 0 <= idx < len(data):
                data_tmp.append(data[idx])
                targets_tmp.append(targets[idx])

        targets_list = targets_tmp
        print(np.unique(targets_list))

        return np.array(data_tmp), np.array(targets_tmp)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple (features, label) where features is a tensor.
        """
        features = self.data[idx]
        label = self.targets[idx]

        # Convert to tensor
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]

        return features, label
    
    def normalize_features(self, features):
        if self.normalize_method == "standard":
            scaler = preprocessing.StandardScaler()
            features = scaler.fit_transform(features)
        elif self.normalize_method == "minmax":
            scaler = preprocessing.MinMaxScaler()
            features = scaler.fit_transform(features)
        elif self.normalize_method == "moving_minmax":
            features = moving_minmax(features)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features


def moving_minmax(features):
    df = pd.DataFrame(features)
    rolling = df.rolling(window=5, min_periods=1)
    rolling_min = rolling.min()
    rolling_max = rolling.max()
    df = (df - rolling_min) / (rolling_max - rolling_min + 1e-6)
    return df.values
