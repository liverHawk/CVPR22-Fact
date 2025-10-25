import os
import pandas as pd

from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

from data_utils import get_original_feature_labels, get_renamed_feature_labels, get_delete_feature_labels


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
    files = glob(os.path.join(data_path, "*.csv"))
    dfs = [pd.read_csv(file) for file in tqdm(files, desc="Loading data")]
    df = pd.concat(dfs)
    df = _preprocess_data(df, label_column)
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

    def __init__(
        self, root, train=True, transform=None, target_transform=None,
        download=False, index=None, base_sess=None, autoaug=1
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
        

