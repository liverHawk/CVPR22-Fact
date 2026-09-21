"""CICFlow flow dataloader for FSCIL (generic for CIC-family flow datasets:
CIC-IDS2017, CIC-DDoS2019, ...; content is CICFlowMeter-format flows).

Reads artifacts produced by scripts/make_session.py:
  <root>/<dataset>/{train,test}.parquet  (index = global row ID)
  <root>/<dataset>/feature_cols.json
  <root>/<dataset>/scaler.pkl            (fit on base-train only)
  data/index_list/<dataset>/session_*.txt (global row IDs, one per line)

__getitem__ returns (FloatTensor[D], int). No PIL / torchvision transforms.
"""
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class CICFlow(Dataset):
    """Generic CIC-family flow dataset (CIC-IDS2017, CIC-DDoS2019, ...).

    The per-dataset directory is <root>/<dataset>/, so one class serves all
    CIC flow datasets; the dataset name selects the directory.
    """

    def __init__(self, root='data/', train=True,
                 index_path=None, index=None, base_sess=False,
                 dataset='cicids2017'):
        self.root = os.path.expanduser(root)
        base = os.path.join(self.root, dataset)
        with open(os.path.join(base, 'feature_cols.json')) as f:
            feats = json.load(f)
        split = 'train.parquet' if train else 'test.parquet'

        import pandas as pd
        df = pd.read_parquet(os.path.join(base, split))
        with open(os.path.join(base, 'scaler.pkl'), 'rb') as f:
            sc = pickle.load(f)
        self.feature_cols = feats
        self.data = sc.transform(df[feats].to_numpy(np.float32))
        self.targets = df['label_id'].to_numpy(np.int64)

        if base_sess:
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
        elif index_path is not None:
            # incremental train: only the few-shot rows listed in session_t.txt
            self.data, self.targets = self.SelectfromTxt(df, self.data, self.targets, index_path)
        elif index is not None:
            # test: all encountered classes so far
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

    def SelectfromTxt(self, df, data, targets, index_path):
        with open(index_path) as f:
            ids = [int(x.strip()) for x in f if x.strip()]
        pos = df.index.get_indexer(ids)  # global row ID -> split position
        if (pos < 0).any():
            missing = [i for i, p in zip(ids, pos) if p < 0][:5]
            raise ValueError(f'{index_path}: {len(missing)}+ row IDs not in split (e.g. {missing})')
        return data[pos], targets[pos]

    def SelectfromClasses(self, data, targets, index):
        mask = np.isin(targets, np.asarray(list(index)))
        return data[mask], targets[mask]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i]), int(self.targets[i])
