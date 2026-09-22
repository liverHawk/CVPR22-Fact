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

# Process-level cache: every FSCIL session rebuilds its Dataset objects, and
# each rebuild used to re-read the parquet + re-apply the scaler (seconds per
# session). Cached arrays are shared read-only — callers only ever fancy-index
# them (SelectfromClasses/SelectfromTxt) and __getitem__ is read-only.
_RAW_CACHE = {}


def _load_scaled_split(base, split):
    """Read + scale one parquet split, cached for the process lifetime."""
    key = (base, split)
    hit = _RAW_CACHE.get(key)
    if hit is None:
        import pandas as pd

        with open(os.path.join(base, "feature_cols.json")) as f:
            feats = json.load(f)
        df = pd.read_parquet(os.path.join(base, split))
        with open(os.path.join(base, "scaler.pkl"), "rb") as f:
            sc = pickle.load(f)
        data = sc.transform(df[feats].to_numpy(np.float32))
        targets = df["label_id"].to_numpy(np.int64)
        # keep only df.index (row-ID labels for SelectfromTxt), not the frame
        hit = (feats, data, targets, df.index)
        _RAW_CACHE[key] = hit
    return hit


class CICFlow(Dataset):
    """Generic CIC-family flow dataset (CIC-IDS2017, CIC-DDoS2019, ...).

    The per-dataset directory is <root>/<dataset>/, so one class serves all
    CIC flow datasets; the dataset name selects the directory.
    """

    def __init__(
        self,
        root="data/",
        train=True,
        index_path=None,
        index=None,
        base_sess=False,
        dataset="cicids2017",
    ):
        self.root = os.path.expanduser(root)
        base = os.path.join(self.root, dataset)
        split = "train.parquet" if train else "test.parquet"

        feats, data, targets, split_index = _load_scaled_split(base, split)
        self.feature_cols = feats
        self.data = data
        self.targets = targets
        # Interface parity with image datasets: trainer/helper code reads and
        # assigns `.transform`. Tabular data needs no transform, so it is None
        # and __getitem__ ignores it.
        self.transform = None

        if base_sess:
            self.data, self.targets = self.SelectfromClasses(
                self.data, self.targets, index
            )
        elif index_path is not None:
            # incremental train: only the few-shot rows listed in session_t.txt
            self.data, self.targets = self.SelectfromTxt(
                split_index, self.data, self.targets, index_path
            )
        elif index is not None:
            # test: all encountered classes so far
            self.data, self.targets = self.SelectfromClasses(
                self.data, self.targets, index
            )

    def SelectfromTxt(self, split_index, data, targets, index_path):
        with open(index_path) as f:
            ids = [int(x.strip()) for x in f if x.strip()]
        pos = split_index.get_indexer(ids)  # global row ID -> split position
        if (pos < 0).any():
            missing = [i for i, p in zip(ids, pos) if p < 0][:5]
            raise ValueError(
                f"{index_path}: {len(missing)}+ row IDs not in split (e.g. {missing})"
            )
        return data[pos], targets[pos]

    def SelectfromClasses(self, data, targets, index):
        mask = np.isin(targets, np.asarray(list(index)))
        return data[mask], targets[mask]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i]), int(self.targets[i])
