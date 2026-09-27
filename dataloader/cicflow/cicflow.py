"""CICFlow flow dataloader for FSCIL (generic for CIC-family flow datasets:
CIC-IDS2017, CIC-DDoS2019, ...; content is CICFlowMeter-format flows).

Reads artifacts produced by scripts/make_session.py:
  <root>/<dataset>/{train,test}.parquet  (index = global row ID)
  <root>/<dataset>/feature_cols.json
  data/index_list/<dataset>/session_*.txt (global row IDs, one per line)

Normalization is selected by `normalize` (see NORMALIZERS) and fit at load
time on base-train rows only (train split, label_id < base_class), so
switching it needs no make_session.py rerun and never leaks test/new-class
statistics.

Columns in `embed_cols` skip scaling and are mapped to integer ids for
nn.Embedding (vocab = the embed_max_vocab-1 most frequent base-train values,
id 0 = unseen/OOV). They are appended after the continuous columns, so each
row is [continuous..., embed ids...]; flow_layout() tells the encoder where
the split is and each column's cardinality.

__getitem__ returns (FloatTensor[D], int). No PIL / torchvision transforms.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# Process-level cache: every FSCIL session rebuilds its Dataset objects, and
# each rebuild used to re-read the parquet + re-apply the scaler (seconds per
# session). Cached arrays are shared read-only — callers only ever fancy-index
# them (SelectfromClasses/SelectfromTxt) and __getitem__ is read-only.
_RAW_CACHE = {}
_SCALED_CACHE = {}

NORMALIZERS = ("none", "standard", "minmax", "robust")


def _make_scaler(name):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    return {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
    }[name]()


def _load_raw_split(base, split):
    """Read one unscaled parquet split, cached for the process lifetime."""
    key = (base, split)
    hit = _RAW_CACHE.get(key)
    if hit is None:
        import pandas as pd

        with open(os.path.join(base, "feature_cols.json")) as f:
            feats = json.load(f)
        df = pd.read_parquet(os.path.join(base, split))
        data = df[feats].to_numpy(np.float32)
        targets = df["label_id"].to_numpy(np.int64)
        # keep only df.index (row-ID labels for SelectfromTxt), not the frame
        hit = (feats, data, targets, df.index)
        _RAW_CACHE[key] = hit
    return hit


def _base_train(base, base_class):
    _, tr_data, tr_targets, _ = _load_raw_split(base, "train.parquet")
    return tr_data[tr_targets < base_class]


def _split_cols(feats, embed_cols):
    unknown = [c for c in embed_cols if c not in feats]
    if unknown:
        raise ValueError(f"embed_cols not in feature_cols.json: {unknown}")
    cat = [feats.index(c) for c in embed_cols]
    cont = [i for i in range(len(feats)) if i not in set(cat)]
    return cont, cat


def _fit_vocab(values, max_vocab):
    """Sorted top-(max_vocab-1) values by frequency; ids are 1-based."""
    uniq, cnt = np.unique(values, return_counts=True)
    top = uniq[np.argsort(-cnt, kind="stable")[: max_vocab - 1]]
    return np.sort(top)


def _to_ids(values, vocab):
    pos = np.searchsorted(vocab, values)
    hit = pos < len(vocab)
    hit[hit] = vocab[pos[hit]] == values[hit]
    return np.where(hit, pos + 1, 0).astype(np.float32)


def _load_scaled_split(base, split, normalize, base_class, embed_cols, max_vocab):
    """Raw split scaled by `normalize` (continuous cols) + embed ids, with
    scaler and vocab fit on base-train rows only."""
    if normalize not in NORMALIZERS:
        raise ValueError(f"normalize={normalize!r}; choose from {NORMALIZERS}")
    key = (base, split, normalize, base_class, embed_cols, max_vocab)
    hit = _SCALED_CACHE.get(key)
    if hit is None:
        feats, data, targets, index = _load_raw_split(base, split)
        if (normalize != "none" or embed_cols) and base_class is None:
            raise ValueError("base_class is required to fit the scaler/vocab")
        cont, cat = _split_cols(feats, embed_cols)
        x = data[:, cont]
        if normalize != "none":
            sc = _make_scaler(normalize).fit(_base_train(base, base_class)[:, cont])
            x = sc.transform(x).astype(np.float32, copy=False)
        if cat:
            bt = _base_train(base, base_class)
            ids = [_to_ids(data[:, j], _fit_vocab(bt[:, j], max_vocab)) for j in cat]
            x = np.concatenate([x, np.stack(ids, axis=1)], axis=1)
        hit = ([feats[i] for i in cont + cat], x, targets, index)
        _SCALED_CACHE[key] = hit
    return hit


def flow_layout(root, dataset, embed_cols=(), base_class=None, max_vocab=1024):
    """(n_continuous, [cardinality per embed col]) for building the encoder."""
    base = os.path.join(os.path.expanduser(root), dataset)
    feats, _, _, _ = _load_raw_split(base, "train.parquet")
    cont, cat = _split_cols(feats, tuple(embed_cols))
    if not cat:
        return len(cont), []
    bt = _base_train(base, base_class)
    return len(cont), [len(_fit_vocab(bt[:, j], max_vocab)) + 1 for j in cat]


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
        normalize="standard",
        base_class=None,
        embed_cols=(),
        embed_max_vocab=1024,
    ):
        self.root = os.path.expanduser(root)
        base = os.path.join(self.root, dataset)
        split = "train.parquet" if train else "test.parquet"

        feats, data, targets, split_index = _load_scaled_split(
            base, split, normalize, base_class, tuple(embed_cols), embed_max_vocab
        )
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
