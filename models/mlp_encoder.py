import os

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """Tabular encoder for CIC flow vectors, FACT-compatible.

    Split into pre/post so MYNET.pre_encode/post_encode can run the
    intermediate-feature Mixup exactly like the ResNet branches:
      pre:  [B, D] -> [B, hidden]   (mixed across samples in helper.base_train)
      post: [B, hidden] -> [B, out] (then cosine/dot classifier in post_encode)
    LayerNorm (not BatchNorm) since few-shot batches can be tiny.
    """

    def __init__(self, in_dim, hidden=256, out_dim=512):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_features = out_dim
        self.pre = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.post = nn.Sequential(
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.post(self.pre(x))


def mlp_encoder(in_dim=78, hidden=256, out_dim=512):
    return MLPEncoder(in_dim, hidden, out_dim)


def flow_in_dim(args, default=78):
    """Feature dim from make_session.py outputs; explicit --mlp-in-dim wins."""
    if getattr(args, "mlp_in_dim", None):
        return int(args.mlp_in_dim)
    try:
        import json

        path = os.path.join(
            os.path.expanduser(args.dataroot), args.dataset, "feature_cols.json"
        )
        with open(path) as f:
            return len(json.load(f))
    except OSError:
        return default
