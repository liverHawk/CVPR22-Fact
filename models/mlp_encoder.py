import os

import torch
from torch import nn


class FlowInput(nn.Module):
    """[B, n_cont + n_cat] -> [B, n_cont + n_cat*embed_dim].

    The last n_cat columns hold integer ids (stored as float by CICFlow) and
    are looked up in per-column nn.Embedding tables; id 0 is OOV.
    """

    def __init__(self, n_cont, cardinalities, embed_dim):
        super().__init__()
        self.n_cont = n_cont
        self.embeds = nn.ModuleList(nn.Embedding(c, embed_dim) for c in cardinalities)

    def forward(self, x):
        ids = x[:, self.n_cont :].long()
        embs = [e(ids[:, i]) for i, e in enumerate(self.embeds)]
        return torch.cat([x[:, : self.n_cont]] + embs, dim=1)


class MLPEncoder(nn.Module):
    """Tabular encoder for CIC flow vectors, FACT-compatible.

    Split into pre/post so MYNET.pre_encode/post_encode can run the
    intermediate-feature Mixup exactly like the ResNet branches:
      pre:  [B, D] -> [B, hidden]   (mixed across samples in helper.base_train)
      post: [B, hidden] -> [B, out] (then cosine/dot classifier in post_encode)
    LayerNorm (not BatchNorm) since few-shot batches can be tiny.

    With cardinalities (embed_cols set), pre starts with FlowInput and in_dim
    counts only the continuous columns. Without them pre is unchanged, so
    existing checkpoints keep loading.
    """

    def __init__(self, in_dim, hidden=256, out_dim=512, cardinalities=(), embed_dim=8):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_features = out_dim
        head = []
        width = in_dim
        if cardinalities:
            head = [FlowInput(in_dim, cardinalities, embed_dim)]
            width = in_dim + len(cardinalities) * embed_dim
        self.pre = nn.Sequential(
            *head,
            nn.Linear(width, hidden),
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


def mlp_encoder(in_dim=78, hidden=256, out_dim=512, cardinalities=(), embed_dim=8):
    return MLPEncoder(in_dim, hidden, out_dim, cardinalities, embed_dim)


def flow_encoder(args):
    """MLPEncoder sized from make_session.py outputs + embed_cols settings."""
    from dataloader.data_utils import embed_cols

    cols = embed_cols(args)
    cards = []
    in_dim = flow_in_dim(args)
    if cols:
        from dataloader.cicflow.cicflow import flow_layout

        n_cont, cards = flow_layout(
            args.dataroot,
            args.dataset,
            cols,
            base_class=args.base_class,
            max_vocab=getattr(args, "embed_max_vocab", 1024),
        )
        if not getattr(args, "mlp_in_dim", None):
            in_dim = n_cont
    return mlp_encoder(
        in_dim=in_dim,
        hidden=args.mlp_hidden,
        out_dim=args.mlp_out,
        cardinalities=cards,
        embed_dim=getattr(args, "embed_dim", 8),
    )


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
