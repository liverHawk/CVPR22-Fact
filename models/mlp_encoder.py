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

    pre_layers / post_layers set how many Linear-LayerNorm-ReLU blocks sit
    before / after the mixup point. post keeps width hidden and only its
    last block maps to out_dim. The defaults (2, 1) reproduce the original
    layout and module indices, so existing checkpoints keep loading.

    With cardinalities (embed_cols set), pre starts with FlowInput and in_dim
    counts only the continuous columns. Without them pre is unchanged, so
    existing checkpoints keep loading.
    """

    def __init__(
        self,
        in_dim,
        hidden=256,
        out_dim=512,
        cardinalities=(),
        embed_dim=8,
        pre_layers=2,
        post_layers=1,
    ):
        super().__init__()
        if pre_layers < 1 or post_layers < 1:
            raise ValueError(
                f"pre_layers/post_layers must be >= 1, got {pre_layers}/{post_layers}"
            )
        self.in_dim = in_dim
        self.hidden = hidden
        self.out_features = out_dim
        head = []
        width = in_dim
        if cardinalities:
            head = [FlowInput(in_dim, cardinalities, embed_dim)]
            width = in_dim + len(cardinalities) * embed_dim
        pre_dims = [width] + [hidden] * pre_layers
        post_dims = [hidden] * post_layers + [out_dim]
        self.pre = nn.Sequential(*head, *_blocks(pre_dims))
        self.post = nn.Sequential(*_blocks(post_dims))

    def forward(self, x):
        return self.post(self.pre(x))

    def layer_outputs(self, x):
        """[(name, activation)] after every block's ReLU: pre1..preN, then
        post1..postM. preN is the mixup point, postM the final embedding."""
        outs = _relu_outputs(self.pre, x, "pre")
        return outs + _relu_outputs(self.post, outs[-1][1], "post")

    def post_outputs(self, h):
        """[(name, activation)] for post1..postM starting from a pre output
        h (e.g. a mixed one); postM equals post(h)."""
        return _relu_outputs(self.post, h, "post")


def _relu_outputs(seq, x, prefix):
    outs, k = [], 0
    for m in seq:
        x = m(x)
        if isinstance(m, nn.ReLU):
            k += 1
            outs.append((f"{prefix}{k}", x))
    return outs


def _blocks(dims):
    """Linear-LayerNorm-ReLU per consecutive (d_in, d_out) pair in dims."""
    layers = []
    for d_in, d_out in zip(dims, dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.ReLU()]
    return layers


def mlp_encoder(
    in_dim=78,
    hidden=256,
    out_dim=512,
    cardinalities=(),
    embed_dim=8,
    pre_layers=2,
    post_layers=1,
):
    return MLPEncoder(
        in_dim, hidden, out_dim, cardinalities, embed_dim, pre_layers, post_layers
    )


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
        pre_layers=getattr(args, "mlp_pre_layers", 2),
        post_layers=getattr(args, "mlp_post_layers", 1),
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
