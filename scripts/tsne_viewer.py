"""Build a standalone t-SNE viewer page from one train run.

Reads <checkpoint-dir>/tsne/session<N>.json (written by train.py with
tsne=true) plus <checkpoint-dir>/config.yaml and inlines them into
scripts/tsne_viewer.template.html. The page shows one panel per saved layer,
so it follows whatever mlp_pre_layers / mlp_post_layers / tsne_layer the run
used.

Usage:
  uv run python scripts/tsne_viewer.py <checkpoint-dir> [-o out.html]
"""

import argparse
import glob
import json
import os
import re

import yaml

CONFIG_KEYS = (
    "dataset", "epochs_base", "lr_base", "batch_size_base", "loss_iter", "balance",
    "normalize", "mlp_hidden", "mlp_out", "mlp_pre_layers", "mlp_post_layers",
    "temperature", "seed", "tsne_samples", "tsne_mixup", "alpha", "eta",
)


def build(ckpt_dir, tsne_dir=None):
    tsne_dir = tsne_dir or os.path.join(ckpt_dir, "tsne")
    files = glob.glob(os.path.join(tsne_dir, "session*.json"))
    files.sort(key=lambda f: int(re.search(r"session(\d+)\.json$", f).group(1)))
    if not files:
        raise SystemExit(f"no session*.json under {tsne_dir} (train with tsne=true or run tsne_export.py)")
    sessions = [json.load(open(f)) for f in files]
    layer_sets = {tuple(l["name"] for l in s["layers"]) for s in sessions}
    if len(layer_sets) != 1:
        raise SystemExit(f"sessions saved different layer sets: {layer_sets}")
    with open(os.path.join(ckpt_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f) or {}
    first = sessions[0]
    return {
        "run": os.path.basename(os.path.normpath(ckpt_dir)),
        "config": {k: cfg.get(k) for k in CONFIG_KEYS},
        "names": first["names"],
        "base_class": first["base_class"],
        "sessions": [
            {k: s[k] for k in ("session", "labels", "conf", "pred", "mix", "cm", "metrics", "layers") if k in s}
            for s in sessions
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--tsne-dir", default=None, help="default: <checkpoint-dir>/tsne")
    ap.add_argument("-o", "--out", default=None, help="default: <tsne-dir>/viewer.html")
    a = ap.parse_args()
    write_page(a.checkpoint_dir, a.tsne_dir, a.out)


def write_page(checkpoint_dir, tsne_dir=None, out=None):
    data = build(checkpoint_dir, tsne_dir)
    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsne_viewer.template.html")).read()
    # "</" inside the inline JSON would close the <script> tag early
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out = out or os.path.join(tsne_dir or os.path.join(checkpoint_dir, "tsne"), "viewer.html")
    with open(out, "w") as f:
        f.write(template.replace("__DATA__", blob))
    n_layers = len(data["sessions"][0]["layers"])
    print(f"wrote {out}: {len(data['sessions'])} sessions x {n_layers} layers")


if __name__ == "__main__":
    main()
