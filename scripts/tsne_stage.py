"""DVC `tsne` stage: t-SNE of the train output, or nothing when disabled.

Reads tsne.yaml. With tsne: false it only writes <out-dir>/SKIPPED (DVC needs
the output to exist), so the stage finishes at once. With tsne: true it
re-evaluates checkpoint/ (scripts/tsne_export.py) and writes per-layer t-SNE,
mixup points and confidence to <out-dir>/session<N>.json + PNGs, plus the
viewer page <out-dir>/viewer.html (scripts/tsne_viewer.py).

Usage (see dvc.yaml):
  uv run python scripts/tsne_stage.py --params tsne.yaml --checkpoint-dir checkpoint --out-dir tsne
"""

import argparse
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TSNE_KEYS = ("tsne_layer", "tsne_samples", "tsne_mixup")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default="tsne.yaml")
    ap.add_argument("--checkpoint-dir", default="checkpoint")
    ap.add_argument("--out-dir", default="tsne")
    a = ap.parse_args()

    with open(a.params) as f:
        cfg = yaml.safe_load(f) or {}
    shutil.rmtree(a.out_dir, ignore_errors=True)  # no leftovers from a longer run
    os.makedirs(a.out_dir)
    if not cfg.get("tsne"):
        with open(os.path.join(a.out_dir, "SKIPPED"), "w") as f:
            f.write(f"tsne: false in {a.params}; set it to true and `dvc repro tsne`\n")
        print(f"tsne disabled in {a.params}: skipped")
        return

    import tsne_export
    import tsne_viewer

    opts = [f"{k}={cfg[k]}" for k in TSNE_KEYS if cfg.get(k) is not None]
    tsne_export.export(a.checkpoint_dir, a.out_dir, opts=opts, write_confusion=False)
    tsne_viewer.write_page(a.checkpoint_dir, a.out_dir)


if __name__ == "__main__":
    main()
