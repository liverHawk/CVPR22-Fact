"""Write t-SNE data for a finished train run that had tsne=false.

Re-evaluates every session<N>_max_acc.pth the way test.py does (so the
per-session metrics are exact for those checkpoints), then runs the same
t-SNE / mixup / confidence export as training with tsne=true:
  <checkpoint-dir>/tsne/session<N>.json and session<N>_<layer>.png
  <checkpoint-dir>/confusion/session<N>.csv (raw confusion-matrix counts)
Build the viewer page afterwards with scripts/tsne_viewer.py.

Usage:
  uv run python scripts/tsne_export.py <checkpoint-dir> [-gpu 0] [--out-dir DIR] [--opts tsne_layer=all]
"""

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def export(checkpoint_dir, out_dir=None, gpu=None, opts=(), write_confusion=True):
    """Re-evaluate every session checkpoint of the run in checkpoint_dir and
    write its t-SNE data to out_dir (default <checkpoint_dir>/tsne).
    gpu=None uses the run's own gpu setting ("" = CPU)."""
    import torch
    import torch.nn.functional as F

    import train
    from dataloader.data_utils import get_test_dataloader
    from utils import set_gpu, set_seed

    d = os.path.normpath(checkpoint_dir)
    argv = ["--config", os.path.join(d, "config.yaml"), "--opts", "use_wandb=false", *opts]
    if gpu is not None:
        argv[2:2] = ["-gpu", str(gpu)]
    args = train.build_args(argv)
    set_seed(args.seed)
    if args.gpu == "":
        args.device, args.num_gpu = "cpu", 0
    else:
        args.device, args.num_gpu = "cuda", set_gpu(args)
    trainer = importlib.import_module(f"models.{args.project}.fscil_trainer").FSCILTrainer(args)
    if os.path.normpath(args.save_path) != d:
        raise SystemExit(f"config resolves to {args.save_path}, not {d}")
    args.tsne_dir = out_dir or os.path.join(d, "tsne")
    from models.fact.helper import test

    for session in range(args.start_session, args.sessions):
        ckpt = os.path.join(d, f"session{session}_max_acc.pth")
        trainer.model.load_state_dict(torch.load(ckpt, map_location=args.device)["params"])
        net = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        testloader = get_test_dataloader(args, session)
        if session == 0:
            _, tsa, tss = test(trainer.model, testloader, 0, args, session, return_stats=True, log_cm=False)
            # same dummy classifiers the train run built after session 0
            trainer.dummy_classifiers = F.normalize(
                net.fc.weight.detach()[args.base_class :, :], p=2, dim=-1
            )
        else:
            net.mode = args.new_mode
            _, tsa, tss = trainer.test_intergrate(
                trainer.model, testloader, 0, args, session, validation=True, return_stats=True
            )
        trainer._record_session(session, tsa, tss)
        if write_confusion:
            trainer._session_cm_counts(session)
        trainer._session_tsne(session, testloader)
        print(f"session {session}: acc={tsa * 100:.3f} -> {args.tsne_dir}/session{session}.json", flush=True)
    return args.tsne_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint_dir")
    ap.add_argument("-gpu", default=None, help="default: the run's own gpu setting")
    ap.add_argument("--out-dir", default=None, help="default: <checkpoint-dir>/tsne")
    ap.add_argument("--opts", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override tsne_* etc. on top of the run's config.yaml")
    a = ap.parse_args()
    export(a.checkpoint_dir, a.out_dir, a.gpu, a.opts)


if __name__ == "__main__":
    main()
