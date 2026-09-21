"""Main test run (pipeline stage 3: make session file -> train -> test).

Loads per-session checkpoints saved by the train run
(<checkpoint-dir>/session<N>_max_acc.pth) and evaluates each session on all
encountered classes. Writes a metrics JSON for DVC metrics / Optuna.

Usage:
  uv run python test.py --config params.yaml --checkpoint-dir checkpoint/cifar100/base/...
  uv run python test.py --config params.yaml --test-metrics-file test_metrics.json
  uv run python test.py --config params.yaml --opts dataset=cifar100 --checkpoint-dir <dir> --gpu "" --num_workers 2
If --checkpoint-dir is omitted, the train save_path resolved from the same
config is used (same-params train -> test chaining for `dvc repro`).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_parser():
    from train import get_command_line_parser
    parser = get_command_line_parser()
    parser.description = 'Main test run for FACT FSCIL checkpoints.'
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='dir holding session<N>_max_acc.pth from the train run (required)')
    parser.add_argument('--test-metrics-file', type=str, default=None,
                        help='output metrics JSON (default: <checkpoint-dir>/test_metrics.json)')
    return parser


def main(argv=None):
    import importlib
    import torch
    from utils import ensure_path, set_seed

    parser = get_parser()
    # build_args parses with the train parser, which lacks test-only flags;
    # parse here with the extended parser, then reuse train helpers for --config/--opts.
    from train import apply_configs, explicit_cli_keys, parse_opt_value
    args = parser.parse_args(argv)
    explicit = explicit_cli_keys(list(argv) if argv is not None else None)
    apply_configs(args, explicit)
    for item in (args.opts or []):
        k, v = item.split('=', 1)
        k = k.strip()
        if not hasattr(args, k):
            raise ValueError(f'--opts unknown key: {k}')
        setattr(args, k, parse_opt_value(v))

    set_seed(args.seed)
    if args.gpu == "":
        args.device = "cpu"
        args.num_gpu = 0
    else:
        from utils import set_gpu
        args.device = "cuda"
        args.num_gpu = set_gpu(args)

    trainer_cls = importlib.import_module('models.%s.fscil_trainer' % args.project).FSCILTrainer
    trainer = trainer_cls(args)  # reuses save_path/dataloader/model construction
    # Default checkpoint dir is the train run's save_path resolved from the same config.
    ckpt_dir = args.checkpoint_dir or args.save_path
    map_loc = args.device if args.device == 'cpu' else None

    session_acc, session_loss = [], []
    for session in range(args.start_session, args.sessions):
        ckpt = os.path.join(ckpt_dir, f'session{session}_max_acc.pth')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'missing checkpoint for session {session}: {ckpt}')
        print(f'== test session {session}: loading {ckpt}')
        state = torch.load(ckpt, map_location=map_loc if map_loc else args.device)
        trainer.model.load_state_dict(state['params'])
        model = trainer.model
        _, _, testloader = trainer.get_dataloader(session)

        if session == 0:
            if args.project == 'fact':
                from models.fact.helper import test
            else:
                from models.base.helper import test
            tsl, tsa = test(model, testloader, 0, args, session)
        else:
            net = model.module if hasattr(model, 'module') else model
            net.mode = args.new_mode
            if args.project == 'fact':
                # rebuild dummy classifiers from the session-0 checkpoint (same as train run)
                import torch.nn.functional as F
                from copy import deepcopy
                s0 = torch.load(os.path.join(ckpt_dir, 'session0_max_acc.pth'),
                                map_location=map_loc if map_loc else args.device)
                fc = s0['params']['module.fc.weight' if any(k.startswith('module.') for k in s0['params']) else 'fc.weight']
                dummy = F.normalize(fc[args.base_class:, :].detach(), p=2, dim=-1)
                trainer.dummy_classifiers = deepcopy(dummy)
                tsl, tsa = trainer.test_intergrate(model, testloader, 0, args, session, validation=True)
            else:
                from models.base.helper import test
                tsl, tsa = test(model, testloader, 0, args, session, validation=False)
        print(f'session {session}: loss={tsl:.4f} acc={tsa:.4f}' if tsl is not None else f'session {session}: acc={tsa:.4f}')
        session_loss.append(float(tsl) if tsl is not None else None)
        session_acc.append(float(tsa * 100))

    metrics = {
        'checkpoint_dir': ckpt_dir,
        'session_acc': session_acc,
        'session_loss': session_loss,
        'avg_acc': sum(session_acc) / len(session_acc),
        'final_acc': session_acc[-1],
    }
    out = args.test_metrics_file or os.path.join(ckpt_dir, 'test_metrics.json')
    ensure_path(os.path.dirname(out) or '.')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'saved {out}: {metrics}')
    return metrics


if __name__ == '__main__':
    main()
