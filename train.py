import argparse
import importlib
import sys
from utils import *
import wandb

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

# Keys that live in params.yaml for the pipeline stages (make_session.py / test.py)
# but are not train.py arguments. apply_config() skips them silently.
PIPELINE_ONLY_KEYS = frozenset([
    'index_list_dir', 'session_seed', 'overwrite_session',
    'checkpoint_dir', 'test_metrics_file',
    # scripts/make_session.py (netflow) keys
    'flow_glob', 'flow_data_dir', 'label_col', 'drop_cols', 'label_aliases',
    'base_classes', 'base_class_num', 'session_way', 'session_shot',
    'exclude_labels', 'test_size',
])


def parse_opt_value(v):
    """Parse KEY=VALUE string value into int/float/bool/None/list."""
    s = v.strip()
    if s.lower() in ('true', 'yes', 'on'):
        return True
    if s.lower() in ('false', 'no', 'off'):
        return False
    if s.lower() in ('none', 'null', '~'):
        return None
    if ',' in s:
        return [parse_opt_value(p) for p in s.split(',')]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def load_config_file(path):
    import yaml
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f'config file must be a mapping: {path}')
    return cfg


def explicit_cli_keys(argv=None):
    """Return set of destination names explicitly passed on the command line."""
    if argv is None:
        argv = sys.argv[1:]
    keys = set()
    for tok in argv:
        if tok.startswith('--'):
            keys.add(tok[2:].split('=')[0])
        elif tok.startswith('-') and not tok.startswith('---'):
            keys.add(tok[1:].split('=')[0])
    return keys


def apply_config(args, cfg, explicit_keys, source='config'):
    for k, v in cfg.items():
        if not hasattr(args, k):
            if k in PIPELINE_ONLY_KEYS:
                continue  # consumed by make_session.py / test.py, not train.py
            print(f'WARNING: unknown {source} key ignored: {k}')
            continue
        if k in explicit_keys:
            continue  # explicit CLI flag wins
        setattr(args, k, v)
    return args


def save_config(args, path):
    import yaml
    import types
    cfg = {}
    for k, v in vars(args).items():
        if isinstance(v, types.ModuleType):
            continue  # e.g. args.Dataset injected by set_up_datasets
        cfg[k] = v
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)


def write_metrics(trainer, args, metrics_file):
    import json
    max_acc = trainer.trlog['max_acc']
    if isinstance(max_acc, (list, tuple)):
        accs = [float(a) for a in max_acc]
    else:
        accs = [float(max_acc.tolist()[0])]
    metrics = {
        'avg_acc': sum(accs) / len(accs),
        'final_acc': accs[-1],
        'max_acc': accs,
        'max_acc_epoch': trainer.trlog.get('max_acc_epoch'),
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics

def get_command_line_parser():
    from dataloader.data_utils import CIC_FLOW_DATASETS
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'] + list(CIC_FLOW_DATASETS))
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #for fact
    parser.add_argument('-balance', type=float, default=1.0)
    parser.add_argument('-loss_iter', type=int, default=200)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')
    parser.add_argument('-no_eval', action='store_true', help='skip in-loop evaluation during training; run test.py afterwards for the main test run')

    # about training
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    
    # wandb configuration
    parser.add_argument('--use_wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='FACT-FSCIL', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (username or team)')
    parser.add_argument('--wandb_cm_freq', type=int, default=20, help='log lightweight confusion-matrix image every N base epochs (0 disables)')

    # config / DVC / Optuna integration
    parser.add_argument('--config', type=str, default=None, help='YAML config file (e.g. params.yaml). Priority: defaults < --config < --opts < explicit CLI flags')
    parser.add_argument('--opts', nargs='*', default=[], metavar='KEY=VALUE', help='override config values, e.g. --opts lr_base=0.05 epochs_base=50 milestones=60,70')
    parser.add_argument('--metrics-file', type=str, default=None, help='fixed path for metrics JSON (for DVC metrics). Default: <save_path>/metrics.json')
    parser.add_argument('--save-config-name', type=str, default='config.yaml', help='resolved-config filename saved into save_path')

    return parser


def build_args(arg_list=None, overrides=None):
    """Build args Namespace with precedence: defaults < --config < --opts < explicit CLI < overrides dict."""
    parser = get_command_line_parser()
    args = parser.parse_args(arg_list)
    argv = sys.argv[1:] if arg_list is None else arg_list
    explicit = explicit_cli_keys(argv)
    if args.config:
        cfg = load_config_file(args.config)
        apply_config(args, cfg, explicit, source=args.config)
    for item in (args.opts or []):
        if '=' not in item:
            raise ValueError(f'--opts must be KEY=VALUE, got: {item}')
        k, v = item.split('=', 1)
        k = k.strip()
        if not hasattr(args, k):
            raise ValueError(f'--opts unknown key: {k}')
        setattr(args, k, parse_opt_value(v))
    if overrides:
        for k, v in overrides.items():
            if not hasattr(args, k):
                raise ValueError(f'override unknown key: {k}')
            setattr(args, k, v)
    return args


def run_training(args):
    set_seed(args.seed)
    pprint(vars(args))

    # Initialize wandb if requested
    if args.use_wandb:
        run_name = f"{args.dataset}_{args.project}_session{args.start_session}"
        config_dict = vars(args)

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config_dict,
            notes="Few-Shot Class Incremental Learning with FACT"
        )

    if args.gpu == "":
        args.device = "cpu"
        args.num_gpu = 0
    else:
        args.device = "cuda"
        args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()

    # Save resolved config next to checkpoints (reproducibility / DVC)
    try:
        save_config(args, os.path.join(args.save_path, args.save_config_name))
    except Exception as e:
        print(f'WARNING: could not save resolved config: {e}')

    # Metrics JSON for DVC metrics / Optuna objective
    metrics_path = args.metrics_file or os.path.join(args.save_path, 'metrics.json')
    ensure_path(os.path.dirname(metrics_path) or '.')
    metrics = write_metrics(trainer, args, metrics_path)

    # Log final results and finish wandb run
    if args.use_wandb:
        # Save the best metrics to wandb
        final_metrics = {
            'best_acc_session_0': trainer.trlog['max_acc'][0],
            'best_epoch_base': trainer.trlog['max_acc_epoch'],
        }

        # Handle max_acc as a list - log individual session accuracies for chart
        max_acc = trainer.trlog['max_acc']
        if isinstance(max_acc, (list, tuple)):
            # Log average accuracy across all sessions
            avg_accuracy = sum(float(acc) for acc in max_acc) / len(max_acc)
            final_metrics['avg_test_accuracy'] = float(avg_accuracy)

            # Also log each session's accuracy separately for reference
            for i, acc in enumerate(max_acc):
                final_metrics[f'session_{i}_acc'] = float(acc)
        else:
            final_metrics['avg_test_accuracy'] = float(max_acc.tolist()[0])

        wandb.log({**final_metrics, **metrics})
        wandb.finish()
    return trainer, metrics


if __name__ == '__main__':
    args = build_args()
    run_training(args)