"""Optuna hyperparameter search for FACT.
Uses train.build_args/run_training so every train.py param is searchable/savable.

Space file format (YAML): every train.py / params.yaml parameter is sweepable.
  lr_base: {type: float, low: 0.001, high: 0.5, log: true}
  epochs_base: {type: int, low: 1, high: 200}
  schedule: {type: categorical, choices: [Step, Milestone, Cosine]}
  milestones: {type: categorical, choices: [[60, 70], [80, 90]]}  # lists OK
  not_data_init: {type: bool}

tune_space.yaml ships with the full parameter list (infra / one-shot keys are
commented there with caveats; uncomment to sweep them). Semantics:
  - keys in --opts are fixed overrides applied to EVERY trial and win over
    the space (a warning is printed on overlap).
  - seed: taken from the space/--opts if present, else --seed + trial number.
  - a parameter set identical to an earlier trial reuses that trial's value
    instead of re-training (protects shared checkpoint dirs from overwrites).
  - keys config/opts/metrics_file/save_config_name are CLI plumbing and are
    rejected in the space and in --opts.

Usage:
  uv run python tune.py --trials 20
  uv run python tune.py --space tune_space.yaml --base-config params.yaml --trials 5 --opts epochs_base=2 epochs_new=1 gpu="" num_workers=2
  uv run python tune.py --jobs 2 --trials 8     # 2 trials in parallel (shared-RNG caveat, see --jobs)
Results: best_params.yaml, best_metrics.json (+ optuna storage for resume).
"""
import argparse
import json
import os
import sys

import yaml

# train.py args that drive tune.py's own machinery; sweeping them would break
# the study or the output contract.
INFRA_KEYS = frozenset(['config', 'opts', 'metrics_file', 'save_config_name'])


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _require(spec, keys, name):
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f'space entry "{name}": missing required key(s) {missing}')


def suggest_params(trial, space):
    params = {}
    for name, spec in space.items():
        t = spec.get('type', 'categorical')
        if t == 'float':
            _require(spec, ('low', 'high'), name)
            params[name] = trial.suggest_float(name, spec['low'], spec['high'], log=spec.get('log', False))
        elif t == 'int':
            _require(spec, ('low', 'high'), name)
            params[name] = trial.suggest_int(name, spec['low'], spec['high'], log=spec.get('log', False))
        elif t == 'bool':
            params[name] = trial.suggest_categorical(name, [True, False])
        elif t == 'categorical':
            _require(spec, ('choices',), name)
            if any(isinstance(c, (list, dict)) for c in spec['choices']):
                # optuna warns that non-primitive choices are "not persistent",
                # but sqlite roundtrip of list choices is verified working on
                # optuna 5.x (milestones, etc.) -> silence only this advisory.
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore',
                        message='Choices for a categorical distribution should be')
                    params[name] = trial.suggest_categorical(name, spec['choices'])
            else:
                params[name] = trial.suggest_categorical(name, spec['choices'])
        else:
            raise ValueError(f'unknown space type for {name}: {t}')
    return params


def validate_keys(space, fixed):
    """Fail fast on unknown / plumbing keys before the study starts."""
    from train import get_command_line_parser
    dests = {a.dest for a in get_command_line_parser()._actions}
    for source, keys in (('space file', space), ('--opts', fixed)):
        unknown = sorted(set(keys) - dests)
        if unknown:
            raise SystemExit(f'ERROR: {source}: unknown train.py keys: {unknown}')
        infra = sorted(set(keys) & INFRA_KEYS)
        if infra:
            raise SystemExit(
                f'ERROR: {source}: plumbing keys cannot be swept: {infra} '
                f'(use --base-config / --opts for config paths, metrics paths are fixed per trial)')
    overlap = sorted(set(space) & set(fixed))
    if overlap:
        print(f'WARNING: keys in both space and --opts (--opts wins): {overlap}')


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Optuna tuning for FACT train.py')
    p.add_argument('--space', type=str, default='tune_space.yaml')
    p.add_argument('--base-config', type=str, default='params.yaml')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--study-name', type=str, default='fact')
    p.add_argument('--storage', type=str, default='sqlite:///optuna.db')
    p.add_argument('--objective', type=str, default='avg_acc', choices=['avg_acc', 'final_acc', 'session_0'])
    p.add_argument('--params-out', type=str, default='best_params.yaml')
    p.add_argument('--metrics-out', type=str, default='best_metrics.json')
    p.add_argument('--seed', type=int, default=1, help='base seed; trial i uses seed+i unless the space/--opts sets seed')
    p.add_argument('--opts', nargs='*', default=[], metavar='KEY=VALUE',
                   help='fixed overrides applied to every trial (win over the space; e.g. epochs_base=2 gpu="" num_workers=2)')
    p.add_argument('--jobs', type=int, default=1,
                   help='run this many trials in parallel inside this process (n_jobs). '
                        'Trials share the process-global RNG, so per-trial seed '
                        'reproducibility is only guaranteed with the default 1; '
                        'num_workers is divided by --jobs')
    return p.parse_args(argv)


def parse_opts(opts):
    from train import parse_opt_value
    out = {}
    for item in opts:
        if '=' not in item:
            raise ValueError(f'--opts must be KEY=VALUE, got: {item}')
        k, v = item.split('=', 1)
        out[k.strip()] = parse_opt_value(v)
    return out


def main(argv=None):
    import optuna
    from train import build_args, run_training

    args = parse_args(argv)
    space = load_yaml(args.space)
    base_cfg = load_yaml(args.base_config) if os.path.exists(args.base_config) else {}
    fixed = parse_opts(args.opts)
    validate_keys(space, fixed)

    # param signature -> (value, save_path, owning trial); identical parameter
    # sets reuse the first result instead of re-training into the same
    # checkpoint dir.
    seen = {}

    def objective(trial):
        suggested = suggest_params(trial, space)
        overrides = dict(suggested)
        overrides.update(fixed)  # --opts are fixed per-trial overrides
        # dedup on the EFFECTIVE params (after --opts win) so trials that only
        # differ in overridden keys reuse the first result instead of
        # re-training into the same checkpoint dir; auto-assigned seeds are
        # not part of the key (they differ per trial by design)
        sig = json.dumps(overrides, sort_keys=True, default=str)
        if sig in seen:
            value, save_path, owner = seen[sig]
            trial.set_user_attr('save_path', save_path)
            trial.set_user_attr('duplicate_of', owner)
            print(f'trial {trial.number}: identical params as trial {owner}, '
                  f'reusing value {value:.4f} (no training)')
            return value
        if 'seed' not in overrides:
            overrides['seed'] = args.seed + trial.number
        if args.jobs > 1:
            # split loader workers across the concurrent trials (4-core box)
            nw = int(overrides.get('num_workers', base_cfg.get('num_workers', 2)))
            overrides['num_workers'] = max(1, nw // args.jobs)
        # Build from base config file so DVC params stay in sync
        arg_list = ['--config', args.base_config] if os.path.exists(args.base_config) else []
        train_args = build_args(arg_list, overrides=overrides)
        _, metrics = run_training(train_args)
        value = metrics['avg_acc'] if args.objective == 'avg_acc' else (
            metrics['final_acc'] if args.objective == 'final_acc' else metrics['max_acc'][0])
        # save_path is injected into this Namespace by FSCILTrainer.set_save_path
        trial.set_user_attr('save_path', getattr(train_args, 'save_path', None))
        seen[sig] = (value, getattr(train_args, 'save_path', None), trial.number)
        return value

    storage = args.storage or None
    study = optuna.create_study(study_name=args.study_name, storage=storage,
                                load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=args.trials, n_jobs=args.jobs)

    best = dict(base_cfg)
    best.update(study.best_trial.params)   # space params (space wins over base cfg)
    best.update(parse_opts(args.opts))     # --opts are fixed for every trial
    if 'seed' not in best:
        # actual seed used by the objective: --seed + trial number
        best['seed'] = args.seed + study.best_trial.number
    with open(args.params_out, 'w') as f:
        yaml.safe_dump(best, f, sort_keys=True)
    with open(args.metrics_out, 'w') as f:
        json.dump({'best_value': study.best_value,
                   'best_trial': study.best_trial.number,
                   'objective': args.objective,
                   'params': study.best_trial.params}, f, indent=2)
    print(f'Best trial {study.best_trial.number}: {args.objective}={study.best_value:.4f}')
    print(f'Saved: {args.params_out}, {args.metrics_out}')
    return study


if __name__ == '__main__':
    sys.exit(main())
