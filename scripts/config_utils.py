"""
Utility functions for loading and managing configuration from params.yaml
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = 'params.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dataset configuration."""
    return config.get('dataset', {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration."""
    return config.get('training', {})


def get_system_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract system configuration."""
    return config.get('system', {})


def get_checkpoint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract checkpoint configuration."""
    return config.get('checkpoint', {})


def get_index_generation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract index generation configuration."""
    return config.get('index_generation', {})


def merge_args_with_config(args, config: Dict[str, Any]):
    """
    Merge argparse arguments with configuration from YAML.
    YAML values take precedence unless command line args are explicitly provided.

    Args:
        args: argparse.Namespace object
        config: Configuration dictionary from YAML

    Returns:
        Updated args object
    """
    dataset_cfg = get_dataset_config(config)
    training_cfg = get_training_config(config)
    system_cfg = get_system_config(config)
    checkpoint_cfg = get_checkpoint_config(config)

    # Dataset settings
    if hasattr(args, 'dataset') and args.dataset == 'cifar100':
        # Only override if default value
        args.dataset = dataset_cfg.get('name', args.dataset)
    args.dataroot = dataset_cfg.get('dataroot', args.dataroot)

    # Training settings
    args.project = training_cfg.get('project', args.project)
    args.epochs_base = training_cfg.get('base', {}).get('epochs', args.epochs_base)
    args.epochs_new = training_cfg.get('incremental', {}).get('epochs', args.epochs_new)
    args.lr_base = training_cfg.get('base', {}).get('learning_rate', args.lr_base)
    args.lr_new = training_cfg.get('incremental', {}).get('learning_rate', args.lr_new)
    args.batch_size_base = training_cfg.get('base', {}).get('batch_size', args.batch_size_base)
    args.batch_size_new = training_cfg.get('incremental', {}).get('batch_size', args.batch_size_new)

    # Optimizer settings
    opt_cfg = training_cfg.get('optimizer', {})
    args.momentum = opt_cfg.get('momentum', args.momentum)
    args.decay = opt_cfg.get('weight_decay', args.decay)

    # Scheduler settings
    sched_cfg = training_cfg.get('scheduler', {})
    args.schedule = sched_cfg.get('type', args.schedule)
    args.milestones = sched_cfg.get('milestones', args.milestones)
    args.gamma = sched_cfg.get('gamma', args.gamma)
    args.step = sched_cfg.get('step', args.step)

    # Model settings
    model_cfg = training_cfg.get('model', {})
    args.base_mode = model_cfg.get('base_mode', args.base_mode)
    args.new_mode = model_cfg.get('new_mode', args.new_mode)
    args.temperature = model_cfg.get('temperature', args.temperature)
    if model_cfg.get('not_data_init', False):
        args.not_data_init = True

    # FACT-specific parameters
    fact_cfg = training_cfg.get('fact', {})
    args.balance = fact_cfg.get('balance', args.balance)
    args.loss_iter = fact_cfg.get('loss_iter', args.loss_iter)
    args.alpha = fact_cfg.get('alpha', args.alpha)
    args.eta = fact_cfg.get('eta', args.eta)

    # Evaluation settings
    eval_cfg = config.get('evaluation', {})
    args.test_batch_size = eval_cfg.get('test_batch_size', args.test_batch_size)
    if not eval_cfg.get('validation', True):
        args.set_no_val = True

    # System settings
    args.gpu = system_cfg.get('gpu', args.gpu)
    args.num_workers = system_cfg.get('num_workers', args.num_workers)
    args.seed = system_cfg.get('seed', args.seed)
    if system_cfg.get('debug', False):
        args.debug = True

    # Checkpoint settings
    args.model_dir = checkpoint_cfg.get('model_dir', args.model_dir)
    args.start_session = checkpoint_cfg.get('start_session', args.start_session)

    return args


def print_config(config: Dict[str, Any], indent: int = 0):
    """
    Pretty print configuration.

    Args:
        config: Configuration dictionary
        indent: Indentation level
    """
    for key, value in config.items():
        if isinstance(value, dict):
            print('  ' * indent + f'{key}:')
            print_config(value, indent + 1)
        else:
            print('  ' * indent + f'{key}: {value}')


if __name__ == '__main__':
    # Test configuration loading
    # Change to parent directory to find params.yaml
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    config = load_config('params.yaml')
    print("Configuration loaded successfully!")
    print("\n" + "=" * 60)
    print("Configuration:")
    print("=" * 60)
    print_config(config)
