# Configuration Guide for CVPR22-Fact

This guide explains how to use `params.yaml` to manage configuration for training and data preparation.

## Overview

All configuration parameters are centralized in `params.yaml`. This includes:
- Dataset configuration (classes, sessions, shots)
- Training hyperparameters (learning rate, batch size, epochs)
- Model settings (modes, temperature)
- System settings (GPU, workers, seed)

## Configuration File Structure

### 1. Dataset Configuration

```yaml
dataset:
  name: 'cicids2017_improved'
  dataroot: 'data/'

  sessions:
    base_class: 4              # Number of base classes
    num_classes: 10            # Total classes
    way: 1                     # Classes per incremental session
    shot: 5                    # Samples per new class
    total_sessions: 7          # Total sessions

  cicids2017:
    base_classes:              # Classes for base session
      - 'BENIGN'
      - 'DDoS'
      - 'DoS'
      - 'Portscan'

    incremental_classes:       # Classes for incremental sessions
      - 'Botnet'
      - 'FTP-Patator'
      - 'Heartbleed'
      - 'Infiltration'
      - 'SSH-Patator'
      - 'Web Attack'
```

### 2. Training Configuration

```yaml
training:
  project: 'fact'

  base:
    epochs: 400
    learning_rate: 0.005
    batch_size: 256

  incremental:
    epochs: 100
    learning_rate: 0.1
    batch_size: 0              # 0 = use all samples

  optimizer:
    momentum: 0.9
    weight_decay: 0.0005

  scheduler:
    type: 'Milestone'
    milestones: [50, 100, 150, 200, 250, 300]
    gamma: 0.25

  model:
    base_mode: 'ft_cos'
    new_mode: 'avg_cos'
    temperature: 16

  fact:
    balance: 0.01
    loss_iter: 0
    alpha: 2.0
    eta: 0.1
```

### 3. System Configuration

```yaml
system:
  gpu: '0,1,2,3'
  num_workers: 8
  seed: 1
  debug: false

checkpoint:
  save_path: 'checkpoint'
  model_dir: null
  start_session: 0
```

### 4. Index Generation Configuration

```yaml
index_generation:
  output_dir: 'data/index_list/cicids2017_improved'
  random_seed: 42
  shots_per_class: 5
```

## Usage

### 1. Creating Session Index Files

The `create_session_indices.py` script uses configuration from `params.yaml`:

```bash
uv run python create_session_indices.py
```

This will:
- Load configuration from `params.yaml`
- Create session index files based on the specified base and incremental classes
- Save files to the configured output directory

### 2. Training Models

The `train.py` script can load configuration from YAML:

```bash
# Use default params.yaml
uv run python train.py

# Use custom config file
uv run python train.py -config my_config.yaml

# Override specific parameters via command line
uv run python train.py -dataset cicids2017_improved -epochs_base 200
```

**Priority**: Command line arguments override YAML configuration.

### 3. Modifying Configuration

To change dataset or training parameters:

1. Edit `params.yaml`
2. Run scripts - they will automatically use the new configuration

Example: Change number of base classes
```yaml
dataset:
  sessions:
    base_class: 6  # Changed from 4 to 6
```

Example: Change learning rate
```yaml
training:
  base:
    learning_rate: 0.01  # Changed from 0.005
```

## Configuration for Different Datasets

### CIFAR100

```yaml
dataset:
  name: 'cifar100'
  sessions:
    base_class: 60
    num_classes: 100
    way: 5
    shot: 5
    total_sessions: 9
```

### CUB200

```yaml
dataset:
  name: 'cub200'
  sessions:
    base_class: 100
    num_classes: 200
    way: 10
    shot: 5
    total_sessions: 11
```

### MiniImageNet

```yaml
dataset:
  name: 'mini_imagenet'
  sessions:
    base_class: 60
    num_classes: 100
    way: 5
    shot: 5
    total_sessions: 9
```

## Advanced Usage

### Using Multiple Configuration Files

You can maintain different configuration files for different experiments:

```bash
# Experiment 1: 4 base classes
uv run python train.py -config configs/exp1_4base.yaml

# Experiment 2: 6 base classes
uv run python train.py -config configs/exp2_6base.yaml

# Experiment 3: Different learning rate
uv run python train.py -config configs/exp3_lr001.yaml
```

### Validating Configuration

Load and print configuration to verify:

```bash
uv run python config_utils.py
```

This will display the entire configuration structure.

### Command Line Override Examples

```bash
# Override GPU
uv run python train.py -gpu 0,1

# Override epochs
uv run python train.py -epochs_base 200 -epochs_new 50

# Override batch size
uv run python train.py -batch_size_base 128

# Debug mode
uv run python train.py -debug
```

## Configuration Priority

Configuration values are applied in this order (later overrides earlier):

1. Default values in `train.py`
2. Values from `params.yaml`
3. Command line arguments

## Benefits of YAML Configuration

1. **Centralized**: All parameters in one place
2. **Version Control**: Easy to track configuration changes with git
3. **Reproducibility**: Save exact configuration for each experiment
4. **Readability**: Clear, human-readable format
5. **Flexibility**: Can still override with command line args

## File Locations

- Main config: `params.yaml`
- Config utilities: `config_utils.py`
- Training script: `train.py`
- Index generation: `create_session_indices.py`
- Model checkpoints: `checkpoint/`
- Session indices: `data/index_list/cicids2017_improved/`

## Troubleshooting

### Config file not found

```
Config file params.yaml not found. Using command line arguments only.
```

**Solution**: Ensure `params.yaml` exists in the project root, or specify path with `-config`

### Invalid YAML syntax

```
Error loading configuration: ...
```

**Solution**: Check YAML syntax (indentation, colons, quotes)

### Missing dependencies

```
ModuleNotFoundError: No module named 'yaml'
```

**Solution**: Install dependencies:
```bash
uv sync
```

## Best Practices

1. **Version control**: Commit `params.yaml` with each experiment
2. **Backup**: Keep copies of working configurations
3. **Document changes**: Add comments in YAML to explain parameter choices
4. **Validate**: Always verify configuration before long training runs
5. **Experiment naming**: Use descriptive names for custom config files
