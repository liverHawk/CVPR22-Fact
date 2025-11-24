# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of FACT (Forward Compatible Training) for Few-Shot Class-Incremental Learning (FSCIL) from CVPR 2022. The codebase trains neural networks to incrementally learn new classes with limited samples while avoiding catastrophic forgetting.

**Paper**: "Forward Compatible Few-Shot Class-Incremental Learning" by Zhou et al.

## Development Workflow

### Package Management
This project uses `uv` for dependency management:
- Install dependencies: `uv sync`
- Run Python scripts: `uv run python <script>.py`

### DVC Pipeline

The project uses DVC (Data Version Control) for ML pipeline management with three stages:

1. **prepare_data**: Split CICIDS2017_improved into train/test
2. **create_session_files**: Generate session files for incremental learning
3. **train**: Execute the training pipeline

Run pipeline stages with:
```bash
dvc repro <stage_name>
```

### Training Commands

Training is configured primarily through `params.yaml` but can be overridden via command-line arguments.

**Using Makefile (recommended)**:
```bash
# Quick debug run (10 epochs, 64 batch)
make train_debug

# Custom training with variables
make train TRAIN_PROJECT=fact TRAIN_DATASET=CICIDS2017_improved TRAIN_EPOCHS_BASE=100

# Benchmark datasets (CIFAR100, CUB200, miniImageNet)
make train_fact_cifar
make train_fact_cub CUB_DATAROOT=/path/to/cub200
make train_fact_mini MINI_IMAGENET_ROOT=/path/to/mini_imagenet
```

**Direct Python execution**:
```bash
uv run python train.py -project fact -dataset CICIDS2017_improved -encoder mlp -base_mode ft_cos -new_mode avg_cos
```

### Data Preparation

**Split CICIDS2017 dataset**:
```bash
make split_cicids SPLIT_TEST_SIZE=0.2 SEED=1
```

**Create session files for incremental learning**:
```bash
make create_cicids_sessions SESSION_BASE_CLASS=15 SESSION_NUM_CLASSES=27 SESSION_WAY=2 SESSION_SHOT=5
```

## Architecture

### Project Structure

```
models/
├── base/          # Baseline FSCIL implementation
├── fact/          # FACT method with virtual prototypes
├── mlp_encoder.py       # MLP encoder for tabular data
├── cnn1d_encoder.py     # 1D CNN encoder for sequences
├── resnet18_encoder.py  # ResNet-18 for images
└── resnet20_cifar.py    # ResNet-20 for CIFAR

dataloader/
├── data_utils.py     # Dataset setup and dataloader creation
├── sampler.py        # Few-shot episode sampling
├── cicids2017/       # CICIDS2017 intrusion detection dataset
├── cifar100/         # CIFAR-100
├── cub200/           # CUB-200 birds
├── miniimagenet/     # Mini-ImageNet
├── imagenet100/      # ImageNet-100
└── imagenet1000/     # ImageNet-1000

checkpoint/           # Saved model weights and training logs
data/                # Datasets and session splits
```

### Key Concepts

**Few-Shot Class-Incremental Learning (FSCIL)**:
- **Base session**: Train on many samples from base classes
- **Incremental sessions**: Sequentially learn new classes with only a few samples (few-shot)
- **Challenge**: Avoid catastrophic forgetting while learning from limited data

**FACT Method**:
- Uses **virtual prototypes** to reserve embedding space for future classes
- Implements **forward compatibility** by preparing the model for future updates
- Two training modes:
  - `base_mode`: For base session (typically `ft_cos` = fine-tune with cosine classifier)
  - `new_mode`: For incremental sessions (typically `avg_cos` = average embeddings with cosine classifier)

### Training Pipeline

1. **Initialization**: Load dataset and configure model based on `project` parameter (`base` or `fact`)
2. **Base Session**: Train on base classes with standard supervised learning
3. **Incremental Sessions**: For each new session:
   - Load few-shot samples for new classes
   - Update model with minimal forgetting of old classes
   - Evaluate on all classes seen so far
4. **Evaluation**: Measure accuracy across all sessions

### Configuration System

**Priority order** (highest to lowest):
1. Command-line arguments
2. `params.yaml` configuration
3. Dataset-specific presets in `params.yaml`
4. Hardcoded defaults

**Key parameters**:
- `project`: `base` or `fact` (training method)
- `dataset`: `CICIDS2017_improved`, `cifar100`, `cub200`, `mini_imagenet`
- `encoder`: `mlp` (tabular), `cnn1d` (sequences), or `resnet18/resnet20` (images)
- `base_mode`/`new_mode`: `ft_cos` (cosine classifier) or `ft_dot` (linear classifier)
- `epochs_base`/`epochs_new`: Training epochs for base/incremental sessions
- `balance`, `alpha`, `eta`: FACT-specific hyperparameters for virtual prototypes

### Encoder Selection

- **MLP** (`mlp`): For tabular/network traffic data (CICIDS2017)
- **CNN1D** (`cnn1d`): For sequential data with temporal patterns
- **ResNet-18** (`resnet18`): For standard image datasets
- **ResNet-20** (`resnet20`): Optimized for CIFAR-100

### Dataloader Architecture

Each dataset module implements:
- **Dataset class**: Loads data and handles indexing
- **SelectfromClasses**: Filters data by class indices
- **SelectfromTxt**: Loads data indices from session files
- **Normalization**: Applied based on `normalize_method` (`standard`, `minmax`, `moving_minmax`)

Session files in `data/index_list/<dataset>/` define which samples belong to each incremental session.

### Logging

The codebase supports two experiment tracking systems:
- **Comet ML**: Initialized by default in `train.py:307`
- **Weights & Biases**: Enabled with `--use_wandb` flag or `params.yaml` configuration

## Dataset Requirements

**CICIDS2017_improved**: Custom network intrusion detection dataset
- Preprocessed CSV files with labeled network traffic features
- Split into train/test with `split_cicids2017.py`

**Standard benchmarks**: Follow setup instructions from [CEC repository](https://github.com/icoz69/CEC-CVPR2021)
- CIFAR-100, CUB-200, miniImageNet: Download and place in `data/` directory
- ImageNet-100/1000: Use splits from [Google Drive](https://drive.google.com/drive/folders/1IBjVEmwmLBdABTaD6cDbrdHMXfHHtFvU)

## Utilities

**Makefile targets**:
- `make help`: Show all available targets
- `make show_paths`: Display configured paths
- `make clean_pycache`: Remove Python cache files
- `make clean_checkpoints`: Remove saved checkpoints (interactive confirmation)

**Important files**:
- `utils.py`: Helper functions (metrics, logging, device setup)
- `dvc_workaround.py`: Fixes stdout buffering issues with DVC
- `params.yaml`: Centralized configuration
- `dvc.yaml`: Pipeline definition
