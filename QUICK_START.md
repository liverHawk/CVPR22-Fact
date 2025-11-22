# Quick Start Guide

## Installation

```bash
# Install dependencies
uv sync
```

## Configuration

All parameters are managed in `params.yaml`. See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for details.

### Key Configuration Files

- **params.yaml**: Main configuration file
- **checkpoint/**: Model save directory
- **data/**: Dataset directory

## Usage

### 1. Prepare Session Indices

```bash
# Generate session index files from params.yaml
uv run python create_session_indices.py
```

This creates session files in `data/index_list/cicids2017_improved/`:
- `session_0.txt`: Base session (4 classes)
- `session_1.txt` to `session_6.txt`: Incremental sessions (1 class each)

### 2. Train Model

```bash
# Train with default configuration (params.yaml)
uv run python train.py

# Train with custom config
uv run python train.py -config my_config.yaml

# Override specific parameters
uv run python train.py -epochs_base 200 -gpu 0,1
```

### 3. Output Files

After training, the following files will be saved in `checkpoint/`:

- **best_model.pth**: Best model from base session
- **session_N.pth**: Model checkpoints for incremental sessions
- **optimizer.pth**: Optimizer state
- **results.txt**: Training results and accuracies
- **session_N_confusion_matrix.pdf**: Confusion matrix for each session
- **session_N_confusion_matrix_cbar.pdf**: Confusion matrix with colorbar

For CICIDS2017_improved dataset, confusion matrices include class names (BENIGN, DDoS, DoS, etc.) for better interpretability.

### 4. Test Dataset Loading

```bash
# Test CICIDS2017_improved dataset
uv run python tests/test_cicids2017.py

# Test confusion matrix generation
uv run python tests/test_confusion_matrix.py

# Test detailed session loading
uv run python test_session_detailed.py
```

## Default Configuration (CICIDS2017_improved)

- **Base classes**: 4 (BENIGN, DDoS, DoS, Portscan)
- **Incremental classes**: 6 (Botnet, FTP-Patator, Heartbleed, Infiltration, SSH-Patator, Web Attack)
- **Total sessions**: 7 (1 base + 6 incremental)
- **Shot**: 5 samples per new class

## Model Checkpoints

Models are saved to `checkpoint/`:
- `best_model.pth`: Best base session model
- `optimizer.pth`: Optimizer state
- `session_N.pth`: Incremental session N model
- `results.txt`: Training results log

## Common Commands

```bash
# View configuration
uv run python config_utils.py

# Create session indices
uv run python create_session_indices.py

# Train on CICIDS2017
uv run python train.py -dataset cicids2017_improved

# Train on CIFAR100
uv run python train.py -dataset cifar100

# Debug mode
uv run python train.py -debug
```

## Customization

Edit `params.yaml` to change:
- Dataset and session configuration
- Training hyperparameters
- Model architecture settings
- System settings (GPU, workers)

See [CONFIG_GUIDE.md](CONFIG_GUIDE.md) for comprehensive documentation.

## Project Structure

```
CVPR22-Fact/
├── params.yaml                 # Main configuration
├── config_utils.py             # Config loading utilities
├── train.py                    # Training script
├── create_session_indices.py  # Session index generator
├── checkpoint/                 # Model checkpoints
│   ├── best_model.pth
│   ├── optimizer.pth
│   └── session_*.pth
├── data/
│   ├── CICIDS2017_improved/   # Dataset files
│   └── index_list/
│       └── cicids2017_improved/  # Session indices
├── dataloader/
│   └── cicids2017/
│       └── cicids2017.py      # CICIDS dataset loader
└── models/
    ├── base/                   # Base trainer
    └── fact/                   # FACT trainer
```

## Troubleshooting

**Missing dependencies:**
```bash
uv sync
```

**Config file not found:**
```bash
# Check file exists
ls params.yaml

# Or specify path
uv run python train.py -config path/to/config.yaml
```

**CUDA out of memory:**
```bash
# Reduce batch size in params.yaml
training:
  base:
    batch_size: 128  # Reduce from 256
```

For more details, see [CONFIG_GUIDE.md](CONFIG_GUIDE.md).
