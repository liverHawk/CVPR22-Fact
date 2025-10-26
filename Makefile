# CICIDS2017 Dataset Training and Evaluation Makefile
# Supports both base and fact models

# Default values
DATASET := cicids2017_improved
DATAROOT := data
EPOCHS_BASE := 1
EPOCHS_NEW := 1
SESSIONS := 7
START_SESSION := 0
BATCH_SIZE_BASE := 128
BATCH_SIZE_NEW := 16
TEST_BATCH_SIZE := 100
LR_BASE := 0.1
LR_NEW := 0.1
MAX_SAMPLES := 1000
DEBUG := false

# Model-specific settings
BASE_MODEL_DIR := checkpoint/$(DATASET)/base
FACT_MODEL_DIR := checkpoint/$(DATASET)/fact

# Output directories
BASE_OUTPUT := confusion_analysis/$(DATASET)_base
FACT_OUTPUT := confusion_analysis/$(DATASET)_fact

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help train-base train-fact eval-base eval-fact clean clean-checkpoints clean-analysis

# Default target
help:
	@echo "$(BLUE)CICIDS2017 Dataset Training and Evaluation Makefile$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@echo "  $(GREEN)train-base$(NC)     - Train base model"
	@echo "  $(GREEN)train-fact$(NC)     - Train fact model"
	@echo "  $(GREEN)eval-base$(NC)      - Evaluate base model with confusion analysis"
	@echo "  $(GREEN)eval-fact$(NC)      - Evaluate fact model with confusion analysis"
	@echo "  $(GREEN)train-all$(NC)      - Train both base and fact models"
	@echo "  $(GREEN)eval-all$(NC)       - Evaluate both base and fact models"
	@echo "  $(GREEN)clean$(NC)          - Clean all generated files"
	@echo "  $(GREEN)clean-checkpoints$(NC) - Clean only checkpoint files"
	@echo "  $(GREEN)clean-analysis$(NC) - Clean only analysis files"
	@echo ""
	@echo "$(YELLOW)Configuration variables:$(NC)"
	@echo "  DATASET=$(DATASET)"
	@echo "  DATAROOT=$(DATAROOT)"
	@echo "  EPOCHS_BASE=$(EPOCHS_BASE)"
	@echo "  EPOCHS_NEW=$(EPOCHS_NEW)"
	@echo "  MAX_SAMPLES=$(MAX_SAMPLES)"
	@echo "  DEBUG=$(DEBUG)"
	@echo ""
	@echo "$(YELLOW)Usage examples:$(NC)"
	@echo "  make train-base                    # Train base model with default settings"
	@echo "  make train-fact EPOCHS_BASE=3     # Train fact model with 3 base epochs"
	@echo "  make eval-base                     # Evaluate base model"
	@echo "  make train-all MAX_SAMPLES=3          # Train both models with 3 max samples"
	@echo "  make clean                         # Clean all files"

# Training targets
train-base:
	@echo "$(BLUE)Training base model for $(DATASET)...$(NC)"
	@mkdir -p $(BASE_MODEL_DIR)
	@echo "$(YELLOW)Note: 'Model file not found' message is normal for new training$(NC)"
	uv run python train.py \
		-dataset $(DATASET) \
		-dataroot $(DATAROOT) \
		-epochs_base $(EPOCHS_BASE) \
		-epochs_new $(EPOCHS_NEW) \
		-start_session $(START_SESSION) \
		-project base \
		-model base \
		-max_samples $(MAX_SAMPLES) \
		$(if $(filter true,$(DEBUG)),-debug)
	@echo "$(GREEN)Base model training completed!$(NC)"

train-fact:
	@echo "$(BLUE)Training fact model for $(DATASET)...$(NC)"
	@mkdir -p $(FACT_MODEL_DIR)
	@echo "$(YELLOW)Note: 'Model file not found' message is normal for new training$(NC)"
	uv run python train.py \
		-dataset $(DATASET) \
		-dataroot $(DATAROOT) \
		-epochs_base $(EPOCHS_BASE) \
		-epochs_new $(EPOCHS_NEW) \
		-start_session $(START_SESSION) \
		-model fact \
		-project fact \
		-max_samples $(MAX_SAMPLES) \
		$(if $(filter true,$(DEBUG)),-debug)
	@echo "$(GREEN)Fact model training completed!$(NC)"

# Evaluation targets
eval-base:
	@echo "$(BLUE)Evaluating base model for $(DATASET)...$(NC)"
	@if [ ! -d "$(BASE_MODEL_DIR)" ]; then \
		echo "$(RED)Error: Base model checkpoint directory not found: $(BASE_MODEL_DIR)$(NC)"; \
		echo "$(YELLOW)Please run 'make train-base' first.$(NC)"; \
		exit 1; \
	fi
	@mkdir -p $(BASE_OUTPUT)
	@BASE_CHECKPOINT_DIR=$$(find $(BASE_MODEL_DIR) -name "session0_max_acc.pth" -exec dirname {} \; | head -1); \
	if [ -z "$$BASE_CHECKPOINT_DIR" ]; then \
		echo "$(RED)Error: No base model checkpoint found in $(BASE_MODEL_DIR)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Using checkpoint directory: $$BASE_CHECKPOINT_DIR$(NC)"; \
	uv run python run_confusion_analysis.py \
		-dataset $(DATASET) \
		-dataroot $(DATAROOT) \
		-checkpoint_dir "$$BASE_CHECKPOINT_DIR" \
		-output_dir $(BASE_OUTPUT) \
		-sessions $$(seq 0 $$(($(SESSIONS)-1))) \
		-batch_size_base $(BATCH_SIZE_BASE) \
		-batch_size_new $(BATCH_SIZE_NEW) \
		-test_batch_size $(TEST_BATCH_SIZE)
	@echo "$(GREEN)Base model evaluation completed! Results saved to $(BASE_OUTPUT)$(NC)"

eval-fact:
	@echo "$(BLUE)Evaluating fact model for $(DATASET)...$(NC)"
	@if [ ! -d "$(FACT_MODEL_DIR)" ]; then \
		echo "$(RED)Error: Fact model checkpoint directory not found: $(FACT_MODEL_DIR)$(NC)"; \
		echo "$(YELLOW)Please run 'make train-fact' first.$(NC)"; \
		exit 1; \
	fi
	@mkdir -p $(FACT_OUTPUT)
	@FACT_CHECKPOINT_DIR=$$(find $(FACT_MODEL_DIR) -name "session0_max_acc.pth" -exec dirname {} \; | head -1); \
	if [ -z "$$FACT_CHECKPOINT_DIR" ]; then \
		echo "$(RED)Error: No fact model checkpoint found in $(FACT_MODEL_DIR)$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Using checkpoint directory: $$FACT_CHECKPOINT_DIR$(NC)"; \
	uv run python run_confusion_analysis.py \
		-dataset $(DATASET) \
		-dataroot $(DATAROOT) \
		-checkpoint_dir "$$FACT_CHECKPOINT_DIR" \
		-output_dir $(FACT_OUTPUT) \
		-sessions $$(seq 0 $$(($(SESSIONS)-1))) \
		-batch_size_base $(BATCH_SIZE_BASE) \
		-batch_size_new $(BATCH_SIZE_NEW) \
		-test_batch_size $(TEST_BATCH_SIZE)
	@echo "$(GREEN)Fact model evaluation completed! Results saved to $(FACT_OUTPUT)$(NC)"

# Combined targets
train-all: train-base train-fact
	@echo "$(GREEN)All models training completed!$(NC)"

eval-all: eval-base eval-fact
	@echo "$(GREEN)All models evaluation completed!$(NC)"

# Quick training (debug mode with fewer epochs)
quick-base:
	@echo "$(BLUE)Quick training base model (debug mode)...$(NC)"
	@$(MAKE) train-base EPOCHS_BASE=1 EPOCHS_NEW=1 SESSIONS=3 DEBUG=true

quick-fact:
	@echo "$(BLUE)Quick training fact model (debug mode)...$(NC)"
	@$(MAKE) train-fact EPOCHS_BASE=1 EPOCHS_NEW=1 SESSIONS=3 DEBUG=true

quick-all: quick-base quick-fact
	@echo "$(GREEN)Quick training completed!$(NC)"

# Clean targets
clean-checkpoints:
	@echo "$(YELLOW)Cleaning checkpoint files...$(NC)"
	@rm -rf checkpoint/$(DATASET)
	@echo "$(GREEN)Checkpoint files cleaned!$(NC)"

clean-analysis:
	@echo "$(YELLOW)Cleaning analysis files...$(NC)"
	@rm -rf confusion_analysis/$(DATASET)_*
	@echo "$(GREEN)Analysis files cleaned!$(NC)"

clean: clean-checkpoints clean-analysis
	@echo "$(GREEN)All generated files cleaned!$(NC)"

# Status check
status:
	@echo "$(BLUE)Status check for $(DATASET):$(NC)"
	@echo ""
	@echo "$(YELLOW)Checkpoint directories:$(NC)"
	@if [ -d "$(BASE_MODEL_DIR)" ]; then \
		echo "$(GREEN)✓ Base model: $(BASE_MODEL_DIR)$(NC)"; \
		echo "  Sessions available: $$(find $(BASE_MODEL_DIR) -name "session*_max_acc.pth" | wc -l)"; \
	else \
		echo "$(RED)✗ Base model: Not found$(NC)"; \
	fi
	@if [ -d "$(FACT_MODEL_DIR)" ]; then \
		echo "$(GREEN)✓ Fact model: $(FACT_MODEL_DIR)$(NC)"; \
		echo "  Sessions available: $$(find $(FACT_MODEL_DIR) -name "session*_max_acc.pth" | wc -l)"; \
	else \
		echo "$(RED)✗ Fact model: Not found$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Analysis directories:$(NC)"
	@if [ -d "$(BASE_OUTPUT)" ]; then \
		echo "$(GREEN)✓ Base analysis: $(BASE_OUTPUT)$(NC)"; \
	else \
		echo "$(RED)✗ Base analysis: Not found$(NC)"; \
	fi
	@if [ -d "$(FACT_OUTPUT)" ]; then \
		echo "$(GREEN)✓ Fact analysis: $(FACT_OUTPUT)$(NC)"; \
	else \
		echo "$(RED)✗ Fact analysis: Not found$(NC)"; \
	fi

# Development helpers
dev-setup:
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@uv sync
	@echo "$(GREEN)Development environment ready!$(NC)"

# Test data loading
test-data:
	@echo "$(BLUE)Testing data loading for $(DATASET)...$(NC)"
	@uv run python -c "import sys; sys.path.append('.'); from dataloader.cicids2017_improved.cicids2017_improved import CICIDS2017Improved; import numpy as np; dataset = CICIDS2017Improved(root='$(DATAROOT)', train=True, index='data/index_list/$(DATASET)/session_0.txt', max_samples=100); print(f'Dataset size: {len(dataset)}'); print(f'Unique classes: {np.unique(dataset.targets)}'); print('Data loading test passed!')"
	@echo "$(GREEN)Data loading test completed!$(NC)"
