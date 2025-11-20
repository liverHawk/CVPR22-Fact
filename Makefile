SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PYTHON := uv run python

DATA_DIR ?= $(PROJECT_ROOT)/data
CHECKPOINT_DIR ?= $(PROJECT_ROOT)/checkpoint
CICIDS_DIR ?= $(DATA_DIR)/CICIDS2017_improved
CICIDS_INDEX_DIR ?= $(DATA_DIR)/index_list/CICIDS2017_improved
CUB_DATAROOT ?= $(DATA_DIR)/cub200
MINI_IMAGENET_ROOT ?= $(DATA_DIR)/mini_imagenet

GPU ?= 0
SEED ?= 1

TRAIN_PROJECT ?= fact
TRAIN_DATASET ?= CICIDS2017_improved
TRAIN_ENCODER ?= mlp
TRAIN_BASE_MODE ?= ft_cos
TRAIN_NEW_MODE ?= avg_cos
TRAIN_BATCH_SIZE ?= 128
TRAIN_TEST_BATCH_SIZE ?= 100
TRAIN_EPOCHS_BASE ?= 100
TRAIN_EPOCHS_NEW ?= 100
TRAIN_START_SESSION ?= 0
TRAIN_EXTRA ?=

FACT_CIFAR_EXTRA ?= -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 600 -schedule Cosine -temperature 16 -batch_size_base 256 -balance 0.001 -loss_iter 0 -alpha 0.5 -gpu $(GPU)
FACT_CUB_EXTRA ?= -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -decay 0.0005 -epochs_base 400 -schedule Milestone -milestones 50 100 150 200 250 300 -temperature 16 -batch_size_base 256 -balance 0.01 -loss_iter 0 -gpu $(GPU)
FACT_MINI_EXTRA ?= -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 1000 -schedule Cosine -temperature 16 -alpha 0.5 -balance 0.01 -loss_iter 150 -eta 0.1 -gpu $(GPU)

WAND_ENABLE ?= 0
WAND_PROJECT ?= origin-fact
WAND_ENTITY ?=
WAND_GROUP ?=
WAND_RUN_NAME ?=
WAND_TAGS ?=
WAND_MODE ?= online
WAND_WATCH ?= gradients
WAND_WATCH_FREQ ?= 100

ifeq ($(WAND_ENABLE),1)
  WAND_ARGS := --use_wandb --wandb_mode $(WAND_MODE) --wandb_watch $(WAND_WATCH) --wandb_watch_freq $(WAND_WATCH_FREQ)
  ifneq ($(strip $(WAND_PROJECT)),)
    WAND_ARGS += --wandb_project $(WAND_PROJECT)
  endif
  ifneq ($(strip $(WAND_ENTITY)),)
    WAND_ARGS += --wandb_entity $(WAND_ENTITY)
  endif
  ifneq ($(strip $(WAND_GROUP)),)
    WAND_ARGS += --wandb_group $(WAND_GROUP)
  endif
  ifneq ($(strip $(WAND_RUN_NAME)),)
    WAND_ARGS += --wandb_run_name $(WAND_RUN_NAME)
  endif
  ifneq ($(strip $(WAND_TAGS)),)
    WAND_ARGS += --wandb_tags $(WAND_TAGS)
  endif
else
  WAND_ARGS :=
endif

SPLIT_TEST_SIZE ?= 0.2
CICIDS_STRATIFY ?= 1
SESSION_BASE_CLASS ?= 15
SESSION_NUM_CLASSES ?= 27
SESSION_WAY ?= 2
SESSION_SHOT ?= 5

WILDCARD := $(if $(filter 1,$(CICIDS_STRATIFY)),,--no_stratify)

.PHONY: help setup train train_fact_cifar train_fact_cub train_fact_mini train_debug \
	create_cicids_sessions split_cicids clean_pycache clean_checkpoints show_paths

help: ## 利用可能なターゲット一覧を表示
	@echo "Usage: make <target> [VAR=value]..."
	@echo ""
	@echo "主要ターゲット:"
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | sed -e 's/:.*##/: ##/' | column -t -s '##'

setup: ## uvの依存関係を同期
	cd $(PROJECT_ROOT) && uv sync

train: ## 変数で指定した設定でtrain.pyを実行
	cd $(PROJECT_ROOT) && \
	$(PYTHON) train.py \
		-project $(TRAIN_PROJECT) \
		-dataset $(TRAIN_DATASET) \
		-dataroot $(DATA_DIR) \
		-encoder $(TRAIN_ENCODER) \
		-base_mode $(TRAIN_BASE_MODE) \
		-new_mode $(TRAIN_NEW_MODE) \
		-epochs_base $(TRAIN_EPOCHS_BASE) \
		-epochs_new $(TRAIN_EPOCHS_NEW) \
		-batch_size_base $(TRAIN_BATCH_SIZE) \
		-test_batch_size $(TRAIN_TEST_BATCH_SIZE) \
		-start_session $(TRAIN_START_SESSION) \
		-gpu $(GPU) \
		-seed $(SEED) \
		$(TRAIN_EXTRA) \
		$(WAND_ARGS)

train_fact_cifar: ## README準拠のFACT+CIFAR100レシピ
	cd $(PROJECT_ROOT) && \
	$(PYTHON) train.py \
		-project fact \
		-dataset cifar100 \
		-base_mode ft_cos \
		-new_mode avg_cos \
		-dataroot $(DATA_DIR) \
		$(FACT_CIFAR_EXTRA) \
		$(TRAIN_EXTRA) \
		$(WAND_ARGS)

train_fact_cub: ## README準拠のFACT+CUB200レシピ（CUB_DATAROOTを指定）
	cd $(PROJECT_ROOT) && \
	$(PYTHON) train.py \
		-project fact \
		-dataset cub200 \
		-base_mode ft_cos \
		-new_mode avg_cos \
		-dataroot $(CUB_DATAROOT) \
		$(FACT_CUB_EXTRA) \
		$(TRAIN_EXTRA) \
		$(WAND_ARGS)

train_fact_mini: ## README準拠のFACT+miniImageNetレシピ（MINI_IMAGENET_ROOTを指定）
	cd $(PROJECT_ROOT) && \
	$(PYTHON) train.py \
		-project fact \
		-dataset mini_imagenet \
		-base_mode ft_cos \
		-new_mode avg_cos \
		-dataroot $(MINI_IMAGENET_ROOT) \
		$(FACT_MINI_EXTRA) \
		$(TRAIN_EXTRA) \
		$(WAND_ARGS)

train_debug: ## 64バッチ・10epochで素早く動作確認
	cd $(PROJECT_ROOT) && \
	$(PYTHON) train.py \
		-project $(TRAIN_PROJECT) \
		-dataset $(TRAIN_DATASET) \
		-dataroot $(DATA_DIR) \
		-epochs_base 10 \
		-epochs_new 5 \
		-batch_size_base 64 \
		-test_batch_size 64 \
		-gpu $(GPU) \
		-debug \
		$(TRAIN_EXTRA) \
		$(WAND_ARGS)

split_cicids: ## CICIDS2017_improvedをtrain/testに分割
	cd $(PROJECT_ROOT) && \
	$(PYTHON) split_cicids2017.py \
		--data_dir $(CICIDS_DIR) \
		--output_dir $(CICIDS_DIR) \
		--test_size $(SPLIT_TEST_SIZE) \
		--random_state $(SEED) \
		$(WILDCARD)

create_cicids_sessions: ## CICIDS2017_improvedのセッションTXTを生成
	cd $(PROJECT_ROOT) && \
	$(PYTHON) create_session_files.py \
		--train_csv $(CICIDS_DIR)/train.csv \
		--output_dir $(CICIDS_INDEX_DIR) \
		--base_class $(SESSION_BASE_CLASS) \
		--num_classes $(SESSION_NUM_CLASSES) \
		--way $(SESSION_WAY) \
		--shot $(SESSION_SHOT) \
		--random_state $(SEED)

clean_pycache: ## __pycache__と*.pycを削除
	cd $(PROJECT_ROOT) && \
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + && \
	find . -name '*.pyc' -delete

clean_checkpoints: ## checkpointディレクトリを削除（要注意）
	@if [ -d "$(CHECKPOINT_DIR)" ]; then \
		read -r -p "Remove $(CHECKPOINT_DIR)? [y/N] " ans && \
		if [[ $$ans =~ ^[Yy]$$ ]]; then rm -rf "$(CHECKPOINT_DIR)"; else echo "skipped"; fi; \
	else \
		echo "checkpoint directory not found"; \
	fi

show_paths: ## 主要パスと変数を表示
	@echo "PROJECT_ROOT      = $(PROJECT_ROOT)"
	@echo "DATA_DIR          = $(DATA_DIR)"
	@echo "CHECKPOINT_DIR    = $(CHECKPOINT_DIR)"
	@echo "CICIDS_DIR        = $(CICIDS_DIR)"
	@echo "CICIDS_INDEX_DIR  = $(CICIDS_INDEX_DIR)"
	@echo "CUB_DATAROOT      = $(CUB_DATAROOT)"
	@echo "MINI_IMAGENET_ROOT= $(MINI_IMAGENET_ROOT)"
	@echo "GPU               = $(GPU)"
	@echo "SEED              = $(SEED)"

