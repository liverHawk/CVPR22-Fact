GPU=""

minimum:
	uv run python train.py \
		-project fact \
		-dataset cifar100 \
		-base_mode ft_cos \
		-new_mode avg_cos \
		-epochs_base 1 \
		-epochs_new 1 \
		-batch_size_base 64 \
		-gpu $(GPU) \
		-num_workers 2 \
		--use_wandb

# Pipeline: make session file -> train -> test
session:
	uv run python scripts/make_session.py --config params.yaml

train:
	uv run python train.py --config params.yaml --metrics-file dvc_metrics.json

test:
	uv run python test.py --config params.yaml --test-metrics-file dvc_test_metrics.json

pipeline:
	$(MAKE) session && $(MAKE) train && $(MAKE) test

# Optuna: run 2 worker processes in parallel, one per physical GPU, both
# writing into the same sqlite-backed study. This bypasses `dvc repro tune`
# (its --storage "" is an in-memory, single-process study, see dvc.yaml) --
# GPU parallelism only works as separate OS processes, each with its own
# CUDA_VISIBLE_DEVICES, never via tune.py's --jobs (that's thread-based and
# shares the process-global CUDA_VISIBLE_DEVICES, which breaks GPU pinning).
# Requires params.yaml's `gpu` to be a quoted string (e.g. "0"): each worker
# then always addresses its GPU as logical device 0, and
# CUDA_VISIBLE_DEVICES picks which physical GPU that actually is.
# Override physical GPU IDs with e.g. `make GPU0=2 GPU1=3 tune-2gpu`.
TUNE_TRIALS=10
TUNE_STUDY=fact
TUNE_STORAGE=sqlite:///optuna.db
GPU0=0
GPU1=1

# Create the sqlite file + study schema up front, sequentially. If the two
# workers below both hit a not-yet-existing storage file at once, they race
# on Optuna/Alembic's one-time schema init and one crashes with
# "UNIQUE constraint failed: alembic_version.version_num" -- this avoids
# that by making sure the schema already exists before either worker starts.
tune-init:
	uv run python -c "import optuna; optuna.create_study(study_name='$(TUNE_STUDY)', storage='$(TUNE_STORAGE)', direction='maximize', load_if_exists=True)"

tune-gpu0:
	CUDA_VISIBLE_DEVICES=$(GPU0) uv run python tune.py \
		--base-config params.yaml --trials $(TUNE_TRIALS) \
		--storage "$(TUNE_STORAGE)" --study-name $(TUNE_STUDY) \
		--params-out best_params.yaml --metrics-out tune_metrics.json

tune-gpu1:
	CUDA_VISIBLE_DEVICES=$(GPU1) uv run python tune.py \
		--base-config params.yaml --trials $(TUNE_TRIALS) \
		--storage "$(TUNE_STORAGE)" --study-name $(TUNE_STUDY) \
		--params-out best_params.yaml --metrics-out tune_metrics.json

TUNE_LOG0=tune-gpu0.log
TUNE_LOG1=tune-gpu1.log

# Both workers print to the same terminal at once otherwise, interleaving
# tqdm bars / prints into unreadable output -- redirect each to its own log
# and tail them yourself with e.g. `tail -f tune-gpu0.log`.
tune-2gpu: tune-init
	$(MAKE) tune-gpu0 > $(TUNE_LOG0) 2>&1 & \
	$(MAKE) tune-gpu1 > $(TUNE_LOG1) 2>&1 & \
	wait
	@echo "done -- logs: $(TUNE_LOG0), $(TUNE_LOG1)"

# Drop the sqlite study (e.g. after a corrupted/partial optuna.db from a
# tune-init race, or to start a fresh search under the same study name).
tune-clear:
	rm -f optuna.db optuna.db-journal optuna.db-wal optuna.db-shm
