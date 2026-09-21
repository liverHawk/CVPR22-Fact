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
