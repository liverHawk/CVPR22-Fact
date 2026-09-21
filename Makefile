minimum:
	uv run python train.py \
		-project fact \
		-dataset cifar100 \
		-base_mode ft_cos \
		-new_mode avg_cos \
		-epochs_base 1 \
		-epochs_new 1 \
		-batch_size_base 64 \
		-gpu "" \
		-num_workers 2


