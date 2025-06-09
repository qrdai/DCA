#!/bin/bash

ID=$RANDOM
PORT=$((29000 + ID % 1000))  # This generates a port number between 29000 and 29999
export header="OMP_NUM_THREADS=20 torchrun --nproc_per_node 2 --nnodes 1 \
--rdzv-endpoint=localhost:$PORT \
--rdzv-id=$ID --rdzv_backend c10d \
-m less.train.train"

# delete `--save_strategy no`, which is probably a typo

# fixed training hyperparams according to open-instruct
# effective_batch_size fixed to 128
# `--learning_rate` fixed to 2e-5
# `--weight_decay` fixed to 0.0
# training data type of lora layers fixed to `bf16` (and frozen base model fixed to fp32), while eval fixed to `fp32`
# `--optim` fixed to `adamw_torch`
# `--seed` fixed to 0; this is the global seed for random, numpy, torch, etc., not sample_data_seed

# initial choice for `--max_seq_length`: 4096 (can scale up to 8192; 2048 is too short for math questions)
# initial choice for `--lr_scheduler_type`: `linear` (can switch to `cosine` for linear warmup & cosine decay)
# initial choice for `--num_train_epochs`: 4 (can lower down to 2 or 1)
# initial choice for `--warmup_ratio`: 0.03 (can scale up to 0.1, with fewer epochs)

# initial choice for `--lora_target_modules`: only attention projection layers `q_proj k_proj v_proj o_proj` (can scale up to all attention and feedforward linear layers)
# initial choice for `--lora_r`: 128 (64); `--lora_alpha`: 512 (16); `lora_dropout`: 0.1 (should adjust according to actual training curve on UI; values in the bracket are used by open-instruct for QLoRA training)

# possibly override `--percentage` in `warmup_lora_train.sh`
# possibly override `--evaluation_strategy`, `--eval_steps`, --save_strategy`, `--save_steps` in `lora_train.sh`, if we need to eval during training / save (step-wise) ckpts for eval after training

# 08/13/2024: add `--torch_dtype float32` to explicitly set the precision of frozen base model as float32 (originally set as None, which will also default to float32; the only difference is that now model.config.torch_dtype will also be float32 instead of None)

# 08/27/2024: in step4: lora_train.sh, `--num_train_epochs` is overridden to be 2 to avoid overfitting

# 01/22/2025: for S2L in step 1:
    # 1. `lora` is overriden to False to enable full training on smaller models
    # 2. `save_strategy` is overidden to be "steps", with `save_steps` set as well
export base_training_args="--do_train True \
--max_seq_length 4096 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--evaluation_strategy no \
--logging_steps 1 \
--num_train_epochs 4 \
--torch_dtype float32 \
--bf16 True \
--tf32 False \
--fp16 False \
--overwrite_output_dir True \
--report_to wandb \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--lora True \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.1 \
--lora_target_modules q_proj k_proj v_proj o_proj \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 64"   # --nproc_per_node 2 -> effective bsz = 64*2 = 128