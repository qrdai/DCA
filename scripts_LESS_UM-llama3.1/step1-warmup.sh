DATA_DIR=../data/UltraMedical
MODEL_PATH=meta-llama/Llama-3.1-8B

PERCENTAGE=0.25
MAX_LENGTH=2048
EPOCHS=4
TORCH_DTYPE=bfloat16 # bfloat16 / float32
DATA_SEED=3

# when using FSDP by transformers=4.42.4, can scale up `per_device_train_batch_size` to speed up
# -> `per_device_train_batch_size=2` still leads to OOM :(
per_device_train_batch_size=2
gradient_accumulation_steps=32  # when setting `--nproc_per_node 4/8` in base_training_args.sh, scale down `gradient_accumulation_steps` accordingly: `4 GPUs * 32 steps` or `8 GPUs * 16 steps`

JOB_NAME="Llama_3.1_8B-p${PERCENTAGE}-lora_attn_only-maxlen${MAX_LENGTH}-epochs${EPOCHS}-basetype_${TORCH_DTYPE}-perdev_bsz${per_device_train_batch_size}-gradstep${gradient_accumulation_steps}-dataseed${DATA_SEED}"
WANDB_PROJECT=LESS_UM-llama3.1

./less/scripts/train/warmup_lora_train-LESS_UM-llama3.1.sh \
"$DATA_DIR" "$MODEL_PATH" "$PERCENTAGE" "$MAX_LENGTH" "$EPOCHS" "$TORCH_DTYPE" "$DATA_SEED" \
"$JOB_NAME" "$WANDB_PROJECT" "$per_device_train_batch_size" "$gradient_accumulation_steps"
