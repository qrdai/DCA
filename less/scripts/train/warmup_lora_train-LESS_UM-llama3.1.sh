#!/bin/bash

source less/scripts/train/base_training_args.sh

data_dir=$1
model_path=$2
percentage=$3
max_seq_length=$4
num_train_epochs=$5
torch_dtype=$6
data_seed=$7
job_name=$8
wandb_project=$9
per_device_train_batch_size=${10}
gradient_accumulation_steps=${11}

output_dir=../out/LESS_UM-llama3.1/${job_name}
if [[ ! -d $output_dir ]]; then
    mkdir -p $output_dir
fi

train_files=(
    "$data_dir/UltraMedical-train-Exam-50k.jsonl"
)


# use fsdp for all types of models, in order to obtain `optimizer.bin` with str-based keys
if [[ $model_path == *"llama"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config llama_finetune"
elif [[ $model_path == *"mistral"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config mistral_7b_finetune"
elif [[ $model_path == *"Qwen"* ]]; then
    base_training_args="$base_training_args --fsdp 'full_shard auto_wrap' --fsdp_config qwen2_finetune"
fi


# (10/30) add new training args: 
    # `run_name`: automatic wandb rename; 
    # `per_device_train_batch_size` & `gradient_accumulation_steps`: scale up bsz when training on more GPUs (still use 2*64 when training on 2*80G or 8*40G)
# (03/30) override `--bf16 True` in base_training_args.sh
    # also set ${torch_dtype} as bfloat16 in step1.sh
    # to see if this deactivates mixed-precision training and thus prevents automatic up-casting to float32
training_args="$base_training_args \
--model_name_or_path $model_path \
--output_dir $output_dir \
--percentage $percentage \
--max_seq_length $max_seq_length \
--num_train_epochs $num_train_epochs \
--torch_dtype $torch_dtype \
--sample_data_seed $data_seed \
--train_files ${train_files[@]} \
--wandb_project $wandb_project \
--run_name $job_name \
--per_device_train_batch_size $per_device_train_batch_size \
--gradient_accumulation_steps $gradient_accumulation_steps \
--bf16 False \
--analysis_mode False 2>&1 | tee $output_dir/train_warmup.log"

eval "$header" "$training_args"