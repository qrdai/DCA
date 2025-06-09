#!/bin/bash

# for validation data, we should always get gradients with sgd
task=$1
model_path=$2
output_path=$3
dims=$4
torch_dtype=$5
max_length=$6


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


python3 -m less.data_selection.get_info \
--task $task \
--info_type grads \
--model_path $model_path \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type sgd \
--torch_dtype $torch_dtype \
--max_length $max_length
# --print_ex
# --overwrite_cache
