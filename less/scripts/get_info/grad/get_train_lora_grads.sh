#!/bin/bash

train_file=$1 #
model=$2 # path to model
output_path=$3 # path to output
dims=$4 # dimension of projection, can be a list
gradient_type=$5
torch_dtype=$6
max_length=$7


if [[ ! -d $output_path ]]; then
    mkdir -p $output_path
fi


python3 -m less.data_selection.get_info \
--train_file $train_file \
--info_type grads \
--model_path $model \
--output_path $output_path \
--gradient_projection_dimension $dims \
--gradient_type $gradient_type  \
--torch_dtype $torch_dtype \
--max_length $max_length
# --max_samples 200 # set `--max_samples` as None -> use all training samples
