idx=1

test_file_list=(
    "../data/MedQA-USMLE-4-options/gpt-4-1106-preview-size5-seed42.json"
    "../data/MedQA-USMLE-4-options/IDENTIFY-4-shuffle=01-direct_gpt-3.5-turbo-1106-size5-seed42.json"
)
model_path_list=(
    "../out/LESS_UM-llama3.1/Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3/checkpoint-388"
)
num_ablations_list=(
    64
    256
)

torch_dtype="bfloat16"
output_dir="./qualitative_results"
LOG_DIR="./logs"


for num_ablations in "${num_ablations_list[@]}"
do
    for test_file in "${test_file_list[@]}"
    do
        # basename with a suffix parameter strips both the path and the .json
        dataset_name="$(basename "$test_file" .json)"
        for model_path in "${model_path_list[@]}"
        do
            CUDA_VISIBLE_DEVICES="${idx}" OMP_NUM_THREADS=20 python -m medqa.medqa_lora \
                --test_file $test_file \
                --model_path $model_path \
                --output_dir $output_dir \
                --torch_dtype $torch_dtype \
                --num_ablations $num_ablations \
                --rag \
                2>&1 | tee "${LOG_DIR}/medqa-lora-${dataset_name}-ablations${num_ablations}-rag.log"
        done
    done
done