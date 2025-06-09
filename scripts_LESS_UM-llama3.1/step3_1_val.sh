# fixed arguments
DIMS="8192" # We use 8192 as our default projection dimension
torch_dtype="bfloat16"   # float32 is too slow: < 1.5 it/s

max_length="2048"   # 4096 by default for UI; 2048 if GPU memory is limited

# looping args
CKPT_list=(
    "97"
    "194"
    "291"
    "388"
)

task_list=(
    "medqa_nonrag"
    "medqa_rag"
)

REFERENCE_MODEL=Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3
LOG_DIR="./scripts_LESS_UM-llama3.1-logs"

if [[ ! -d $LOG_DIR ]]; then
    mkdir -p $LOG_DIR
fi

# nested loops
for CKPT in "${CKPT_list[@]}"
do
    for TASK in "${task_list[@]}"
    do
        MODEL_PATH=../out/LESS_UM-llama3.1/${REFERENCE_MODEL}/checkpoint-${CKPT}
        OUTPUT_PATH=../grads/LESS_UM-llama3.1/step3_1_${torch_dtype}_maxlen${max_length}-${REFERENCE_MODEL}/${TASK}-ckpt${CKPT}-sgd # for validation data, we always use sgd

        if [ "$CKPT" == "${CKPT_list[0]}" ]; then
            # for the first ckpt, 
            # 1. print all the validation examples for inspection
            # (2. overwrite datasets cache to ensure changes in transforms are all applied) 
                # -> deprecated, since it won't make later ckpts reuse the already overwritten cache
            SCRIPT_PATH="./less/scripts/get_info/grad/get_eval_lora_grads_print_ex_overwrite_cache.sh"
        else
            SCRIPT_PATH="./less/scripts/get_info/grad/get_eval_lora_grads_not_print_or_overwrite.sh"
        fi

        CUDA_VISIBLE_DEVICES=0 $SCRIPT_PATH "$TASK" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$torch_dtype" ""$max_length"" \
        2>&1 | tee -a "${LOG_DIR}/step3_1_${torch_dtype}_maxlen${max_length}.log" 
        # >> "${LOG_DIR}/step3_1_${torch_dtype}_maxlen${max_length}.log" 2>&1
    done
done