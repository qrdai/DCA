# fixed arguments
idx=0
GRADIENT_TYPE="adam"
DIMS="8192" # 8192 by default; 4096 if storage if limited

torch_dtype="bfloat16"   # float32 is too slow: < 1.5 it/s
# torch_dtype="float32"   # according to Jiaqi, float32 is sometimes crucial for gradient-based TDA methods?

max_length="2048"   # 4096 by default for UI; 2048 if GPU memory is limited

# looping args
CKPT_list=(
    # "97"
    "194"
    # "291"
    # "388"
)

dataset_list=(
    "UltraMedical-train-Exam-50k"
)

REFERENCE_MODEL=Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3
LOG_DIR="./scripts_LESS_UM-llama3.1-logs"

if [[ ! -d $LOG_DIR ]]; then
    mkdir -p $LOG_DIR
fi

# nested loops
for CKPT in "${CKPT_list[@]}"
do
    for TRAINING_DATA_NAME in "${dataset_list[@]}"
    do
        TRAINING_DATA_FILE=../data/UltraMedical/${TRAINING_DATA_NAME}.jsonl
        MODEL_PATH=../out/LESS_UM-llama3.1/${REFERENCE_MODEL}/checkpoint-${CKPT}
        OUTPUT_PATH=../grads/LESS_UM-llama3.1/step2_${GRADIENT_TYPE}_${torch_dtype}_maxlen${max_length}-${REFERENCE_MODEL}/${TRAINING_DATA_NAME}-ckpt${CKPT}-${GRADIENT_TYPE}

        CUDA_VISIBLE_DEVICES="${idx}" OMP_NUM_THREADS=20 ./less/scripts/get_info/grad/get_train_lora_grads.sh \
        "$TRAINING_DATA_FILE" "$MODEL_PATH" "$OUTPUT_PATH" "$DIMS" "$GRADIENT_TYPE" "$torch_dtype" "$max_length" \
        >> "${LOG_DIR}/step2_${GRADIENT_TYPE}_${torch_dtype}_maxlen${max_length}-${idx}.log" 2>&1
        # 2>&1 | tee -a "${LOG_DIR}/step2_${GRADIENT_TYPE}_${torch_dtype}_maxlen${max_length}-${idx}.log"
        # >> "${LOG_DIR}/step2_${GRADIENT_TYPE}_${torch_dtype}_maxlen${max_length}-${idx}.log" 2>&1


        # # Directory to check and clean
        # clean_dir="${OUTPUT_PATH}/dim${DIMS}"

        # # Check if 'all_orig.pt' exists in the directory
        # if [ -f "${clean_dir}/all_orig.pt" ]; then
        #     echo "'all_orig.pt' exists. Deleting other files in ${clean_dir}"

        #     # Find and delete all files in the directory except 'all_orig.pt'
        #     find "${clean_dir}/" -type f ! -name 'all_orig.pt' -exec rm {} +
        # else
        #     echo "'all_orig.pt' does not exist. No files will be deleted."
        # fi
    done
done