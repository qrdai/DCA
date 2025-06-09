train_file_names=(
    "UltraMedical-train-Exam-50k"
)

# NOTE: the order of target_task_names matters: it decides the order of columns in the final instance AM!
target_task_names=(
    "medqa_rag"
)
# final_dir_name="all_7"
dtype="float64" # float64 / (default in original LESS) float32

# target_task_names=( "human_eval" )
# target_task_names=( "mbpp" )
# target_task_names=( "bbh" )
# target_task_names=( "mmlu" )
# target_task_names=( "gsm_plus" )
# target_task_names=( "MATH" )
# target_task_names=( "if_eval" )
final_dir_name=${target_task_names[0]}


TRAIN_FILE_NAMES="${train_file_names[*]}"  # Using * within quotes to create a single string
TARGET_TASK_NAMES="${target_task_names[*]}"

DIM=8192
step2_gradient_type=adam

CKPTS="97 194 291 388"
CHECKPOINT_WEIGHTS="1.6804e-05 1.2887e-05 7.7320e-06 2.5773e-06" # midpoints of continuous lines

REFERENCE_MODEL=Llama_3.1_8B-p0.25-lora_attn_only-maxlen2048-epochs4-basetype_bfloat16-perdev_bsz2-gradstep32-dataseed3
project_dir=".."
LOG_DIR="./scripts_LESS_UM-llama3.1-logs"

if [[ ! -d $LOG_DIR ]]; then
    mkdir -p $LOG_DIR
fi

GRADIENT_PATH="${project_dir}/grads/LESS_UM-llama3.1/step2_${step2_gradient_type}_bfloat16_maxlen2048-${REFERENCE_MODEL}/{}-ckpt{}-${step2_gradient_type}/dim${DIM}"   # also use sgd grads for training points
VALIDATION_GRADIENT_PATH="${project_dir}/grads/LESS_UM-llama3.1/step3_1_bfloat16_maxlen2048-${REFERENCE_MODEL}/{}-ckpt{}-sgd/dim${DIM}"

AM_OUTPUT_PATH="${project_dir}/attribution_matrix/LESS_UM-llama3.1/step2_${step2_gradient_type}_valsize10_${REFERENCE_MODEL}"

./less/scripts/data_selection/matching_instance_AM.sh \
"$GRADIENT_PATH" "$TRAIN_FILE_NAMES" "$CKPTS" "$CHECKPOINT_WEIGHTS" "$VALIDATION_GRADIENT_PATH" "$TARGET_TASK_NAMES" "$AM_OUTPUT_PATH" "$dtype" \
2>&1 | tee -a "${LOG_DIR}/step3_2_${step2_gradient_type}_valsize10_instance_AM_${final_dir_name}_${dtype}.log"
