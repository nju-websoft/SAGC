#!/bin/bash
set -e

# ================= Configuration =================
EXP_NAME="Qwen2.5-7B-SAGeC-KMeans"
MODEL_PATH="output/models/${EXP_NAME}/final_hf"

# Output
OUTPUT_DIR="output/evaluation/${EXP_NAME}"
GENERATION_FILE="${OUTPUT_DIR}/generations.json"
SCORES_FILE="${OUTPUT_DIR}/generations_scores.json"

# --- Isolated Environment Configuration ---
# Point to the Python executable we created in README step 1
PROJECT_ROOT=$(pwd)
LCB_PYTHON="${PROJECT_ROOT}/envs/.venv_eval/bin/python"

# Evaluation Params
RELEASE_VERSION="release_v6"
START_DATE="2024-09-20"
TEMP=0.2
N_SAMPLES=10

echo "========================================================"
echo "   Stage 5: Evaluation (Internal LCB Runner)"
echo "   Model:      ${MODEL_PATH}"
echo "   Python:     ${LCB_PYTHON}"
echo "========================================================"

# 0. Check Prerequisites
if [ ! -f "${LCB_PYTHON}" ]; then
    echo "Error: Evaluation environment not found at ${LCB_PYTHON}"
    echo "Please run the setup commands in README (Stage 5) to create 'envs/.venv_eval'."
    exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
    echo "Error: Model not found at ${MODEL_PATH}. Run Stage 4 first."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Export PYTHONPATH so the isolated python can see your local src/eval/lcb_runner
export PYTHONPATH="${PROJECT_ROOT}/src/eval:${PYTHONPATH}"

# 1. Inference
if [ -f "${GENERATION_FILE}" ]; then
    echo ">>> [1/2] Generation file exists. Skipping inference."
else
    echo ">>> [1/2] Running Inference..."
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    
    # Run lcb_runner using the isolated python + local source code
    "${LCB_PYTHON}" -m lcb_runner.runner.main \
        --model "${EXP_NAME}" \
        --local_model_path "${MODEL_PATH}" \
        --scenario codegeneration \
        --evaluate \
        --release_version ${RELEASE_VERSION} \
        --start_date ${START_DATE} \
        --n ${N_SAMPLES} \
        --temperature ${TEMP} \
        --tensor_parallel_size ${NUM_GPUS} \
        --stop $'```\n' \
        --max_tokens 8192 \
        --timeout 6 \
        --output_path "${GENERATION_FILE}"
fi

# 2. Scoring
echo ">>> [2/2] Computing MCS Scores..."
ABS_GEN_FILE=$(readlink -f "${GENERATION_FILE}")

"${LCB_PYTHON}" src/eval/evaluate_generations_mcs.py \
    --input_file "${ABS_GEN_FILE}" \
    --release_version ${RELEASE_VERSION} \
    --start_date ${START_DATE} \
    --num_process_evaluate 16

echo "✅ Stage 5 Completed!"
echo "   Scores: ${SCORES_FILE}"