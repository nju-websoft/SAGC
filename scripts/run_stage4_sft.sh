#!/bin/bash
set -e

# ================= Configuration =================
# Experiment Name (Will create a folder inside output/models/)
EXP_NAME="Qwen2.5-7B-SAGeC-KMeans"

# Input Data (From Stage 3)
DATA_PATH="data/training_sets/selected_kmeans_max_n20000.jsonl"

# Output Paths (Adhering to strict structure)
# Intermediate checkpoints go here: output/models/EXP_NAME
WORK_DIR="output/models/${EXP_NAME}"

# Final converted HF model goes here: output/models/EXP_NAME/final_hf
FINAL_MODEL_DIR="${WORK_DIR}/final_hf"

# Config
CONFIG_FILE="src/sft/qwen_config.py"
PORT=$(shuf -n 1 -i 29500-65535)

echo "========================================================"
echo "   Stage 4: Supervised Fine-Tuning (SFT)"
echo "   Experiment: ${EXP_NAME}"
echo "   Work Dir:   ${WORK_DIR}"
echo "========================================================"

if [ ! -f "${DATA_PATH}" ]; then
    echo "Error: Training data not found at ${DATA_PATH}"
    exit 1
fi

export XTUNER_DATA_PATH="${DATA_PATH}"
export XTUNER_WORK_DIR="${WORK_DIR}"

# 1. Start Training
echo ">>> [1/2] Starting XTuner Training..."
xtuner train ${CONFIG_FILE} \
    --deepspeed zero2 \
    --port ${PORT}

# 2. Convert to HF
echo ">>> [2/2] Converting Checkpoint..."
# Find latest pth
PTH_FILE=$(find ${WORK_DIR} -name "epoch_*.pth" | sort | tail -n 1)

if [ -z "${PTH_FILE}" ]; then
    echo "Error: No checkpoint found in ${WORK_DIR}"
    exit 1
fi

echo "Converting ${PTH_FILE} to ${FINAL_MODEL_DIR}..."
xtuner convert pth_to_hf ${CONFIG_FILE} ${PTH_FILE} ${FINAL_MODEL_DIR}

echo "✅ Stage 4 Completed!"
echo "   Final Model: ${FINAL_MODEL_DIR}"