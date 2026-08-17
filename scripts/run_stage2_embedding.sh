#!/bin/bash
set -e

# ================= Configuration =================
# [UPDATED] Using the specific Qwen3 embedding model
MODEL_PATH="Qwen/Qwen3-Embedding-8B"

# Inputs
LABEL_VOCAB="results/labels/label_thre5.txt"
DATA_FILE="data/processed/cleaned_labeled_dataset.jsonl"

# Outputs
OUTPUT_BASE="results/embeddings"
LABEL_EMB_DIR="${OUTPUT_BASE}/labels"
DATA_EMB_DIR="${OUTPUT_BASE}/data"

# Compute Config
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
BATCH_SIZE=32
MAX_LEN=512

echo "========================================================"
echo "   Stage 2: Dual-View Embedding Generation"
echo "   Model: ${MODEL_PATH}"
echo "   GPUs:  ${NUM_GPUS}"
echo "========================================================"

# 1. Embed Labels (Macro Intent View)
echo ">>> [1/2] Generating Label Embeddings..."
torchrun --nproc_per_node=${NUM_GPUS} src/embedding/embed_labels.py \
    --model_path "${MODEL_PATH}" \
    --input_file "${LABEL_VOCAB}" \
    --output_dir "${LABEL_EMB_DIR}" \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len 128

# 2. Embed Data (Micro Context View)
echo ">>> [2/2] Generating Data Embeddings (Mode: Query)..."
torchrun --nproc_per_node=${NUM_GPUS} src/embedding/embed_data.py \
    --model_path "${MODEL_PATH}" \
    --data_path "${DATA_FILE}" \
    --output_dir "${DATA_EMB_DIR}" \
    --mode "query" \
    --batch_size ${BATCH_SIZE} \
    --max_seq_len ${MAX_LEN}

echo "✅ Stage 2 Completed!"