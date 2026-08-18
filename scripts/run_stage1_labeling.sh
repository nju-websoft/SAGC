#!/bin/bash
set -e

# ================= Configuration =================
# Modify this path to point to your local model
MODEL_PATH="Qwen/Qwen3-8B" 

# Input: The cleaned dataset from Step 0
INPUT_DATA="data/processed/dataset_cleaned.jsonl"

# Intermediate: Data with raw LLM generated labels
RAW_LABELED_DATA="data/processed/labeled_dataset_raw.jsonl"

# Output: Final cleaned labeled data
CLEANED_DATA="data/processed/cleaned_labeled_dataset.jsonl"

# Metrics: Vocab file and statistics
LABEL_VOCAB="results/labels/label_thre5.txt"
LABEL_STATS="results/labels/labels_analyzed.csv"

# Parameters
MIN_COUNT=5  # Labels appearing fewer than 5 times are discarded
BATCH_SIZE=100

echo "========================================================"
echo "   Stage 1: Label Generation & Processing Pipeline"
echo "========================================================"

# 1. Generation
echo ">>> [1/3] Generating Labels using ${MODEL_PATH}..."
python src/labeling/gen_labels.py \
    --model_path "${MODEL_PATH}" \
    --input_file "${INPUT_DATA}" \
    --output_file "${RAW_LABELED_DATA}" \
    --batch_size ${BATCH_SIZE} 

# 2. Cleaning
echo ">>> [2/3] Cleaning & Normalizing Labels..."
python src/labeling/clean_labels.py \
    --input_file "${RAW_LABELED_DATA}" \
    --output_file "${CLEANED_DATA}"

# 3. Analysis
echo ">>> [3/3] Analyzing Statistics & Exporting Vocab..."
python src/labeling/analyze_labels.py \
    --input_file "${CLEANED_DATA}" \
    --output_csv "${LABEL_STATS}" \
    --output_vocab "${LABEL_VOCAB}" \
    --min_count ${MIN_COUNT} \
    --top_n 20

echo "✅ Stage 1 Completed!"
echo "   Output Data: ${CLEANED_DATA}"
echo "   Label Vocab: ${LABEL_VOCAB}"
