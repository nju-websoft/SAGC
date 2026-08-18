#!/bin/bash
set -e

# ========================================================
# Stage 3: SAGC Selection (Golden Path Configuration)
# ========================================================
# The configuration below reproduces the main reported SAGC result.
# Method: SAGC (K-Means Clustering)
# Fusion: Max-Pooling of Label Embeddings
METHOD="kmeans"
CONCAT_TYPE="max"      # Options: 'sum', 'max', 'tfidf'
K_CLUSTERS=256         # Number of clusters for diversity
ALPHA=0.5              # Fusion weight: 0.5 * Data + 0.5 * Label

# --- 2. Data & Paths ---
# Inputs (from Stage 1 & 2)
INPUT_EMB_DIR="results/embeddings/data"          # Main Data Embeddings (Micro Context)
LABEL_EMB_DIR="results/embeddings/labels"        # Label Embeddings (Macro Intent)
ORIGINAL_DATA="data/processed/cleaned_labeled_dataset.jsonl"
LABEL_VOCAB="results/labels/label_thre5.txt"

# Outputs
OUTPUT_DIR="data/training_sets"

# Fixed Parameters
DATASET_SIZE=446062    # Total size of the source dataset
N_SAMPLES=20000        # Target SFT data size
SEED=42

echo "========================================================"
echo "   Stage 3: Diversity-Driven Data Selection"
echo "   Method:  ${METHOD} (k=${K_CLUSTERS})"
echo "   Fusion:  ${CONCAT_TYPE}-Pooling (alpha=${ALPHA})"
echo "   Samples: ${N_SAMPLES} / ${DATASET_SIZE}"
echo "========================================================"

mkdir -p "${OUTPUT_DIR}"

python src/selection/main_selection.py \
    --input_emb_dir "${INPUT_EMB_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --original_data_path "${ORIGINAL_DATA}" \
    --label_vocab_path "${LABEL_VOCAB}" \
    --label_emb_dir "${LABEL_EMB_DIR}" \
    --dataset_size ${DATASET_SIZE} \
    --n_samples ${N_SAMPLES} \
    --method "${METHOD}" \
    --k_clusters ${K_CLUSTERS} \
    --concat_type "${CONCAT_TYPE}" \
    --alpha ${ALPHA} \
    --seed ${SEED}

echo "✅ Stage 3 Completed!"
echo "   Selected Indices: ${OUTPUT_DIR}/selected_${METHOD}_${CONCAT_TYPE}_n${N_SAMPLES}.npy"
echo "   SFT Training Set: ${OUTPUT_DIR}/selected_${METHOD}_${CONCAT_TYPE}_n${N_SAMPLES}.jsonl"
