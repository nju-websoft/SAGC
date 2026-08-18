import argparse
import numpy as np
import os
import torch
import random

# Modules
from utils_io import load_and_reorder_embeddings, save_indices, save_subset_to_jsonl
from utils_labels import get_one_hot_embeddings, get_summed_embeddings, get_max_pooled_embeddings, get_tfidf_embeddings
from utils_fusion import combine_embeddings
from algo_clustering import run_kmeans
from algo_stratified import stratified_sample
from algo_sampling import fps_max_sum_gpu, fps_max_min_gpu

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Data Selection & Feature Fusion Pipeline")
    
    # --- Input/Output ---
    parser.add_argument("--input_emb_dir", type=str, required=True, help="Main semantic embeddings (.npz chunks)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--original_data_path", type=str, default=None, help="Raw JSONL path (required for label processing & export)")
    
    # --- Data Config ---
    parser.add_argument("--dataset_size", type=int, default=446062)
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    # --- Method ---
    parser.add_argument("--method", type=str, choices=['kmeans', 'max_sum', 'max_min'], required=True)
    parser.add_argument("--k_clusters", type=int, default=256)

    # --- Feature Fusion (Concatenation) ---
    parser.add_argument("--concat_type", type=str, default=None, 
                        choices=['onehot', 'sum', 'max', 'tfidf'], 
                        help="Type of label embedding to concatenate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for label embeddings (0.0 to 1.0)")
    parser.add_argument("--label_vocab_path", type=str, help="Path to label_vocab.txt")
    parser.add_argument("--label_emb_dir", type=str, help="Dir containing label's own semantic embeddings (for sum/max/tfidf)")

    args = parser.parse_args()
    
    # Init Random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 1. Load Main Semantic Embeddings
    print(f"Loading main embeddings from {args.input_emb_dir}...")
    indices, main_embeddings = load_and_reorder_embeddings(args.input_emb_dir, args.dataset_size)
    
    # 2. Feature Fusion Logic
    final_embeddings = main_embeddings
    
    if args.concat_type:
        if not args.original_data_path or not args.label_vocab_path:
            raise ValueError("To use concat_type, you must provide --original_data_path and --label_vocab_path")
            
        print(f"Preparing secondary embeddings: {args.concat_type}")
        aux_embeddings = None
        
        if args.concat_type == 'onehot':
            aux_embeddings = get_one_hot_embeddings(args.original_data_path, args.label_vocab_path)
        elif args.concat_type == 'sum':
            if not args.label_emb_dir: raise ValueError("--label_emb_dir required for sum pooling")
            aux_embeddings = get_summed_embeddings(args.original_data_path, args.label_vocab_path, args.label_emb_dir)
        elif args.concat_type == 'max':
            if not args.label_emb_dir: raise ValueError("--label_emb_dir required for max pooling")
            aux_embeddings = get_max_pooled_embeddings(args.original_data_path, args.label_vocab_path, args.label_emb_dir)
        elif args.concat_type == 'tfidf':
            if not args.label_emb_dir: raise ValueError("--label_emb_dir required for tfidf")
            aux_embeddings = get_tfidf_embeddings(args.original_data_path, args.label_vocab_path, args.label_emb_dir)
            
        # Combine!
        final_embeddings = combine_embeddings(main_embeddings, aux_embeddings, args.alpha)
    
    # 3. Selection Algorithm
    selected_indices_local = []
    
    if args.method == 'kmeans':
        # A. Clustering (CPU)
        labels = run_kmeans(final_embeddings, args.k_clusters, seed=args.seed)
        # B. Stratified Sampling
        selected_indices_local = stratified_sample(labels, args.n_samples)
        
    elif args.method == 'max_sum':
        selected_indices_local = fps_max_sum_gpu(final_embeddings, args.n_samples, start_idx=args.seed)
        
    elif args.method == 'max_min':
        selected_indices_local = fps_max_min_gpu(final_embeddings, args.n_samples, start_idx=args.seed)

    # 4. Save Results
    final_selected_ids = indices[selected_indices_local]
    
    suffix = f"_{args.concat_type}" if args.concat_type else ""
    out_name = f"selected_{args.method}{suffix}_n{args.n_samples}"
    
    # Save .npy
    out_npy = os.path.join(args.output_dir, f"{out_name}.npy")
    save_indices(final_selected_ids, out_npy)
    
    # Save .jsonl
    if args.original_data_path:
        out_jsonl = os.path.join(args.output_dir, f"{out_name}.jsonl")
        save_subset_to_jsonl(args.original_data_path, final_selected_ids, out_jsonl)

if __name__ == "__main__":
    main()
