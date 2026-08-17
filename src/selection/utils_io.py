import os
import glob
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def load_and_reorder_embeddings(base_directory: str, target_size: int = None):
    """
    Load distributed embedding chunks (embedding_rankX_partY.npz) 
    and reorder them in memory to match the original dataset order.
    """
    search_pattern = os.path.join(base_directory, 'embedding_rank*_part*.npz')
    file_paths = sorted(glob.glob(search_pattern))

    if not file_paths:
        raise FileNotFoundError(f"Error: No .npz files found at {search_pattern}")

    print(f"Found {len(file_paths)} embedding files. Loading...")
    
    # 1. Group files by Rank
    rank_embeddings = defaultdict(list)
    max_rank = 0

    for file_path in tqdm(file_paths, desc="Loading chunks"):
        filename = os.path.basename(file_path)
        try:
            # Example filename: embedding_rank0_part0.npz
            rank_part = filename.split('rank')[1].split('_')[0]
            rank = int(rank_part)
            max_rank = max(max_rank, rank)
            
            with np.load(file_path, allow_pickle=True) as data:
                # Handle compatibility for different key names
                if 'embedding' in data:
                    emb = data['embedding']
                elif 'embeddings' in data:
                    emb = data['embeddings']
                else:
                    continue
                rank_embeddings[rank].append(emb)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # 2. Merge parts within each Rank
    merged_rank_embeddings = {}
    max_len_per_rank = 0
    for rank in rank_embeddings:
        merged = np.concatenate(rank_embeddings[rank], axis=0)
        merged_rank_embeddings[rank] = merged
        max_len_per_rank = max(max_len_per_rank, len(merged))

    num_ranks = max_rank + 1
    print(f"Loaded data from {num_ranks} ranks. Max length per rank: {max_len_per_rank}")

    # 3. De-interleave to restore original order
    # Global Index = (Local Index * Num Ranks) + Rank ID
    total_alloc_size = max_len_per_rank * num_ranks
    embedding_dim = merged_rank_embeddings[0].shape[1]
    
    # Use float32 for Faiss compatibility
    all_embeddings = np.zeros((total_alloc_size, embedding_dim), dtype=np.float32)
    
    print("Reordering distributed data...")
    # This logic assumes Stage 2 used standard DistributedSampler with shuffle=False
    for i in tqdm(range(max_len_per_rank)):
        for r in range(num_ranks):
            if r in merged_rank_embeddings and i < len(merged_rank_embeddings[r]):
                global_idx = i * num_ranks + r
                all_embeddings[global_idx] = merged_rank_embeddings[r][i]
    
    # 4. Truncate to original dataset size (remove padding)
    if target_size:
        print(f"Truncating to original dataset size: {target_size}")
        all_embeddings = all_embeddings[:target_size]
        indices = np.arange(target_size)
    else:
        indices = np.arange(len(all_embeddings))

    return indices, all_embeddings

def save_indices(indices, output_path):
    """Save selected indices to a .npy file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, indices)
    print(f"Saved {len(indices)} indices to {output_path}")

def save_subset_to_jsonl(original_jsonl_path, selected_indices, output_jsonl_path):
    """
    Extract and save a subset of data from the original JSONL file based on selected indices.
    """
    print(f"\n--- Saving Subset JSONL ---")
    print(f"Source: {original_jsonl_path}")
    print(f"Target: {output_jsonl_path}")
    print(f"Selected count: {len(selected_indices)}")

    if not os.path.exists(original_jsonl_path):
        print(f"Error: Original data not found at {original_jsonl_path}")
        return

    # Convert to set for O(1) lookup
    selected_set = set(selected_indices)
    
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
    
    saved_count = 0
    with open(original_jsonl_path, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl_path, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(tqdm(f_in, desc="Scanning & Writing")):
            if idx in selected_set:
                f_out.write(line)
                saved_count += 1
                
    print(f"Successfully saved {saved_count} lines to {output_jsonl_path}")