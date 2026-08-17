import torch
import numpy as np
from tqdm import tqdm
import random

def fps_max_sum_gpu(embeddings, n_samples, start_idx=None):
    """
    FPS Variant 1: Max-Sum (Maximize Sum of Distances).
    Logic: Select the point that maximizes the sum of distances to ALL previously selected points.
    """
    print(f"--- Starting FPS (Max-Sum) on GPU (Target: {n_samples}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU (this might be slow).")

    # Move embeddings to GPU
    embeddings_gpu = torch.from_numpy(embeddings).to(device)
    num_total = embeddings.shape[0]
    
    selected_indices = []
    # Distance sum accumulator
    dist_sum = torch.zeros(num_total, device=device, dtype=torch.float32)
    
    if start_idx is None:
        start_idx = random.randint(0, num_total - 1)
    
    current_idx = start_idx
    selected_indices.append(current_idx)
    dist_sum[current_idx] = -1e9 # Mask selected point with a very small value
    
    for _ in tqdm(range(1, n_samples)):
        last_emb = embeddings_gpu[current_idx].unsqueeze(0)
        
        # Calculate squared distance from the NEW point to ALL points
        # torch.cdist is highly optimized on GPU
        dists = torch.cdist(last_emb, embeddings_gpu, p=2).squeeze(0).pow(2)
        
        # Add to cumulative sum
        mask = dist_sum > -1e8
        dist_sum[mask] += dists[mask]
        
        # Select the point with maximum total distance
        current_idx = torch.argmax(dist_sum).item()
        
        selected_indices.append(current_idx)
        dist_sum[current_idx] = -1e9 # Mask selected
        
    return selected_indices

def fps_max_min_gpu(embeddings, n_samples, start_idx=None):
    """
    FPS Variant 2: Max-Min (Standard FPS).
    Logic: Select the point that is farthest from the CLOSEST selected point.
    """
    print(f"--- Starting FPS (Max-Min) on GPU (Target: {n_samples}) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU (this might be slow).")

    embeddings_gpu = torch.from_numpy(embeddings).to(device)
    num_total = embeddings.shape[0]
    
    selected_indices = []
    # Initialize minimum distances to infinity
    min_dists = torch.full((num_total,), float('inf'), device=device, dtype=torch.float32)
    
    if start_idx is None:
        start_idx = random.randint(0, num_total - 1)
        
    current_idx = start_idx
    selected_indices.append(current_idx)
    
    for _ in tqdm(range(1, n_samples)):
        last_emb = embeddings_gpu[current_idx].unsqueeze(0)
        
        # Calculate distance from the NEW point to ALL points
        dists = torch.cdist(last_emb, embeddings_gpu, p=2).squeeze(0).pow(2)
        
        # Update minimum distance: min(old_min, new_dist)
        min_dists = torch.minimum(min_dists, dists)
        
        # Mask selected points
        min_dists[current_idx] = -1.0
        for idx in selected_indices[-5:]: # Double check safety for recent indices
             min_dists[idx] = -1.0

        # Select the point with the largest minimum distance (The most isolated point)
        current_idx = torch.argmax(min_dists).item()
        selected_indices.append(current_idx)
        
    return selected_indices