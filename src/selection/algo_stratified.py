import numpy as np
import random
from collections import defaultdict

def stratified_sample(cluster_labels: np.ndarray, n_samples: int) -> list:
    """
    Perform Stratified Random Sampling based on cluster labels.
    
    Core Logic:
    1. Calculate quota for each cluster (Total / K).
    2. Handle insufficient clusters: If a cluster has fewer samples than the quota,
       take all of them and redistribute the deficit to other rich clusters.
    """
    print(f"--- Starting Stratified Sampling (Target: {n_samples}) ---")
    
    # 1. Organize indices by cluster
    clusters = defaultdict(list)
    for idx, label in enumerate(cluster_labels):
        clusters[label].append(idx)
    
    unique_labels = sorted(clusters.keys())
    n_clusters = len(unique_labels)
    
    if n_clusters == 0:
        return []

    # 2. Pre-shuffle indices within each cluster (Ensures randomness)
    for label in unique_labels:
        random.shuffle(clusters[label])

    # 3. Calculate initial quotas (Average distribution)
    base_quota = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    quotas = {label: base_quota for label in unique_labels}
    
    # Distribute remainder randomly
    for label in random.sample(unique_labels, remainder):
        quotas[label] += 1
        
    # 4. Dynamic Re-balancing
    # If a cluster is too small, take all its samples and distribute the deficit to others.
    while True:
        deficit = 0
        rich_clusters = []
        
        for label in unique_labels:
            available = len(clusters[label])
            assigned = quotas[label]
            
            if assigned > available:
                deficit += (assigned - available)
                quotas[label] = available # Max out this cluster
            elif assigned < available:
                rich_clusters.append(label)
        
        if deficit == 0 or not rich_clusters:
            break
            
        # Redistribute deficit to rich clusters
        for _ in range(deficit):
            if not rich_clusters: break
            target = random.choice(rich_clusters)
            quotas[target] += 1

    # 5. Execute Sampling
    selected_indices = []
    for label, quota in quotas.items():
        # Since we shuffled in step 2, slicing implies random sampling
        selected_indices.extend(clusters[label][:quota])
        
    print(f"Stratified sampling finished. Selected: {len(selected_indices)}")
    return selected_indices