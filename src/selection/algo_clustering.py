import numpy as np
import faiss
import time

def run_kmeans(embeddings: np.ndarray, k: int, seed: int = 1234):
    """
    Run K-Means clustering using Faiss (CPU Mode).
    
    Using CPU ensures deterministic results and avoids float16 precision issues
    often encountered with Faiss-GPU clustering.
    
    Returns:
        labels: (N,) array of cluster assignments
    """
    print(f"--- Starting K-Means (k={k}, mode=CPU) ---")
    n, d = embeddings.shape
    start = time.time()

    # 1. Instantiate (niter=25 is a standard configuration)
    kmeans = faiss.Kmeans(d, k, niter=25, verbose=True, seed=seed)
    
    # 2. Train
    kmeans.train(embeddings)
    
    # 3. Assign Labels
    # index.search returns (distances, labels)
    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.ravel()

    print(f"K-Means finished in {time.time() - start:.2f}s")
    return labels