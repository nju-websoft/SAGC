import numpy as np
import faiss

def normalize_l2(embeddings):
    """Apply L2 normalization in-place using Faiss."""
    faiss.normalize_L2(embeddings)
    return embeddings

def combine_embeddings(primary_emb, secondary_emb, alpha=0.5):
    """
    Concatenate two embedding spaces with weighting.
    Formula: [ primary * (1-alpha), secondary * alpha ]
    Then L2 normalize the result.
    """
    print(f"--- Combining Embeddings (Alpha={alpha}) ---")
    print(f"Primary Shape: {primary_emb.shape}")
    print(f"Secondary Shape: {secondary_emb.shape}")
    
    if primary_emb.shape[0] != secondary_emb.shape[0]:
        raise ValueError("Sample count mismatch between primary and secondary embeddings.")

    # 1. Normalize individually first
    normalize_l2(primary_emb)
    normalize_l2(secondary_emb)
    
    # 2. Scale
    scaled_primary = primary_emb * (1.0 - alpha)
    scaled_secondary = secondary_emb * alpha
    
    # 3. Concatenate
    combined = np.concatenate([scaled_primary, scaled_secondary], axis=1)
    print(f"Combined Shape: {combined.shape}")
    
    # 4. Final Normalization
    normalize_l2(combined)
    
    return combined

def combine_three_embeddings(emb1, emb2, emb3):
    """Concatenate three embeddings (Simple concat + Normalize)."""
    print(f"--- Combining Three Embeddings ---")
    normalize_l2(emb1)
    normalize_l2(emb2)
    normalize_l2(emb3)
    
    combined = np.concatenate([emb1, emb2, emb3], axis=1)
    normalize_l2(combined)
    return combined