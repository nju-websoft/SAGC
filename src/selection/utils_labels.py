import numpy as np
import json
from tqdm import tqdm
from utils_io import load_and_reorder_embeddings

def read_label_vocab_strictly(path):
    """
    [CRITICAL] Read label vocabulary strictly line-by-line.
    Do NOT sort or dedup here. The order must match 'label_vocab.txt' exactly,
    because Stage 2 (Embedding) used this exact order.
    """
    if not path:
        return []
    with open(path, 'r', encoding='utf-8') as f:
        # Read lines, strip whitespace, keep exact order
        labels = [line.strip() for line in f if line.strip()]
    
    # Optional: Check for duplicates just in case, but don't reorder
    if len(labels) != len(set(labels)):
        print(f"Warning: Duplicate labels found in {path}. Indices might be ambiguous.")
        
    return labels

def read_jsonl_labels(path):
    """Read all labels from the dataset jsonl."""
    labels_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading Labels from JSONL"):
            try:
                item = json.loads(line)
                # Logic to extract labels compatible with Stage 1 output
                # Priority: item['label']['labels'] -> item['algorithm_labels']
                if 'label' in item and isinstance(item['label'], dict) and 'labels' in item['label']:
                    labels_list.append(item['label']['labels'])
                elif 'algorithm_labels' in item:
                    labels_list.append(item['algorithm_labels'])
                else:
                    labels_list.append([])
            except:
                labels_list.append([])
    return labels_list

def get_one_hot_embeddings(data_path, vocab_path):
    """Generate Multi-Hot (One-Hot) embeddings from raw labels."""
    print("Generating Multi-Hot Label Embeddings...")
    
    # 1. Load Data
    raw_labels = read_jsonl_labels(data_path)
    
    # 2. Load Vocab (Strict Order)
    vocab = read_label_vocab_strictly(vocab_path)
    print(f"Vocab size: {len(vocab)}")
    
    # Map label string to column index
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    
    n = len(raw_labels)
    dim = len(vocab)
    
    one_hot = np.zeros((n, dim), dtype=np.float32)
    
    # 3. Build Matrix
    for i, labels in enumerate(tqdm(raw_labels, desc="Encoding One-Hot")):
        for label in labels:
            # Only use labels that exist in our vocab (ignore low-freq ones filtered in Stage 1)
            if label in label_to_idx:
                idx = label_to_idx[label]
                one_hot[i, idx] = 1.0
                
    return one_hot

def get_summed_embeddings(data_path, vocab_path, label_emb_dir):
    """
    Generate Summed embeddings: Sum(Label_Embeddings).
    Mathematically equivalent to: OneHot_Matrix @ Label_Embedding_Matrix
    """
    print("Generating Summed Label Embeddings...")
    
    # 1. Get Multi-Hot Matrix (N, VocabSize)
    one_hot = get_one_hot_embeddings(data_path, vocab_path)
    
    # 2. Load Pre-computed Label Embeddings (VocabSize, HiddenDim)
    # The 'target_size' ensures we match the vocab dimension exactly
    _, semantic_matrix = load_and_reorder_embeddings(label_emb_dir, target_size=one_hot.shape[1])
    
    print(f"Computing dot product: {one_hot.shape} @ {semantic_matrix.shape}")
    
    # 3. Matrix Multiplication
    # Result: (N, HiddenDim)
    summed = one_hot @ semantic_matrix
    return summed

def get_tfidf_embeddings(data_path, vocab_path, label_emb_dir):
    """Generate TF-IDF Weighted Label Embeddings."""
    print("Generating TF-IDF Label Embeddings...")
    
    one_hot = get_one_hot_embeddings(data_path, vocab_path)
    _, semantic_matrix = load_and_reorder_embeddings(label_emb_dir, target_size=one_hot.shape[1])
    
    # Calculate IDF (Inverse Document Frequency)
    # doc_freq: How many samples contain label i
    doc_freq = one_hot.sum(axis=0) 
    n_samples = one_hot.shape[0]
    
    # Smoothing to avoid div by zero
    idf = np.log((n_samples + 1) / (doc_freq + 1)) + 1
    
    # Apply weights (Broadcasting: each column j is multiplied by idf[j])
    print("Applying IDF weights...")
    weighted_one_hot = one_hot * idf
    
    # Project to embedding space
    tfidf_emb = weighted_one_hot @ semantic_matrix
    return tfidf_emb.astype(np.float32)

def get_max_pooled_embeddings(data_path, vocab_path, label_emb_dir):
    """
    Generate Max-Pooled Label Embeddings.
    Logic: For each sample, take the element-wise MAX of all its label vectors.
    """
    print("Generating Max-Pooled Label Embeddings...")
    raw_labels = read_jsonl_labels(data_path)
    vocab = read_label_vocab_strictly(vocab_path)
    
    _, semantic_matrix = load_and_reorder_embeddings(label_emb_dir, target_size=len(vocab))
    
    label_to_idx = {l: i for i, l in enumerate(vocab)}
    
    n = len(raw_labels)
    dim = semantic_matrix.shape[1]
    
    max_pooled = np.zeros((n, dim), dtype=np.float32)
    
    for i, labels in enumerate(tqdm(raw_labels, desc="Max Pooling")):
        # Find indices of labels present in this sample
        indices = [label_to_idx[l] for l in labels if l in label_to_idx]
        
        if indices:
            # Gather vectors: [Num_Labels_In_Sample, Dim]
            vectors = semantic_matrix[indices]
            # Max along axis 0
            max_pooled[i] = np.max(vectors, axis=0)
        # Else: keeps as zeros (no labels found)
            
    return max_pooled