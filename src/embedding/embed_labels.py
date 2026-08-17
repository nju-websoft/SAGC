import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel

# ==========================================
# Utils
# ==========================================

class LabelDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def load_labels_strictly(input_path: str) -> list:
    """
    Load labels line-by-line, strictly preserving the order.
    Do NOT sort or filter. This ensures alignment with Stage 3's One-Hot matrix.
    """
    print(f"Loading labels from {input_path}...")
    labels = []
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Label vocab file not found: {input_path}")

    # Only support txt/jsonl line-based format to ensure strict order
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                labels.append(text)
            
    print(f"Loaded {len(labels)} labels. Order preserved.")
    return labels

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Standard pooling for decoder-only models (like Qwen/GTE) with left-padding.
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# ==========================================
# Core Logic
# ==========================================

def run_embedding(args):
    # --- DDP Init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # --- Load Model ---
    if rank == 0:
        print(f"Loading model from {args.model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    
    model = AutoModel.from_pretrained(
        args.model_path, 
        attn_implementation="sdpa", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).cuda()
    model.eval()

    # --- Prepare Data (Strict Order) ---
    labels_to_process = []
    if rank == 0:
        # Only Rank 0 reads the file
        labels_to_process = load_labels_strictly(args.input_file)
    
    # Broadcast to ensure all ranks have exact same list
    object_list = [labels_to_process]
    dist.broadcast_object_list(object_list, src=0)
    labels_to_process = object_list[0]

    # Dataset
    raw_data = [{'label_text': l} for l in labels_to_process]
    dataset = LabelDataset(raw_data)
    # shuffle=False is critical here!
    sampler = DistributedSampler(dataset, shuffle=False) 
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    # --- Inference ---
    buffer_labels = []
    buffer_embeddings = []
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        iterator = tqdm(loader, desc="Vectorizing Labels")
    else:
        iterator = loader

    for batch in iterator:
        batch_texts = batch['label_text']
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=args.max_seq_len, 
            return_tensors="pt"
        ).to(local_rank)

        with torch.no_grad():
            outputs = model(**inputs)
            # Adapt this if your model is BERT-like (use cls_token) vs LLM-like (use last_token)
            # Assuming Qwen/GTE (LLM-based):
            embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
        buffer_labels.extend(batch_texts)
        buffer_embeddings.append(embeddings.float().cpu().numpy())

    # --- Save Partial Results ---
    if buffer_embeddings:
        all_embeddings = np.concatenate(buffer_embeddings, axis=0)
        # We save 'embedding' (singular) to match utils_io.py expectation
        # We also save the labels so we can double-check order later if needed
        save_path = os.path.join(args.output_dir, f'embedding_rank{rank}.npz')
        
        np.savez(save_path, 
                 labels=np.array(buffer_labels), 
                 embedding=all_embeddings) # Key name matches Stage 3 loader
                 
        print(f"[Rank {rank}] Saved {len(buffer_labels)} vectors to {save_path}")

    dist.barrier()
    if rank == 0:
        print(f"All ranks finished. Output saved to {args.output_dir}")

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vectorize labels strictly following input order.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to label_vocab.txt (must be pre-sorted)")
    parser.add_argument("--output_dir", type=str, required=True, help="Dir to save .npz files")
    parser.add_argument("--model_path", type=str, required=True, help="Path to Embedding Model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=128)
    # Removed --min_count since filtering should happen in Stage 1
    
    args = parser.parse_args()
    
    if "LOCAL_RANK" in os.environ:
        run_embedding(args)
    else:
        print("Please run using: torchrun --nproc_per_node=N src/embedding/embed_labels.py ...")