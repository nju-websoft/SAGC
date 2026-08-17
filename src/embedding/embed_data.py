import os
import re
import json
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ==========================================
# Utils & Extraction Logic
# ==========================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def extract_before_and_code_block_all(s: str):
    """
    Extract CoT (content before code) and Code block.
    """
    if not isinstance(s, str):
        return s, "", "TYPE_ERROR"

    
    patterns = {
        'python': re.compile(r"```(python)\s*?\n(.*?)\n```", flags=re.DOTALL),
        'cpp': re.compile(r"```(cpp)\s*?\n(.*?)\n```", flags=re.DOTALL),
        'any_lang': re.compile(r"```([a-z]+)\s*?\n(.*?)\n```", flags=re.DOTALL),
    }
    
    found_match = None
    
    for key in ['python', 'cpp', 'any_lang']:
        match = patterns[key].search(s)
        if match:
            found_match = match
            break 

    if found_match:
        code_content = found_match.group(2).strip()
        cot_content = s[:found_match.start()].strip()
        return cot_content, code_content, None
    else:
        return s.strip(), "", "NO_CODE_FOUND"

def extract_user_cot_code_all(dialogue):
    """
    Parse dialogue format: [{'role': 'user', ...}, {'role': 'assistant', ...}]
    """
    if not isinstance(dialogue, list) or len(dialogue) < 2:
        return "", "", "", "INVALID_DIALOGUE_STRUCTURE"
    
    user_query = dialogue[0]['content']
    assistant_response = dialogue[1]['content']
    
    cot, code, flag = extract_before_and_code_block_all(assistant_response)
    return user_query, cot, code, flag

def read_jsonl(path):
    data = []
    if not os.path.exists(path):
        print(f"Warning: File not found {path}")
        return []
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    return data

# ==========================================
# Dataset & Model
# ==========================================

class TextDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def input_features(input_texts, tokenizer, max_seq_len, device):
    """Tokenize inputs."""
    # Handle empty/None strings
    processed_texts = [t if t and t.strip() else " " for t in input_texts]
    
    encoding = tokenizer(
        processed_texts,
        padding='longest',
        max_length=max_seq_len,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    return input_ids, attention_mask

# ==========================================
# Main Embedding Loop
# ==========================================

def run_embedding(args):
    # --- DDP Init ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # --- Load Model ---
    if rank == 0:
        print(f"Loading model from {args.model_path}...")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use AutoModel for generic encoder or CausalLM if generative (Qwen style)
    try:
        model = AutoModel.from_pretrained(
            args.model_path, 
            attn_implementation="sdpa", 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to(device)
    except:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).to(device)
        model.config.pad_token_id = tokenizer.pad_token_id
        
    model.eval()

    # --- Process Data (Rank 0 reads) ---
    raw_data = []
    if rank == 0:
        print(f"Reading data from {args.data_path}")
        original_data = read_jsonl(args.data_path)
        
        # Debug Mode
        if args.debug:
            original_data = original_data[:100]
            
        print("Extracting CoT/Code fields...")
        error_log = []
        for item in tqdm(original_data):
            # Adapt to key name 'dialogs' (from Stage 1) or 'messages'
            dialogs = item.get('dialogs', item.get('messages', []))
            
            query, cot, code, flag = extract_user_cot_code_all(dialogs)
            
            if flag and flag != "NO_CODE_FOUND":
                 error_log.append(item.get('id_ddm', 'unknown'))

            # Store all fields, select later based on 'mode'
            raw_data.append({
                'id': item.get('id_ddm', str(random.randint(0, 1e9))),
                'query': query,
                'response': dialogs[1]['content'] if len(dialogs)>1 else "",
                'cot': cot,
                'code': code
            })
            
        if error_log:
            print(f"Warning: {len(error_log)} items had extraction issues.")

    # Broadcast data to all ranks
    object_list = [raw_data]
    dist.broadcast_object_list(object_list, src=0)
    raw_data = object_list[0]

    # --- DataLoader ---
    dataset = TextDataset(raw_data)
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
    save_dir = os.path.join(args.output_dir, args.mode) # e.g. outputs/code
    os.makedirs(save_dir, exist_ok=True)
    
    buffer_ids = []
    buffer_embeddings = []
    part_id = 0
    save_interval = 10000 

    if rank == 0:
        iterator = tqdm(loader, desc="Vectorizing")
    else:
        iterator = loader

    for batch in iterator:
        ids = batch['id']
        
        # Select text based on mode
        texts = []
        if args.mode == 'code':
            texts = batch['code']
        elif args.mode == 'cot':
            texts = batch['cot']
        elif args.mode == 'query':
            texts = batch['query']
        elif args.mode == 'response':
            texts = batch['response']
        else:
            # Fallback or combination logic
            texts = batch['code']

        input_ids, attention_mask = input_features(texts, tokenizer, args.max_seq_len, device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True, 
                return_dict=True
            )
            # Use last token of the last hidden state
            last_hidden = outputs.hidden_states[-1]
            embeddings = last_hidden[:, -1, :] 

        buffer_ids.extend(ids)
        buffer_embeddings.append(embeddings.float().cpu().numpy())

        # Periodic Save
        if len(buffer_ids) >= save_interval:
            embeddings_np = np.concatenate(buffer_embeddings, axis=0)
            save_path = os.path.join(save_dir, f'embedding_rank{rank}_part{part_id}.npz')
            np.savez(save_path, id_ddm=np.array(buffer_ids), embedding=embeddings_np)
            
            part_id += 1
            buffer_ids = []
            buffer_embeddings = []

    # Final Save
    if buffer_ids:
        embeddings_np = np.concatenate(buffer_embeddings, axis=0)
        save_path = os.path.join(save_dir, f'embedding_rank{rank}_part{part_id}.npz')
        np.savez(save_path, id_ddm=np.array(buffer_ids), embedding=embeddings_np)

    dist.barrier()
    if rank == 0:
        print(f"Done. Embeddings saved to {save_dir}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to Embedding Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/embeddings")
    
    parser.add_argument("--mode", type=str, default="code", choices=['code', 'cot', 'query', 'response'], help="Which field to embed")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--debug", action="store_true", help="Run on small subset")

    args = parser.parse_args()
    
    set_seed(42)
    
    if "LOCAL_RANK" in os.environ:
        run_embedding(args)
    else:
        print("Please run with: torchrun --nproc_per_node=N src/stage2_embedding/embed_data.py ...")