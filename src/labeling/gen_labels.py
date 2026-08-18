import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Optional


try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError:
    print("Error: vllm or transformers not installed. Please install them in the main environment.")
    exit(1)


def read_jsonl(path):
    data = []
    if not os.path.exists(path):
        return []
        
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data

def save_jsonl(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def load_processed_ids(path_save: str) -> set:
    if not os.path.exists(path_save):
        return set()
    ids = set()
    with open(path_save, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                if "id_ddm" in item:
                    ids.add(item["id_ddm"])
            except:
                continue
    return ids

def extract_label_dict(text: str) -> Optional[Dict]:
    #Extract JSON object from markdown code block
    start_tag = "```label"
    end_tag = "```"

    start_index = text.find(start_tag)
    if start_index == -1:
        return None

    end_index = text.find(end_tag, start_index + len(start_tag))
    if end_index == -1:
        return None
    
    start_index = start_index + len(start_tag)
    label_json = text[start_index:end_index].strip()
    
    try:
        label_dict = json.loads(label_json)
        if "labels" in label_dict and isinstance(label_dict["labels"], list):
            label_dict["labels"] = [str(l).lower().strip() for l in label_dict["labels"]]
            return label_dict
        return None
    except json.JSONDecodeError:
        return None

def truncate_user_prompt_robust(tokenizer, sys_prompt, user_prompt, max_length, reserved_for_output):
    """Ensure the prompt doesn't exceed model context length"""
    sys_tokens = tokenizer.encode(sys_prompt, add_special_tokens=False)
    buffer = 100 
    user_max_tokens = max_length - len(sys_tokens) - reserved_for_output - buffer
    
    if user_max_tokens <= 0:
        return "" 

    user_tokens = tokenizer.encode(user_prompt, add_special_tokens=False)
    if len(user_tokens) <= user_max_tokens:
        return user_prompt
    
    truncated_user_tokens = user_tokens[:user_max_tokens]
    return tokenizer.decode(truncated_user_tokens, clean_up_tokenization_spaces=False)

# ==========================================
# Prompts
# ==========================================

def get_prompt_complex(problem):
    """Original 'gen_label_wo_selection_prompt' (Chain of Thought)"""
    sys_prompt = "You are a senior AI Python programming assistant. Your task is to analyze a given programming problem to generate a set of descriptive labels. These labels should accurately capture the core essence of the task, including the problem domain, algorithms/data structures used, and key programming concepts."

    
    user_prompt = f"""
Input Provided:
- Problem Description

Your Task:
1. Analyze the Input: Carefully read the problem description. Understand what the problem is asking and how to use code to solve it.
2. Identify Key Characteristics: Deconstruct its fundamental components.
    - What is the main goal? (e.g., searching, sorting, data processing)
    - What specific algorithms are needed? (e.g., binary search, depth-first search, dynamic programming)
    - What primary data structures are involved? (e.g., array, hash map, tree, graph)
    - Are there any notable programming paradigms or techniques? (e.g., recursion, bit manipulation, object-oriented programming)
3. Generate Labels: Based on your analysis, create a list of labels that are: 
    - Concise: Use short, widely-recognized terms (e.g., "Binary Search").
    - Canonical: Use standard, industry-accepted names.
    - Relevant: Each label must be supported by evidence in the problem description.

Output Format (Strictly Follow This Order):

First, provide a detailed analysis. Second, provide the final JSON output of labels enclosed in markdown fences (start with "```label" and end with "```").

1. Analysis of Key Characteristics: 
    - Main Goal: ...
    - Algorithms: ...
    - Data Structures: ...
    - Techniques: ...

2. Generated Labels
```label
{{
  "labels": [
    "Generated_Label_1",
    "Generated_Label_2",
    "..."
  ]
}}
```

Input:
Problem:
{problem}
"""
    return sys_prompt, user_prompt

def get_prompt_easy(problem):
    """Original 'gen_label_wo_selection_prompt_eazy'"""
    sys_prompt = "You are a senior AI Python programming assistant. Your task is to analyze a given programming problem to extract a set of descriptive labels. These labels should accurately capture the core essence of the task."

    
    user_prompt = f"""
Input Provided:
- Problem Description

Your Task:
Analyze the given problem. Generate descriptive labels that capture the core essence of the task. Provide the final JSON output of labels enclosed in markdown fences (start with "```label" and end with "```").

Output Format (Strictly Follow This Order):

```label
{{
  "labels": [
    "Generated_Label_1",
    "Generated_Label_2",
    "..."
  ]
}}
```

Input:
Problem:
{problem}
"""
    return sys_prompt, user_prompt

# ==========================================
# Main Logic
# ==========================================

def main(args):
    # 1. Initialize Model
    print(f"Loading model from: {args.model_path}")
    model = LLM(
        model=args.model_path, 
        gpu_memory_utilization=0.9, 
        tensor_parallel_size=1,  # Assuming single GPU per process
        max_model_len=32768,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Setup Sampling Params
    sampling_params = SamplingParams(
        max_tokens=2048, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=20
    )

    # 2. Load Data
    print(f"Loading data from: {args.input_file}")
    dataset = read_jsonl(args.input_file)
    
    # Handle slicing
    start = args.input_start
    end = args.input_end if args.input_end > 0 else len(dataset)
    dataset = dataset[start:end]
    
    processed_ids = load_processed_ids(args.output_file)
    print(f"Total samples to process: {len(dataset)}, Already processed: {len(processed_ids)}")

    # 3. Prepare Batches
    messages = []
    samples = []

    for sample in dataset:
        if sample.get("id_ddm") in processed_ids:
            continue
            
        # Extract problem text
        try:
            problem = sample['dialogs'][0]['content']
        except KeyError:
            continue

        if args.easy_mode:
            sys_prompt, user_prompt_full = get_prompt_easy(problem)
        else:
            sys_prompt, user_prompt_full = get_prompt_complex(problem)
            
        user_prompt_truncated = truncate_user_prompt_robust(
            tokenizer, sys_prompt, user_prompt_full, 
            model.llm_engine.model_config.max_model_len, 
            sampling_params.max_tokens
        )

        messages.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt_truncated},
        ])
        samples.append(sample)

    if not messages:
        print("No new samples to process.")
        return

    # 4. Inference Loop
    block_size = args.batch_size
    print(f"Starting inference on {len(messages)} samples...")
    
    for i in tqdm(range(0, len(messages), block_size), desc="Generating Labels"):
        batch_msgs = messages[i:i + block_size]
        batch_samples = samples[i:i + block_size]
        
        # VLLM Inference
        responses = model.chat(batch_msgs, sampling_params=sampling_params, use_tqdm=False)
        outputs = [r.outputs[0].text for r in responses]

        output_data = []
        for sample, output_text in zip(batch_samples, outputs):
            # Parse Result
            
            label_dict = extract_label_dict(output_text)
            
            # Construct Output Sample
            new_sample = sample.copy()
            
            if "dialogs" in new_sample:
                new_sample['dialogs'].append({
                    "role": "assistant",
                    "content": output_text
                })
            
            if label_dict:
                new_sample['algorithm_labels'] = label_dict["labels"]
            else:
                new_sample['algorithm_labels'] = {} # Empty if failed
            
            output_data.append(new_sample)

        # Save Checkpoint 
        save_jsonl(output_data, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output jsonl")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-8B", help="Path to LLM")
    
    parser.add_argument("--input_start", type=int, default=0)
    parser.add_argument("--input_end", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--easy_mode", action="store_true", help="Use simple prompt instead of CoT")
    
    args = parser.parse_args()
    main(args)
