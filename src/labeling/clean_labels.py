import json
import argparse
import os
import re
from tqdm import tqdm
from collections import Counter, defaultdict
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ==========================================
# Text Normalization Utils
# ==========================================

lemmatizer = WordNetLemmatizer()

def setup_nltk(nltk_path=None):
    """
    Ensure NLTK data is available. 
    If a path is provided, add it. Otherwise, try to download if missing.
    """
    if nltk_path and os.path.exists(nltk_path):
        print(f"Adding custom NLTK path: {nltk_path}")
        nltk.data.path.append(nltk_path)
    
    required_packages = ['punkt', 'wordnet', 'omw-1.4']
    # For newer NLTK versions, 'punkt_tab' might be needed instead of 'punkt'
    # We try to be robust here.
    
    print("Checking NLTK dependencies...")
    for pkg in required_packages:
        try:
            if pkg == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif pkg == 'wordnet':
                nltk.data.find('corpora/wordnet')
            elif pkg == 'omw-1.4':
                nltk.data.find('corpora/omw-1.4')
        except LookupError:
            print(f"Downloading missing NLTK package: {pkg}")
            try:
                nltk.download(pkg)
            except Exception as e:
                print(f"Warning: Failed to download {pkg}. Error: {e}")
                print("If you are offline, please use --nltk-path to point to local data.")

def normalize_for_display(label_text: str) -> str:
    """
    Level 1 Normalization: For display.
    Lowercase, remove special chars, keep natural form (e.g., "sorting").
    """
    if not label_text: return ""
    text = label_text.lower()
    text = text.replace('_', ' ')
    text = text.replace('-', ' ')
    # Remove non-alphanumeric except space
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Collapse multiple spaces
    return re.sub(r'\s+', ' ', text).strip()

def normalize_for_semantic(label_text: str) -> str:
    """
    Level 2 Normalization: For semantic grouping.
    Includes Lemmatization (e.g., "sorting" -> "sort").
    """
    # Base normalization first
    text = normalize_for_display(label_text)
    if not text: return ""
    
    try:
        tokens = word_tokenize(text)
        # Lemmatize as verbs (usually best for algorithms/actions)
        lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
        return ' '.join(lemmatized_tokens)
    except LookupError:
        # Fallback if NLTK data is completely missing
        return text

# ==========================================
# Main Processing Logic
# ==========================================

def process_labels(input_path, output_path):
    print(f"--- Reading file: {input_path} ---")
    
    all_records = []
    # Resilience: Check input exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            
    # --- Step 1: Global Analysis (Build Semantic Groups) ---
    print("\n--- Step 1: Analyzing global label frequencies... ---")
    
    # Cache: origin -> (display, semantic)
    orig_label_cache = {}
    # Map: semantic_form -> Counter({display_form: count})
    semantic_to_display_freqs = defaultdict(Counter)

    for record in tqdm(all_records, desc="Analyzing"):
        # Adapt to the output of Stage 1 (1_gen_labels.py)
        # Usually stored in record['label']['labels']
        current_labels = []
        if 'label' in record and isinstance(record['label'], dict) and 'labels' in record['label']:
            current_labels = record['label']['labels']
        elif 'algorithm_labels' in record: # Fallback for old data format
            current_labels = record['algorithm_labels']
            
        for orig_label in current_labels:
            if orig_label not in orig_label_cache:
                display_form = normalize_for_display(orig_label)
                semantic_form = normalize_for_semantic(orig_label)
                orig_label_cache[orig_label] = (display_form, semantic_form)
            
            display_form, semantic_form = orig_label_cache[orig_label]
            
            if display_form and semantic_form:
                semantic_to_display_freqs[semantic_form][display_form] += 1

    # --- Step 2: Determine Canonical Representatives ---
    print("\n--- Step 2: Selecting canonical representatives... ---")
    
    # Map: semantic_form -> canonical_display_form
    canonical_map = {}
    for semantic_form, display_counter in semantic_to_display_freqs.items():
        if display_counter:
            # Pick the most frequent display form as the "Official" one
            most_common = display_counter.most_common(1)[0][0]
            canonical_map[semantic_form] = most_common

    # --- Step 3: Apply Mapping & Save ---
    print("\n--- Step 3: Rewriting file with clean labels... ---")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for record in tqdm(all_records, desc="Saving"):
            
            # Identify where labels are stored
            target_key_parent = None
            target_key_child = None
            
            if 'label' in record and isinstance(record['label'], dict) and 'labels' in record['label']:
                raw_labels = record['label']['labels']
                target_key_parent = 'label'
                target_key_child = 'labels'
            elif 'algorithm_labels' in record:
                raw_labels = record['algorithm_labels']
                target_key_parent = None # Top level
                target_key_child = 'algorithm_labels'
            else:
                raw_labels = []

            final_labels = set()
            for orig_label in raw_labels:
                display_form, semantic_form = orig_label_cache.get(orig_label, (None, None))
                
                if not semantic_form: 
                    continue

                # Find canonical, fallback to display form
                canonical_label = canonical_map.get(semantic_form, display_form)
                if canonical_label:
                    final_labels.add(canonical_label)
            
            # Update record in place
            sorted_clean_labels = sorted(list(final_labels))
            
            if target_key_parent:
                record[target_key_parent][target_key_child] = sorted_clean_labels
            elif target_key_child:
                record[target_key_child] = sorted_clean_labels
            
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\nDone! Normalized data saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Clean and normalize labels via global frequency analysis.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL (from Stage 1)")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL")
    parser.add_argument("--nltk_path", type=str, default=None, help="Optional: Path to local NLTK data folder")
    
    args = parser.parse_args()

    # Setup NLTK (download or add path)
    setup_nltk(args.nltk_path)
    
    process_labels(args.input_file, args.output_file)