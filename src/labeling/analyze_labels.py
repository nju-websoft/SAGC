import json
import argparse
import os
import csv
from collections import Counter
from itertools import combinations
from tqdm import tqdm

def read_jsonl(path: str) -> list:
    """Read JSONL file efficiently."""
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return []
    
    dataset = []
    with open(path, 'r', encoding='utf-8') as f:
        print(f"Reading data from {path}...")
        for line in tqdm(f):
            if line.strip():
                try:
                    dataset.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return dataset

def analyze_and_export(data: list, top_n: int, output_csv_path: str = None, output_vocab_path: str = None, min_count: int = 1):
    if not data:
        print("No data to analyze.")
        return

    print("Analyzing label statistics...")
    label_counter = Counter()
    word_document_frequency = Counter()
    labels_per_record_dist = Counter()
    cooccurrence_counter = Counter()
    
    total_labels_count = 0
    records_with_labels = 0

    for record in data:
        # Compatible with both old and new data formats
        labels = []
        if 'label' in record and isinstance(record['label'], dict) and 'labels' in record['label']:
            labels = record['label']['labels']
        elif 'algorithm_labels' in record:
            labels = record['algorithm_labels']
            
        if not labels:
            continue

        records_with_labels += 1
        num_labels = len(labels)
        total_labels_count += num_labels
        labels_per_record_dist[num_labels] += 1
        
        # Count labels
        label_counter.update(labels)
        
        # Count words
        words_in_record = set()
        for label in labels:
            words_in_record.update(str(label).lower().split())
        word_document_frequency.update(words_in_record)

        # Count Co-occurrence
        if len(labels) >= 2:
            sorted_labels = sorted(labels)
            cooccurrence_counter.update(combinations(sorted_labels, 2))

    if records_with_labels == 0:
        print("No records with valid labels found.")
        return

    # --- 1. Export Statistics CSV ---
    if output_csv_path:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        try:
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Label', 'Count', 'Frequency(%)'])
                # Sort by count desc
                for label, count in label_counter.most_common():
                    freq = (count / records_with_labels) * 100
                    csv_writer.writerow([label, count, f"{freq:.4f}"])
            print(f"Full statistics saved to: {output_csv_path}")
        except IOError as e:
            print(f"Error saving CSV: {e}")

    # --- 2. [CRITICAL] Export Canonical Vocab TXT ---
    # This file serves as the index map for Stage 2 (Embedding) and Stage 3 (OneHot)
    if output_vocab_path:
        os.makedirs(os.path.dirname(output_vocab_path), exist_ok=True)
        
        # Filter by min_count
        valid_labels = [l for l, c in label_counter.items() if c >= min_count]
        
        # MUST BE SORTED to ensure index alignment!
        sorted_labels = sorted(valid_labels)
        
        print(f"\nExporting {len(sorted_labels)} unique labels (freq >= {min_count}) to {output_vocab_path}...")
        
        try:
            with open(output_vocab_path, 'w', encoding='utf-8') as f:
                for label in sorted_labels:
                    f.write(label + '\n')
            print("Vocab export successful.")
        except IOError as e:
            print(f"Error saving Vocab: {e}")

    # --- Console Summary ---
    print("\n" + "="*50)
    print(" LABEL ANALYSIS REPORT ")
    print("="*50)
    print(f"Total records: {len(data)}")
    print(f"Unique labels (Total): {len(label_counter)}")
    if output_vocab_path:
        print(f"Unique labels (Filtered >= {min_count}): {len(sorted_labels)}")
        
    print(f"\nTop {top_n} Common Labels:")
    for label, count in label_counter.most_common(top_n):
        freq = (count / records_with_labels) * 100
        print(f"{label:<30} | {count:<6} | {freq:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Analyze label statistics and generate vocab.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to cleaned JSONL file")
    
    parser.add_argument("--output_csv", type=str, default=None, help="Path to save statistics CSV")
    parser.add_argument("--output_vocab", type=str, default=None, help="Path to save sorted label vocabulary (.txt)")
    
    parser.add_argument("--min_count", type=int, default=5, help="Minimum frequency to include in vocab")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top labels to show in console")
    
    args = parser.parse_args()
    dataset = read_jsonl(args.input_file)
    analyze_and_export(dataset, args.top_n, args.output_csv, args.output_vocab, args.min_count)

if __name__ == '__main__':
    main()