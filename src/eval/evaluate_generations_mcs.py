import os
import json
import argparse
import sys
from typing import List

# ==========================================
# [Patch] Ensure lcb_runner can be imported
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import lcb_runner modules
from lcb_runner.utils.scenarios import Scenario
from lcb_runner.runner.scenario_router import (
    CodeGenerationProblem,
    load_code_generation_dataset
)
from lcb_runner.evaluation.compute_code_generation_metrics import (
    compute_comprehensive_metrics
)

def main():
    parser = argparse.ArgumentParser(
        description="Run standalone Standard MCS evaluation on generation files."
    )
    
    # Input file path (instead of directory)
    parser.add_argument("--input_file", type=str, required=True, help="Full path to the generations.json file.")
    
    parser.add_argument("--release_version", type=str, default="release_v6", help="The dataset version to use.")
    parser.add_argument("--start_date", type=str, default="2024-09-20", help="Filter start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", type=str, default=None, help="Filter end date (YYYY-MM-DD)")
    parser.add_argument("--num_process_evaluate", type=int, default=16, help="Number of processes for evaluation.")
    parser.add_argument("--timeout", type=int, default=6, help="Timeout in seconds for each test case.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    
    args = parser.parse_args()

    # 1. Check input file
    print(f"Loading generated results from: {args.input_file}")
    if not os.path.exists(args.input_file):
        print(f"Error: File not found at {args.input_file}")
        return
        
    with open(args.input_file, 'r') as f:
        generated_results = json.load(f)

    # Handle different output formats (List or Dict)
    if isinstance(generated_results, dict) and "code_list" not in generated_results:
        # Placeholder for handling nested LCB native formats if necessary
        pass

    generations_map = {item['question_id']: item['code_list'] for item in generated_results}
    print(f"Found {len(generations_map)} problems with generated code.")

    # 2. Load the dataset
    print(f"Loading dataset version: {args.release_version} (Date >= {args.start_date})")
    full_dataset = load_code_generation_dataset(
        release_version=args.release_version, 
        start_date=args.start_date,
        end_date=args.end_date
    )

    # 3. Align data (Match generations with dataset problems)
    benchmark: List[CodeGenerationProblem] = []
    generations_for_eval: List[List[str]] = []
    
    for problem in full_dataset:
        if problem.question_id in generations_map:
            benchmark.append(problem)
            generations_for_eval.append(generations_map[problem.question_id])
        
    print(f"Matched {len(benchmark)} problems for evaluation.")
    
    if len(benchmark) == 0:
        print("Warning: No matching problems found! Check your dataset version or date filter.")
        return

    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]

    # 4. Run Comprehensive Evaluation
    print("\nStarting Comprehensive Evaluation...")
    summary_metrics, detailed_mcs_scores = compute_comprehensive_metrics(
        eval_samples,
        generations_for_eval,
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.timeout,
        debug=args.debug
    )

    # 5. Print Summary Results
    print("\n" + "="*80)
    print("Comprehensive Evaluation Results:")
    print("="*80)
    
    if "standard_mcs" in summary_metrics:
        val = summary_metrics["standard_mcs"]
        print(f"Standard MCS:\n  - Overall Average MCS: {val:.4f}\n")

    for name, scores in summary_metrics.items():
        if "pass_" in name:
            print(f"{name.replace('_', ' ').title()}:")
            if isinstance(scores, dict):
                for k, v in scores.items():
                    print(f"  - {k}: {v:.4f}")
            else:
                print(f"  - {scores}")
            print("")
    print("="*80)

    # 6. Save Detailed Results (Saved to the same directory as input file)
    input_dir = os.path.dirname(args.input_file)
    base_filename = os.path.basename(args.input_file).replace('.json', '')
    results_path = os.path.join(input_dir, f"{base_filename}_scores.json")
    
    problem_ids = [p.question_id for p in benchmark]
    final_detailed_results = {
        problem_ids[idx]: mcs_scores 
        for idx, mcs_scores in detailed_mcs_scores.items()
    }

    save_data = {
        "summary_metrics": summary_metrics,
        "detailed_mcs_per_sample": final_detailed_results
    }

    with open(results_path, 'w') as f:
        json.dump(save_data, f, indent=4)
        
    print(f"\nDetailed results saved to: {results_path}")

if __name__ == "__main__":
    main()