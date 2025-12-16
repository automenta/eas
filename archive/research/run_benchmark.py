#!/usr/bin/env python3
"""
run_benchmark.py - Master script for AFT Universality Benchmark (No Rich)
"""

import os
import sys
import json
import time

# Add current directory to path to import aft_universality
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from aft_universality import run_experiment

# Configuration
MODELS = [
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "EleutherAI/pythia-410m",
    "microsoft/phi-1_5",
    "stabilityai/stablelm-2-1_6b",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat"
]

DATASETS = [
    "hellaswag",
    "arc_challenge",
    "gsm8k"
]

RESULTS_FILE = "results/benchmark_summary.json"

def main():
    import gc
    import torch
    
    results_grid = {}
    
    print(f"🚀 Starting Benchmark: {len(MODELS)} Models x {len(DATASETS)} Datasets")
    print(f"Results will be saved to {RESULTS_FILE}")
    
    # Initialize grid
    for model in MODELS:
        for ds in DATASETS:
            results_grid[f"{model}_{ds}"] = {"status": "pending"}

    # Run Benchmark
    for model in MODELS:
        for ds in DATASETS:
            key = f"{model}_{ds}"
            results_grid[key]["status"] = "running"
            
            print(f"\n👉 Running {model} on {ds}...")
            
            try:
                # Memory Cleanup
                gc.collect()
                torch.cuda.empty_cache()
                
                # Run Experiment
                start_time = time.time()
                res = run_experiment(
                    model_name=model,
                    dataset_name=ds,
                    auto_layer=True,
                    epochs=3, 
                    lr=1e-3,
                    device="cuda",
                    verbose=True, # Show live progress
                    limit=100 # Rapid Prototyping Mode
                )
                
                if "error" in res:
                    results_grid[key] = {"status": "error", "error": res["error"]}
                    print(f"  ❌ Error: {res['error']}")
                else:
                    results_grid[key] = {
                        "status": "done",
                        "improvement": res["improvement"],
                        "baseline": res["baseline_acc"],
                        "final": res["final_acc"]
                    }
                    imp = res["improvement"]
                    print(f"  ✅ Done! Improvement: {imp:+.1%}")
                    
            except Exception as e:
                results_grid[key] = {"status": "error", "error": str(e)}
                print(f"  ❌ Exception: {e}")
            
            # Save partial results
            with open(RESULTS_FILE, "w") as f:
                json.dump(results_grid, f, indent=2)
                
            # Aggressive Cleanup
            gc.collect()
            torch.cuda.empty_cache()
                
    print(f"\n✅ Benchmark Complete! Results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
