#!/usr/bin/env python3
"""
run_all_pre_mortems.py

Master script to execute all pre-mortem tests and make go/no-go decision.
Run this first before investing time in experiments.
"""

import subprocess
import json
import os
from datetime import datetime


def run_command(cmd):
    """Run command and return exit code."""
    print(f"\n>>> Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    print("=" * 70)
    print("ACTIVATION SPACE RESEARCH: PRE-MORTEM TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Create results directory
    os.makedirs("results/pre_mortems", exist_ok=True)
    
    # Run pre-mortems
    tests = [
        ("Pre-Mortem 1: Cross-Model Similarity", "python pre_mortem_1_cross_model.py"),
        ("Pre-Mortem 2: Error Prediction", "python pre_mortem_2_error_prediction.py"),
        ("Pre-Mortem 3: Activation Patching", "python pre_mortem_3_patching.py"),
    ]
    
    for name, cmd in tests:
        print(f"\n{'='*70}")
        print(f"RUNNING: {name}")
        print("="*70)
        exit_code = run_command(cmd)
        if exit_code != 0:
            print(f"WARNING: {name} exited with code {exit_code}")
    
    # Load and summarize results
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL PRE-MORTEMS")
    print("=" * 70)
    
    results = {}
    result_files = [
        ("PM1: Cross-Model", "results/pre_mortems/pm1_results.json"),
        ("PM2: Error Prediction", "results/pre_mortems/pm2_results.json"),
        ("PM3: Patching", "results/pre_mortems/pm3_results.json"),
    ]
    
    for name, path in result_files:
        try:
            with open(path) as f:
                data = json.load(f)
                results[name] = data.get("result", "UNKNOWN")
                print(f"  {name}: {results[name]}")
        except FileNotFoundError:
            results[name] = "NOT_RUN"
            print(f"  {name}: NOT RUN")
    
    # Decision
    pass_count = sum(1 for r in results.values() if r == "PASS")
    fail_count = sum(1 for r in results.values() if r == "FAIL")
    
    print("\n" + "=" * 70)
    print("DECISION")
    print("=" * 70)
    print(f"PASS: {pass_count}/3")
    print(f"FAIL: {fail_count}/3")
    
    if pass_count == 0:
        print("\n>>> STOP: All pre-mortems failed.")
        print(">>> Activation-based approaches are not viable for small LMs.")
        print(">>> Consider: larger models, different tasks, or abandon direction.")
        recommendation = "STOP"
    elif pass_count == 1:
        passing = [k for k, v in results.items() if v == "PASS"][0]
        print(f"\n>>> PROCEED WITH CAUTION: Only {passing} passed.")
        print(">>> Focus on this single direction.")
        recommendation = f"SINGLE: {passing}"
    elif pass_count == 2:
        passing = [k for k, v in results.items() if v == "PASS"]
        print(f"\n>>> GOOD: Two pre-mortems passed: {', '.join(passing)}")
        print(">>> Proceed with both directions.")
        recommendation = f"DUAL: {', '.join(passing)}"
    else:
        print("\n>>> EXCELLENT: All pre-mortems passed!")
        print(">>> Proceed with full research program.")
        recommendation = "FULL"
    
    # Save summary
    summary = {
        "date": datetime.now().isoformat(),
        "results": results,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "recommendation": recommendation
    }
    
    with open("results/pre_mortems/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to results/pre_mortems/summary.json")
    print("=" * 70)
    
    return pass_count


if __name__ == "__main__":
    main()
