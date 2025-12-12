#!/usr/bin/env python3
"""
context_aligned_demo.py
Demonstrates Context-Aligned Raw EAS on GPT-2.
Target: Validate +16% improvement claim.
"""

import sys
import os
import torch
import numpy as np

# Add archive to path to access eas package
sys.path.append(os.path.join(os.getcwd(), 'archive'))

from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite

def run_demo():
    print("="*60)
    print("CONTEXT-ALIGNED RAW EAS DEMO")
    print("Target: Demonstrate validity and benefit on GPT-2")
    print("="*60)

    model_name = "gpt2" # Small GPT-2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {model_name} on {device}...")

    try:
        model = PretrainedTransformer(model_name=model_name, device=device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize Suite with RAW EAS (no whitening)
    print("Initializing Advanced Validation Suite (Mode: Raw EAS)...")
    suite = AdvancedValidationSuite(
        transductive=False,
        use_whitening=False # Raw EAS
    )

    # Inject model
    suite.model = model
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    # OVERRIDE: Intervention Layer 2 (as per memory for GPT-2)
    suite.default_intervention_layer = 2
    suite.warmup_size = 50

    print("\nRunning Evaluation with Layer 2 Intervention...")

    # --- SCENARIO 1: Synthetic Logic ---
    print("\n--- SCENARIO 1: SYNTHETIC LOGIC ---")

    # Baseline
    res_base = suite.run_scenario("Baseline", "complex_synthetic", "none", num_samples=50)
    print(f"Baseline Accuracy: {res_base['accuracy']:.2%}")

    # EAS
    suite.reset_watcher()
    suite.default_intervention_layer = 2 # Ensure it persists after reset
    suite.warmup_watcher(num_samples=50, dataset_type="synthetic")
    res_eas = suite.run_scenario("Context-Aligned EAS", "complex_synthetic", "standard", num_samples=50)
    print(f"EAS Accuracy: {res_eas['accuracy']:.2%}")

    delta_syn = res_eas['accuracy'] - res_base['accuracy']
    print(f"Delta: {delta_syn:+.2%}")

    # --- SCENARIO 2: Avicenna (Real World) ---
    print("\n--- SCENARIO 2: AVICENNA (Real NLI) ---")

    # Baseline
    res_base_avi = suite.run_scenario("Baseline", "avicenna", "none", num_samples=30)
    print(f"Baseline Accuracy: {res_base_avi['accuracy']:.2%}")

    # EAS (Warmup on Synthetic + small Avicenna shot)
    suite.reset_watcher()
    suite.default_intervention_layer = 2
    suite.warmup_watcher(num_samples=50, dataset_type="synthetic")
    suite.warmup_watcher(num_samples=10, dataset_type="avicenna") # Few-shot adaptation

    res_eas_avi = suite.run_scenario("Context-Aligned EAS", "avicenna", "standard", num_samples=30)
    print(f"EAS Accuracy: {res_eas_avi['accuracy']:.2%}")

    delta_avi = res_eas_avi['accuracy'] - res_base_avi['accuracy']
    print(f"Delta: {delta_avi:+.2%}")

    # Summary
    print("\n" + "="*60)
    print(f"FINAL RESULTS SUMMARY for {model_name} (Layer 2)")
    print("="*60)
    print(f"Synthetic Delta: {delta_syn:+.2%}")
    print(f"Avicenna Delta:  {delta_avi:+.2%}")
    print("="*60)

if __name__ == "__main__":
    run_demo()
