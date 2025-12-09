import torch
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite
import os
import sys
import json
import numpy as np

# Ensure project root is in path
sys.path.append(os.getcwd())

def sweep_layers(model_name):
    print(f"\n{'='*60}")
    print(f"STARTING LAYER SWEEP FOR: {model_name}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = PretrainedTransformer(model_name=model_name, device=device)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None

    # Inject Model into Suite
    suite = AdvancedValidationSuite(model_path=None)
    suite.model = model
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    num_layers = model.num_layers
    print(f"Model has {num_layers} layers.")

    sweep_results = []

    for layer_idx in range(num_layers):
        print(f"\n>>> Sweeping Layer {layer_idx}/{num_layers - 1}")

        # CLEAR ALL HOOKS FROM PREVIOUS ITERATIONS
        suite.model.intervention_hooks.clear()

        # Configure suite to use this layer
        suite.intervention_layer = layer_idx

        # Run reduced validation (fewer trials for speed, but enough for signal)
        # We focus on Synthetic Standard accuracy as the primary metric for EAS efficacy
        stats = suite.run_multiple_trials(num_trials=2)

        # Extract key metrics
        # We want the "EAS_Standard" on "complex_synthetic" accuracy
        eas_acc = 0.0
        baseline_acc = 0.0

        for s in stats:
            if s['scenario'] == 'EAS_Standard' and s['dataset'] == 'complex_synthetic':
                eas_acc = s['mean_accuracy']
            if s['scenario'] == 'Baseline' and s['dataset'] == 'complex_synthetic':
                baseline_acc = s['mean_accuracy']

        improvement = eas_acc - baseline_acc

        print(f"Layer {layer_idx} Result: Base={baseline_acc:.4f}, EAS={eas_acc:.4f}, Delta={improvement:+.4f}")

        sweep_results.append({
            "layer_idx": layer_idx,
            "baseline_accuracy": baseline_acc,
            "eas_accuracy": eas_acc,
            "improvement": improvement
        })

    return sweep_results

def main():
    # Priority: GPT-2 as per TODO
    model_name = "openai-community/gpt2"

    results = sweep_layers(model_name)

    if results:
        output_file = "eas/advanced_validation/results/gpt2_layer_sweep.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSweep results saved to {output_file}")

        # Find best layer
        best_layer = max(results, key=lambda x: x['improvement'])
        print(f"\nBest Layer: {best_layer['layer_idx']} (Improvement: {best_layer['improvement']:+.4f})")

if __name__ == "__main__":
    main()
