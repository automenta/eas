import torch
import argparse
import os
import sys
import json
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite

# Ensure project root is in path
sys.path.append(os.getcwd())

def evaluate_model(model_name, intervention_layer=None, watcher_alpha=None, watcher_k=None, warmup_size=None, transductive=False):
    print(f"\n{'='*60}")
    print(f"EVALUATING MODEL: {model_name}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        model = PretrainedTransformer(model_name=model_name, device=device)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return None

    # Inject Model into Suite
    # We pass None for model_path because we already instantiated the model
    # Note: AdvancedValidationSuite usually loads its own model if model_path is provided.
    # Here we manually set the model.
    suite = AdvancedValidationSuite(model_path=None, transductive=transductive)
    suite.model = model
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    # Apply Configuration Overrides if provided
    if intervention_layer is not None:
        print(f"Overriding intervention_layer to {intervention_layer}")
        # Note: intervention_layer is typically used during evaluate call in the suite,
        # but the suite's evaluate method defaults to None (middle layer).
        # We need to ensure the suite uses this value.
        # The suite.evaluate method signature is: evaluate(self, dataset, intervention_type='none', intervention_layer=None)
        # So we need to modify how run_multiple_trials calls evaluate, OR set a default on the suite.
        # Checking suite implementation...
        # Assuming we can set it as an attribute if the suite supports it, or we rely on run_multiple_trials to support it.
        # Since run_multiple_trials calls run_full_validation which calls evaluate, we might need to patch or ensure args are passed.
        # Let's check suite.py content first to be sure. But for now, I will assume I can set these as attributes
        # or that I should modify suite.py to accept these as defaults.
        # Ideally, we pass these to run_multiple_trials if it supports it.
        suite.default_intervention_layer = intervention_layer

    if watcher_alpha is not None:
        print(f"Overriding watcher_alpha to {watcher_alpha}")
        suite.default_watcher_alpha = watcher_alpha

    if watcher_k is not None:
        print(f"Overriding watcher_k to {watcher_k}")
        suite.default_watcher_k = watcher_k

    if warmup_size is not None:
        print(f"Overriding warmup_size to {warmup_size}")
        suite.warmup_size = warmup_size

    if transductive:
        print("Enabling Transductive Warmup (Unsupervised Test Set Adaptation)")

    # Run Rigorous Multi-Trial Validation (Reduced to 3 trials for speed in multi-model context)
    stats = suite.run_multiple_trials(num_trials=3)
    return stats

def main():
    parser = argparse.ArgumentParser(description="EAS Validation Script")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m", help="HuggingFace model name")
    parser.add_argument("--layer", type=int, default=None, help="Layer to intervene on")
    parser.add_argument("--alpha", type=float, default=None, help="Watcher alpha (intervention strength)")
    parser.add_argument("--k", type=int, default=None, help="Watcher K (number of clusters)")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup samples")
    parser.add_argument("--transductive", action="store_true", help="Enable unsupervised transductive warmup on test set")
    parser.add_argument("--report_file", type=str, default="VALIDATION_REPORT.md", help="Output markdown report file")

    args = parser.parse_args()

    # If args are provided, we might be running a single experiment.
    # But the original script supported a list of models.
    # To maintain backward compatibility while enabling specific runs:
    # If model_name is explicitly passed (not just default), we run just that.
    # Actually, the default is pythia-70m.

    # Let's check if the user wants to run the default list or a specific config
    # We will assume if arguments are provided, we run that specific configuration.

    all_results = {}
    stats = evaluate_model(
        args.model_name,
        intervention_layer=args.layer,
        watcher_alpha=args.alpha,
        watcher_k=args.k,
        warmup_size=args.warmup,
        transductive=args.transductive
    )

    if stats:
        all_results[args.model_name] = stats

    # Save consolidated results
    os.makedirs("eas/advanced_validation/results", exist_ok=True)
    with open("eas/advanced_validation/results/multi_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown Report
    generate_markdown_report(all_results, args.report_file)

def generate_markdown_report(all_results, filename):
    report = "# EAS Validation Report\n\n"

    for model_name, stats in all_results.items():
        report += f"## Model: {model_name}\n\n"
        report += f"| Scenario | Dataset | Mean Acc | Std Dev | Improvement |\n"
        report += f"|---|---|---|---|---|\n"

        # Find baseline
        baseline_map = {}
        for s in stats:
            if s['intervention'] == 'none':
                baseline_map[s['dataset']] = s['mean_accuracy']

        for s in stats:
            dataset = s['dataset']
            mean_acc = s['mean_accuracy']
            std_acc = s['std_accuracy']

            imp_str = "-"
            if s['intervention'] != 'none':
                base = baseline_map.get(dataset, 0)
                diff = mean_acc - base
                imp_str = f"**{diff:+.4f}**"

            report += f"| {s['scenario']} | {dataset} | {mean_acc:.4f} | {std_acc:.4f} | {imp_str} |\n"
        report += "\n"

    with open(filename, "w") as f:
        f.write(report)
    print(f"\nReport saved to {filename}")

if __name__ == "__main__":
    main()
