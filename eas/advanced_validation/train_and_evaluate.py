import torch
import argparse
import os
import sys
import json
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite

# Ensure project root is in path
sys.path.append(os.getcwd())

def evaluate_model(model_name, intervention_layer=None, watcher_alpha=None, watcher_k=None, warmup_size=None, transductive=False, use_whitening=True):
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
    suite = AdvancedValidationSuite(model_path=None, transductive=transductive, use_whitening=use_whitening)
    suite.model = model
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    # Apply Configuration Overrides if provided
    if intervention_layer is not None:
        print(f"Overriding intervention_layer to {intervention_layer}")
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

    # Run Rigorous Multi-Trial Validation
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
    parser.add_argument("--no_whitening", action="store_true", help="Disable whitening (Raw EAS Mode)")
    parser.add_argument("--report_file", type=str, default="VALIDATION_REPORT.md", help="Output markdown report file")

    args = parser.parse_args()

    use_whitening = not args.no_whitening

    all_results = {}
    stats = evaluate_model(
        args.model_name,
        intervention_layer=args.layer,
        watcher_alpha=args.alpha,
        watcher_k=args.k,
        warmup_size=args.warmup,
        transductive=args.transductive,
        use_whitening=use_whitening
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
