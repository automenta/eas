import torch
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite
import os
import sys
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

def evaluate_model(model_name):
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
    suite = AdvancedValidationSuite(model_path=None)
    suite.model = model
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    # Run Rigorous Multi-Trial Validation (Reduced to 3 trials for speed in multi-model context)
    stats = suite.run_multiple_trials(num_trials=3)
    return stats

def main():
    models_to_test = ["EleutherAI/pythia-70m", "openai-community/gpt2"]

    all_results = {}

    for model_name in models_to_test:
        stats = evaluate_model(model_name)
        if stats:
            all_results[model_name] = stats

    # Save consolidated results
    with open("eas/advanced_validation/results/multi_model_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown Report
    generate_markdown_report(all_results)

def generate_markdown_report(all_results):
    report = "# EAS Multi-Model Validation Report\n\n"

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

    with open("VALIDATION_REPORT.md", "w") as f:
        f.write(report)
    print("\nReport saved to VALIDATION_REPORT.md")

if __name__ == "__main__":
    main()
