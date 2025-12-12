import torch
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite
import os
import sys
import json

# Ensure project root is in path
sys.path.append(os.getcwd())

def test_model(model_name, label):
    print(f"\n{'='*60}")
    print(f"TESTING MODEL: {model_name} ({label})")
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

    # Run Validation
    stats = suite.run_multiple_trials(num_trials=2) # Reduced to 2 for speed with larger models
    return stats

def main():
    models = [
        ("EleutherAI/pythia-160m", "Scaling Test (NeoX)"),
        ("facebook/opt-125m", "Architecture Test (OPT)"),
        ("bigscience/bloom-560m", "Architecture Test (Bloom/ALiBi)"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Architecture Test (Llama/RoPE)")
    ]

    all_results = {}

    for model_name, label in models:
        stats = test_model(model_name, label)
        if stats:
            all_results[model_name] = stats

    # Save results
    os.makedirs("eas/advanced_validation/results", exist_ok=True)
    with open("eas/advanced_validation/results/zoo_sweep.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nZoo sweep complete. Results saved.")

if __name__ == "__main__":
    main()
