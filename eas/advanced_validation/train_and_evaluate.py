import torch
from eas.src.models.transformer import PretrainedTransformer
from eas.advanced_validation.suite import AdvancedValidationSuite
from eas.advanced_validation.analysis import run_analysis
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

def train_and_evaluate():
    print("Initializing Pre-trained Model for Validation...")

    # 1. Initialize Model (Pre-trained)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainedTransformer(model_name="EleutherAI/pythia-70m", device=device)

    # 2. Skip Training (It's pre-trained!)
    print("Model loaded (Frozen). Skipping training.")

    # 3. Inject Model into Suite (In-Memory)
    # The suite usually instantiates its own model. We need to override it.
    suite = AdvancedValidationSuite(model_path=None)
    suite.model = model
    # Also need to make sure suite uses the correct tokenizer!
    suite.tokenizer = model.tokenizer
    suite.is_pretrained = True

    # 4. Run Scenarios
    # Baseline
    print("\nRunning Baseline Scenarios...")
    suite.run_scenario("Baseline", "complex_synthetic", intervention_type="none", num_samples=30)
    suite.run_scenario("Baseline", "avicenna", intervention_type="none", num_samples=20)

    # EAS Standard
    print("\nRunning EAS Standard Scenarios...")
    suite.run_scenario("EAS_Standard", "complex_synthetic", intervention_type="standard", num_samples=30)
    suite.run_scenario("EAS_Standard", "avicenna", intervention_type="standard", num_samples=20)

    # EAS Adversarial
    print("\nRunning EAS Adversarial Scenarios...")
    suite.run_scenario("EAS_Adversarial", "complex_synthetic", intervention_type="adversarial", num_samples=30)

    print("\nSaving results...")
    suite.save_results()
    print("Running analysis...")
    run_analysis()

if __name__ == "__main__":
    train_and_evaluate()
