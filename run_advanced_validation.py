from eas.advanced_validation.suite import AdvancedValidationSuite
from eas.advanced_validation.analysis import run_analysis
import os
import sys

def main():
    print("Starting Advanced Validation Run...")

    model_path = "eas/advanced_validation/models/baseline_model.pt"
    if not os.path.exists(model_path):
        print("Warning: Baseline model not found. Running with random weights (expect 0% accuracy).")
        model_path = None
    else:
        print(f"Loading trained baseline model from {model_path}")

    # 1. Run the Suite
    suite = AdvancedValidationSuite(model_path=model_path)

    # Baseline
    suite.run_scenario("Baseline", "complex_synthetic", intervention_type="none", num_samples=50)
    suite.run_scenario("Baseline", "avicenna", intervention_type="none", num_samples=30)

    # EAS Standard
    suite.run_scenario("EAS_Standard", "complex_synthetic", intervention_type="standard", num_samples=50)
    suite.run_scenario("EAS_Standard", "avicenna", intervention_type="standard", num_samples=30)

    # EAS Adversarial
    suite.run_scenario("EAS_Adversarial", "complex_synthetic", intervention_type="adversarial", num_samples=50)

    suite.save_results()

    # 2. Run Analysis
    run_analysis()

    print("Validation Complete.")

if __name__ == "__main__":
    main()
