from eas.advanced_validation.suite import AdvancedValidationSuite
from eas.advanced_validation.analysis import run_analysis
import os

def main():
    print("Starting Advanced Validation Run...")

    # 1. Run the Suite
    suite = AdvancedValidationSuite()

    # Baseline
    suite.run_scenario("Baseline", "complex_synthetic", intervention_type="none", num_samples=20)
    suite.run_scenario("Baseline", "avicenna", intervention_type="none", num_samples=20)

    # EAS Standard
    suite.run_scenario("EAS_Standard", "complex_synthetic", intervention_type="standard", num_samples=20)
    suite.run_scenario("EAS_Standard", "avicenna", intervention_type="standard", num_samples=20)

    # EAS Adversarial
    suite.run_scenario("EAS_Adversarial", "complex_synthetic", intervention_type="adversarial", num_samples=20)

    suite.save_results()

    # 2. Run Analysis
    run_analysis()

    print("Validation Complete.")

if __name__ == "__main__":
    main()
