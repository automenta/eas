import json
import os
import numpy as np

def run_analysis(results_dir="eas/advanced_validation/results"):
    with open(os.path.join(results_dir, "validation_results.json"), 'r') as f:
        results = json.load(f)

    report = "# EAS Advanced Validation Report\n\n"

    # 1. Overall Performance Analysis
    report += "## 1. Overall Performance\n"
    report += "| Scenario | Dataset | Intervention | Accuracy | Latency (s) |\n"
    report += "|---|---|---|---|---|\n"

    scenarios = {}

    for r in results:
        key = (r['dataset'], r['intervention'])
        scenarios[key] = r['accuracy']
        report += f"| {r['scenario']} | {r['dataset']} | {r['intervention']} | {r['accuracy']:.4f} | {r['latency']:.4f} |\n"

    report += "\n\n"

    # 2. Honest Assessment of Effectiveness
    report += "## 2. Honest Assessment of Effectiveness\n"

    # Comparison: Baseline vs Standard
    # Check Complex Synthetic
    base_synth = scenarios.get(('complex_synthetic', 'none'), 0)
    eas_synth = scenarios.get(('complex_synthetic', 'standard'), 0)
    delta_synth = eas_synth - base_synth

    # Check Avicenna
    base_avi = scenarios.get(('avicenna', 'none'), 0)
    eas_avi = scenarios.get(('avicenna', 'standard'), 0)
    delta_avi = eas_avi - base_avi

    report += f"### Impact on Complex Synthetic Logic\n"
    report += f"- Baseline Accuracy: {base_synth:.2%}\n"
    report += f"- EAS Accuracy: {eas_synth:.2%}\n"
    report += f"- Improvement: {delta_synth:+.2%}\n"

    if abs(delta_synth) < 0.05:
        report += "**Assessment:** No significant impact. The model performance is dominated by base capabilities (or lack thereof).\n"
    elif delta_synth > 0.05:
        report += "**Assessment:** Positive impact detected. EAS successfully guided the model.\n"
    else:
        report += "**Assessment:** Negative impact. EAS interfered with reasoning.\n"

    report += "\n### Impact on Real-World Data (Avicenna)\n"
    report += f"- Baseline Accuracy: {base_avi:.2%}\n"
    report += f"- EAS Accuracy: {eas_avi:.2%}\n"
    report += f"- Improvement: {delta_avi:+.2%}\n"

    if abs(delta_avi) < 0.05:
        report += "**Assessment:** No significant impact on real data. "
        if base_avi < 0.1:
            report += "This is likely due to the 'Cold Start' problem: the base model is too weak to form attractors.\n"
        else:
            report += "This suggests EAS does not generalize to unstructured inputs.\n"

    # 3. Robustness Analysis
    report += "\n## 3. Robustness & Adversarial Analysis\n"
    adv_synth = scenarios.get(('complex_synthetic', 'adversarial'), 0)
    report += f"- Adversarial Accuracy: {adv_synth:.2%}\n"

    if adv_synth < eas_synth - 0.1:
        report += "**Observation:** Significant degradation under adversarial conditions (distractors). EAS failed to filter noise.\n"
    else:
        report += "**Observation:** Robust performance maintained (or equally poor).\n"

    # 4. Introspection & Homogeneity
    report += "\n## 4. Addressing Homogeneity Concerns\n"
    if abs(base_synth - eas_synth) < 0.01 and abs(base_avi - eas_avi) < 0.01:
        report += "The results show extreme homogeneity (no change). This confirms the user's concern that previous 'variations' were likely artifacts or simulations. "
        report += "In this rigorous test with a frozen, random/weak base model, EAS shows its true limitation: it cannot create logic ex nihilo.\n"
    else:
        report += "We observed distinct performance profiles, suggesting the validation framework successfully captured variance in model behavior.\n"

    with open("VALIDATION_REPORT.md", 'w') as f:
        f.write(report)

    print("Report generated: VALIDATION_REPORT.md")

if __name__ == "__main__":
    run_analysis()
