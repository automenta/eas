#!/usr/bin/env python3
"""
Comprehensive EAS Validation Suite
Generates detailed report with explanations, data, charts, and statistical validation
"""
import os
import sys
import json
import time
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats
import pandas as pd

# Handle matplotlib import gracefully
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available, plots will be skipped")

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from eas.src.models.tokenizer import create_small_tokenizer
from eas.src.models.transformer import create_small_model
from eas.src.watcher import EmergentWatcher
from eas.src.datasets import LogicCorpusGenerator
from eas.src.experiments import EASEvaluator
from eas.src.experiments.baselines import (
    BaseEvaluator, RandomControlEvaluator, 
    FixedSteeringEvaluator, NoClampingEvaluator
)


class ComprehensiveEASValidator:
    """Comprehensive validator for EAS with detailed reporting and statistical validation"""
    
    def __init__(self, output_dir: str = "comprehensive_validation"):
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.plots_dir = os.path.join(output_dir, "plots")
        self.reports_dir = os.path.join(output_dir, "reports")
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Set reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def log_progress(self, message: str):
        """Log progress with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run_statistical_validation(self, baseline_accs: List[float], eas_accs: List[float]) -> Dict[str, float]:
        """Run statistical validation to confirm significance of improvements"""
        self.log_progress("Running statistical validation...")
        
        # Convert to numpy arrays for analysis
        baseline_array = np.array(baseline_accs)
        eas_array = np.array(eas_accs)
        
        # Perform paired t-test to check if improvement is statistically significant
        t_stat, p_value = stats.ttest_rel(eas_array, baseline_array)
        
        # Calculate effect size (Cohen's d)
        mean_diff = np.mean(eas_array - baseline_array)
        pooled_std = np.sqrt(((len(baseline_array)-1)*np.var(baseline_array) + 
                             (len(eas_array)-1)*np.var(eas_array)) / 
                            (len(baseline_array) + len(eas_array) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        
        # Calculate confidence intervals
        confidence_level = 0.95
        alpha = 1 - confidence_level
        baseline_ci = stats.t.interval(confidence_level, len(baseline_array)-1, 
                                      loc=np.mean(baseline_array), 
                                      scale=stats.sem(baseline_array))
        eas_ci = stats.t.interval(confidence_level, len(eas_array)-1,
                                 loc=np.mean(eas_array),
                                 scale=stats.sem(eas_array))
        
        improvement_ci = stats.t.interval(confidence_level, len(eas_array-baseline_array)-1,
                                         loc=np.mean(eas_array - baseline_array),
                                         scale=stats.sem(eas_array - baseline_array))
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'statistically_significant': float(p_value) < 0.05,
            'baseline_mean': float(np.mean(baseline_array)),
            'baseline_ci': [float(baseline_ci[0]), float(baseline_ci[1])],
            'eas_mean': float(np.mean(eas_array)),
            'eas_ci': [float(eas_ci[0]), float(eas_ci[1])],
            'improvement_mean': float(np.mean(eas_array - baseline_array)),
            'improvement_ci': [float(improvement_ci[0]), float(improvement_ci[1])],
            'baseline_std': float(np.std(baseline_array)),
            'eas_std': float(np.std(eas_array))
        }
    
    def run_multiple_trials(self, num_trials: int = 10) -> Dict[str, List[float]]:
        """Run multiple trials to get reliable statistics"""
        self.log_progress(f"Running {num_trials} trials for robust statistics...")
        
        baseline_accuracies = []
        eas_accuracies = []
        
        generator = LogicCorpusGenerator()
        
        for trial in range(num_trials):
            self.log_progress(f"Running trial {trial + 1}/{num_trials}...")
            
            # Create new random seed for each trial to get different data
            torch.manual_seed(42 + trial)
            random.seed(42 + trial)
            np.random.seed(42 + trial)

            # We also need to reset the model for each trial to avoid identical results
            # Recreate the model and tokenizer to ensure fresh initialization
            
            # Create new dataset for each trial
            dataset = [generator.generate_challenging_sample() for _ in range(30)]
            
            # Create model and tokenizer
            tokenizer = create_small_tokenizer(vocab_size=200)
            model = create_small_model(vocab_size=tokenizer.get_vocab_size())
            model.to(self.device)
            
            # Train model
            evaluator = EASEvaluator(model, tokenizer, device=self.device)
            evaluator.train_base_model(dataset, epochs=1)
            
            # Run baseline evaluation
            baseline_eval = BaseEvaluator(model, tokenizer, device=self.device)
            baseline_eval.model_trained = True
            baseline_results = baseline_eval.evaluate_baseline(dataset, num_iterations=50)
            baseline_acc = baseline_results['accuracy'][-1] if baseline_results['accuracy'] else 0
            baseline_accuracies.append(baseline_acc)
            
            # Run EAS evaluation
            watcher = EmergentWatcher(
                dim=128,
                k=5,
                alpha_base=0.3,
                max_delta=0.3,
                update_frequency=3
            ).to(self.device)
            eas_eval = EASEvaluator(model, tokenizer, watcher, device=self.device)
            eas_eval.model_trained = True
            eas_results = eas_eval.evaluate_with_eas(dataset, num_iterations=50)
            eas_acc = eas_results['accuracy'][-1] if eas_results['accuracy'] else 0
            eas_accuracies.append(eas_acc)
            
            self.log_progress(f"  Trial {trial + 1}: Baseline: {baseline_acc:.4f}, EAS: {eas_acc:.4f}, Diff: {eas_acc - baseline_acc:.4f}")
        
        return {
            'baseline_accuracies': baseline_accuracies,
            'eas_accuracies': eas_accuracies
        }
    
    def run_comprehensive_baselines(self) -> Dict[str, List[float]]:
        """Run comprehensive baseline comparisons"""
        self.log_progress("Running comprehensive baseline comparisons...")
        
        generator = LogicCorpusGenerator()
        dataset = [generator.generate_challenging_sample() for _ in range(30)]
        
        # Create model and tokenizer
        tokenizer = create_small_tokenizer(vocab_size=200)
        model = create_small_model(vocab_size=tokenizer.get_vocab_size())
        model.to(self.device)
        
        # Train model
        evaluator = EASEvaluator(model, tokenizer, device=self.device)
        evaluator.train_base_model(dataset, epochs=1)
        
        results = {}
        
        # Baseline: No Watcher
        baseline_eval = BaseEvaluator(model, tokenizer, device=self.device)
        baseline_eval.model_trained = True
        baseline_results = baseline_eval.evaluate_baseline(dataset, num_iterations=50)
        results['baseline'] = [baseline_results['accuracy'][-1] if baseline_results['accuracy'] else 0]
        
        # EAS
        watcher = EmergentWatcher(dim=128, k=5, alpha_base=0.3, max_delta=0.3, update_frequency=3).to(self.device)
        eas_eval = EASEvaluator(model, tokenizer, watcher, device=self.device)
        eas_eval.model_trained = True
        eas_results = eas_eval.evaluate_with_eas(dataset, num_iterations=50)
        results['eas'] = [eas_results['accuracy'][-1] if eas_results['accuracy'] else 0]
        
        # Random Control
        watcher_rc = EmergentWatcher(dim=128, k=5, alpha_base=0.3, max_delta=0.3, update_frequency=3).to(self.device)
        rc_eval = RandomControlEvaluator(model, tokenizer, watcher_rc, device=self.device)
        rc_eval.model_trained = True
        rc_results = rc_eval.evaluate_random_control(dataset, num_iterations=50)
        results['random_control'] = [rc_results['accuracy'][-1] if rc_results['accuracy'] else 0]
        
        # Fixed Steering
        fs_eval = FixedSteeringEvaluator(model, tokenizer, fixed_alpha=0.3, device=self.device)
        fs_eval.model_trained = True
        fs_results = fs_eval.evaluate_fixed_steering(dataset, num_iterations=50)
        results['fixed_steering'] = [fs_results['accuracy'][-1] if fs_results['accuracy'] else 0]
        
        # No Clamping
        watcher_nc = EmergentWatcher(dim=128, k=5, alpha_base=0.3, max_delta=0.3, update_frequency=3).to(self.device)
        nc_eval = NoClampingEvaluator(model, tokenizer, watcher_nc, device=self.device)
        nc_eval.model_trained = True
        nc_results = nc_eval.evaluate_no_clamping(dataset, num_iterations=50)
        results['no_clamping'] = [nc_results['accuracy'][-1] if nc_results['accuracy'] else 0]
        
        return results
    
    def generate_plots(self, baseline_accuracies: List[float], eas_accuracies: List[float],
                      baseline_comparisons: Dict[str, List[float]]):
        """Generate comprehensive plots if matplotlib is available"""
        if not MATPLOTLIB_AVAILABLE:
            self.log_progress("Matplotlib not available, skipping plots")
            return

        self.log_progress("Generating comprehensive plots...")

        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive EAS Validation Results', fontsize=16, fontweight='bold')

        # Plot 1: Box plot comparison
        data_for_box = [baseline_accuracies, eas_accuracies]
        labels = ['Baseline', 'EAS']
        axes[0, 0].boxplot(data_for_box, labels=labels)
        axes[0, 0].set_title('Accuracy Distribution Comparison')
        axes[0, 0].set_ylabel('Accuracy')

        # Plot 2: Individual trial results
        x = range(len(baseline_accuracies))
        axes[0, 1].plot(x, baseline_accuracies, 'o-', label='Baseline', alpha=0.7)
        axes[0, 1].plot(x, eas_accuracies, 's-', label='EAS', alpha=0.7)
        axes[0, 1].set_title('Trial-by-Trial Performance')
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Improvement distribution
        improvements = [e - b for e, b in zip(eas_accuracies, baseline_accuracies)]
        axes[0, 2].hist(improvements, bins=10, edgecolor='black', alpha=0.7)
        axes[0, 2].set_title(f'Distribution of Improvements\n(Mean: {np.mean(improvements):.3f})')
        axes[0, 2].set_xlabel('EAS Improvement over Baseline')
        axes[0, 2].set_ylabel('Frequency')

        # Plot 4: Comprehensive baseline comparison
        baseline_names = list(baseline_comparisons.keys())
        baseline_means = [np.mean(vals) for vals in baseline_comparisons.values()]
        axes[1, 0].bar(baseline_names, baseline_means, alpha=0.7, color=plt.cm.Set3(range(len(baseline_names))))
        axes[1, 0].set_title('Comprehensive Baseline Comparison')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Plot 5: Performance improvement with error bars
        baseline_mean = np.mean(baseline_accuracies)
        eas_mean = np.mean(eas_accuracies)
        baseline_sem = stats.sem(baseline_accuracies)
        eas_sem = stats.sem(eas_accuracies)

        x_pos = [0, 1]
        means = [baseline_mean, eas_mean]
        errors = [baseline_sem, eas_sem]
        names = ['Baseline', 'EAS']

        axes[1, 1].bar(x_pos, means, yerr=errors, capsize=5, alpha=0.7,
                      color=['lightcoral', 'lightblue'], edgecolor='black')
        axes[1, 1].set_title('Mean Performance with Error Bars')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(names)

        # Add value labels on bars
        for i, (mean, err) in enumerate(zip(means, errors)):
            axes[1, 1].text(i, mean + err + 0.01, f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

        # Plot 6: Improvement over baseline
        improvement_mean = eas_mean - baseline_mean
        improvement_error = np.sqrt(baseline_sem**2 + eas_sem**2)

        axes[1, 2].bar(['Improvement'], [improvement_mean], yerr=[improvement_error],
                      capsize=10, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 2].set_title('EAS Improvement over Baseline')
        axes[1, 2].set_ylabel('Improvement in Accuracy')
        axes[1, 2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].text(0, improvement_mean + improvement_error + 0.005,
                       f'{improvement_mean:.3f}', ha='center', va='bottom', fontweight='bold')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save the plot
        plot_path = os.path.join(self.plots_dir, "comprehensive_validation_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.log_progress(f"Plots saved to: {plot_path}")
    
    def run_complete_validation(self) -> Dict[str, any]:
        """Run the complete validation suite"""
        start_time = time.time()
        
        self.log_progress("Starting Comprehensive EAS Validation Suite")
        
        # 1. Run multiple trials for robust statistics
        trial_results = self.run_multiple_trials(num_trials=15)  # More trials for robustness
        
        # 2. Run comprehensive baselines
        baseline_results = self.run_comprehensive_baselines()
        
        # 3. Run statistical validation
        stats_results = self.run_statistical_validation(
            trial_results['baseline_accuracies'], 
            trial_results['eas_accuracies']
        )
        
        # 4. Generate plots
        self.generate_plots(
            trial_results['baseline_accuracies'],
            trial_results['eas_accuracies'],
            baseline_results
        )
        
        # 5. Compile comprehensive report
        comprehensive_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_runtime': time.time() - start_time,
                'device': str(self.device),
                'num_trials': len(trial_results['baseline_accuracies']),
                'trial_seeds_used': list(range(42, 42 + len(trial_results['baseline_accuracies'])))
            },
            'performance_results': {
                'baseline_accuracies': [float(x) for x in trial_results['baseline_accuracies']],
                'eas_accuracies': [float(x) for x in trial_results['eas_accuracies']],
                'improvements': [float(e - b) for e, b in zip(trial_results['eas_accuracies'], 
                                                              trial_results['baseline_accuracies'])]
            },
            'statistical_validation': stats_results,
            'baseline_comparisons': {k: [float(x) for x in v] for k, v in baseline_results.items()},
            'summary_metrics': {
                'baseline_mean_accuracy': float(np.mean(trial_results['baseline_accuracies'])),
                'eas_mean_accuracy': float(np.mean(trial_results['eas_accuracies'])),
                'mean_improvement': float(np.mean([e - b for e, b in zip(
                    trial_results['eas_accuracies'], trial_results['baseline_accuracies'])])),
                'improvement_percentage': float(np.mean([e - b for e, b in zip(
                    trial_results['eas_accuracies'], trial_results['baseline_accuracies'])]) / 
                    np.mean(trial_results['baseline_accuracies']) * 100),
                'statistical_significance': stats_results['statistically_significant'],
                'effect_size': stats_results['cohens_d'],
                'confidence_level': 0.95
            }
        }
        
        # 6. Save results
        results_path = os.path.join(self.results_dir, "comprehensive_validation_results.json")
        with open(results_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        self.log_progress(f"Comprehensive validation results saved to: {results_path}")
        self.log_progress("Comprehensive EAS Validation Suite Completed")
        
        return comprehensive_results


def generate_detailed_report(results: Dict[str, any], output_dir: str):
    """Generate a detailed markdown report for engineers"""
    report_path = os.path.join(output_dir, "reports", "detailed_eas_validation_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# Comprehensive EAS Validation Report\n\n")
        f.write(f"**Generated:** {results['metadata']['timestamp']}\n")
        f.write(f"**Runtime:** {results['metadata']['total_runtime']:.2f} seconds\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This report presents comprehensive validation of the Emergent Activation Snapping (EAS) approach.\n")
        f.write("The validation includes multiple trials, statistical analysis, and comparison against multiple baselines.\n\n")
        
        summary = results['summary_metrics']
        f.write(f"- **Baseline Mean Accuracy:** {summary['baseline_mean_accuracy']:.4f}\n")
        f.write(f"- **EAS Mean Accuracy:** {summary['eas_mean_accuracy']:.4f}\n") 
        f.write(f"- **Mean Improvement:** {summary['mean_improvement']:.4f} ({summary['improvement_percentage']:.2f}%)\n")
        f.write(f"- **Statistical Significance:** {'Yes' if summary['statistical_significance'] else 'No'}\n")
        f.write(f"- **Effect Size (Cohen\'s d):** {summary['effect_size']:.4f}\n\n")
        
        f.write("## Statistical Analysis\n\n")
        stats = results['statistical_validation']
        f.write(f"- **T-statistic:** {stats['t_statistic']:.4f}\n")
        f.write(f"- **P-value:** {stats['p_value']:.6f} ({'Significant' if stats['statistically_significant'] else 'Not Significant'})\n")
        f.write(f"- **95% CI for Baseline:** [{stats['baseline_ci'][0]:.4f}, {stats['baseline_ci'][1]:.4f}]\n")
        f.write(f"- **95% CI for EAS:** [{stats['eas_ci'][0]:.4f}, {stats['eas_ci'][1]:.4f}]\n")
        f.write(f"- **95% CI for Improvement:** [{stats['improvement_ci'][0]:.4f}, {stats['improvement_ci'][1]:.4f}]\n\n")
        
        f.write("## Methodology\n\n")
        f.write("### Experimental Design\n")
        f.write(f"- **Trials:** {results['metadata']['num_trials']} independent trials\n")
        f.write("- **Each trial uses different random seed for robustness\n")
        f.write("- **Dataset:** Randomly generated logical reasoning problems\n")
        f.write("- **Evaluation:** 50 iterations per condition per trial\n\n")
        
        f.write("### Baseline Conditions\n")
        f.write("- **Baseline:** No watcher intervention\n")
        f.write("- **Random Control:** Watcher with disabled updates\n")
        f.write("- **Fixed Steering:** Constant alpha (non-adaptive)\n")
        f.write("- **No Clamping:** Without safety constraints\n")
        f.write("- **EAS:** Full Emergent Activation Snapping approach\n\n")
        
        f.write("## Results\n\n")
        f.write("### Performance Comparison\n\n")
        f.write("| Condition | Mean Accuracy | Standard Deviation |\n")
        f.write("|-----------|---------------|-------------------|\n")
        
        baseline_comparisons = results['baseline_comparisons']
        for condition, accuracies in baseline_comparisons.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            f.write(f"| {condition.replace('_', ' ').title()} | {mean_acc:.4f} | {std_acc:.4f} |\n")
        
        f.write("\n### Trial Results\n\n")
        f.write("Individual trial results showing consistency across runs:\n\n")
        f.write("| Trial | Baseline | EAS | Improvement |\n")
        f.write("|-------|----------|-----|-------------|\n")
        
        perf_results = results['performance_results']
        for i, (baseline, eas) in enumerate(zip(perf_results['baseline_accuracies'], 
                                               perf_results['eas_accuracies']), 1):
            improvement = eas - baseline
            f.write(f"| {i} | {baseline:.4f} | {eas:.4f} | {improvement:.4f} |\n")
        
        f.write("\n## Statistical Validation\n\n")
        f.write("To ensure the observed improvement is not due to random chance, we conducted:\n\n")
        f.write("1. **Paired t-test**: Compares performance on the same test set for each trial\n")
        f.write("2. **Effect size calculation (Cohen's d)**: Measures practical significance\n") 
        f.write("3. **Confidence intervals**: Provides range of likely true values\n\n")
        
        if summary['statistical_significance']:
            f.write(f"**RESULT**: The improvement is statistically significant (p < 0.05).\n")
        else:
            f.write(f"**RESULT**: The improvement is not statistically significant (p ≥ 0.05).\n")
        
        effect_size = summary['effect_size']
        if abs(effect_size) >= 0.8:
            f.write(f"The effect size (Cohen's d = {effect_size:.3f}) indicates a **large** practical effect.\n")
        elif abs(effect_size) >= 0.5:
            f.write(f"The effect size (Cohen's d = {effect_size:.3f}) indicates a **medium** practical effect.\n")
        elif abs(effect_size) >= 0.2:
            f.write(f"The effect size (Cohen's d = {effect_size:.3f}) indicates a **small** practical effect.\n")
        else:
            f.write(f"The effect size (Cohen's d = {effect_size:.3f}) indicates a **negligible** practical effect.\n")
        
        f.write("\n## Visualization\n\n")
        if MATPLOTLIB_AVAILABLE:
            f.write("![Comprehensive Validation Results](../plots/comprehensive_validation_results.png)\n\n")
        else:
            f.write("*Plots not generated due to missing matplotlib/seaborn dependencies*\n\n")
        
        f.write("## Conclusion\n\n")
        if summary['statistical_significance'] and summary['mean_improvement'] > 0.1:
            f.write("**EAS demonstrates clear, statistically significant improvement** over baseline methods.\n")
            f.write(f"The approach shows a {summary['improvement_percentage']:.1f}% improvement with large effect size, ")
            f.write("providing strong evidence for the effectiveness of activation snapping to successful reasoning patterns.\n")
        elif summary['statistical_significance']:
            f.write("EAS shows **statistically significant but modest improvement** over baseline methods.\n")
            f.write(f"While the {summary['improvement_percentage']:.1f}% improvement is significant, ")
            f.write("the effect size suggests room for optimization.\n")
        else:
            f.write("The improvement did not reach statistical significance, suggesting\n")
            f.write("either the effect is small or additional validation is needed.\n")
        
        f.write("\n## Recommendations\n\n")
        f.write("Based on the results:\n\n")
        if summary['statistical_significance'] and summary['mean_improvement'] > 0.1:
            f.write("1. **Continue R&D on EAS**: Strong evidence supports further development\n")
            f.write("2. **Scale to larger models**: Validate effectiveness on standard architectures\n")
            f.write("3. **Optimize hyperparameters**: Fine-tune for maximum benefit\n")
        elif summary['statistical_significance']:
            f.write("1. **Refine EAS approach**: Analyze conditions where benefit is maximized\n")
            f.write("2. **Test on more diverse tasks**: Evaluate generalizability\n")
            f.write("3. **Conduct larger studies**: Increase sample size for stronger evidence\n")
        else:
            f.write("1. **Reconsider approach**: May need fundamental modifications\n")
            f.write("2. **Analyze failure cases**: Understand when EAS does not help\n")
            f.write("3. **Test alternative implementations**: Explore different attractor mechanisms\n")
    
    print(f"Detailed report saved to: {report_path}")


def main():
    print("=" * 80)
    print("COMPREHENSIVE EAS VALIDATION SUITE")
    print("Complete validation with statistical analysis, visualizations, and engineering report")
    print("=" * 80)
    
    # Create validator
    validator = ComprehensiveEASValidator(output_dir="comprehensive_validation_results")
    
    # Run comprehensive validation
    results = validator.run_complete_validation()
    
    # Generate detailed report
    generate_detailed_report(results, validator.output_dir)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Results directory: {validator.output_dir}")
    print(f"- Results: {os.path.join(validator.output_dir, 'results')}")
    print(f"- Plots: {os.path.join(validator.output_dir, 'plots')}")
    print(f"- Report: {os.path.join(validator.output_dir, 'reports', 'detailed_eas_validation_report.md')}")
    
    # Print summary
    summary = results['summary_metrics']
    print(f"\nSUMMARY:")
    print(f"  Baseline Accuracy: {summary['baseline_mean_accuracy']:.4f}")
    print(f"  EAS Accuracy: {summary['eas_mean_accuracy']:.4f}") 
    print(f"  Improvement: {summary['mean_improvement']:.4f} ({summary['improvement_percentage']:.2f}%)")
    print(f"  Statistical Significance: {'Yes' if summary['statistical_significance'] else 'No'}")
    print(f"  Effect Size: {summary['effect_size']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())