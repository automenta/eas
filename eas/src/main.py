"""
Main EAS Experiment Runner
Runs the complete small-scale experiment to validate the EAS approach
"""
import torch
import os
import sys
from typing import Dict, Any

# Add the project root to path
sys.path.append('/home/me/eas')

from eas.src.models.transformer import AutoregressiveTransformer
from eas.src.models.tokenizer import LogicTokenizer
from eas.src.watcher import EmergentWatcher
from eas.src.datasets import LogicCorpusGenerator
from eas.src.experiments import EASEvaluator
from eas.src.experiments.baselines import run_baseline_experiments
from eas.src.utils.metrics import EASLogger, log_experiment_comparison


def run_small_scale_eas_experiment():
    """Run the complete small-scale EAS validation experiment."""
    print("Starting small-scale EAS validation experiment...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Create tokenizer
    print("\n1. Creating tokenizer...")
    tokenizer = LogicTokenizer(vocab_size=200)  # Smaller vocab for rapid prototyping
    print(f"Tokenizer created with vocab size: {tokenizer.get_vocab_size()}")
    
    # 2. Create small model for rapid prototyping
    print("\n2. Creating small model...")
    model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,      # Smaller dimension
        num_layers=1,     # Single layer
        num_heads=4       # Fewer heads
    ).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Create watcher for the small model
    print("\n3. Creating Emergent Watcher...")
    watcher = EmergentWatcher(
        dim=128,          # Same as model dimension
        k=5,              # Fewer attractors for small model
        alpha_base=0.3,
        max_delta=0.3,
        update_frequency=3  # Update more frequently for small experiment
    ).to(device)
    print("Watcher created successfully")
    
    # 4. Generate small datasets for rapid testing
    print("\n4. Generating small datasets...")
    generator = LogicCorpusGenerator()
    datasets = {
        'pretrain': generator.generate_pretraining_dataset(100),   # Small pre-training set
        'evaluation': generator.generate_evaluation_dataset(30),   # Small evaluation set
        'validation': generator.generate_validation_dataset(20)    # Small validation set
    }
    
    print(f"Pre-training set size: {len(datasets['pretrain'])}")
    print(f"Evaluation set size: {len(datasets['evaluation'])}")
    print(f"Validation set size: {len(datasets['validation'])}")
    
    # 5. Initialize logger
    print("\n5. Initializing logger...")
    logger = EASLogger(log_dir="logs", experiment_name="EAS_Small_Scale_Validation")
    
    # Log hyperparameters
    config = {
        'model_d_model': 128,
        'model_num_layers': 1,
        'model_num_heads': 4,
        'watcher_k': 5,
        'watcher_alpha_base': 0.3,
        'watcher_max_delta': 0.3,
        'vocab_size': 200,
        'dataset_sizes': {k: len(v) for k, v in datasets.items()},
        'device': str(device)
    }
    logger.log_hyperparameters(config)
    
    # 6. Train base model (quick training)
    print("\n6. Training base model...")
    evaluator = EASEvaluator(model, tokenizer, watcher, device=device)
    evaluator.train_base_model(datasets['pretrain'], epochs=2)  # Quick training
    
    # 7. Run baselines for comparison
    print("\n7. Running baseline experiments...")
    baseline_results = run_baseline_experiments()
    print("Baseline experiments completed")
    
    # 8. Run EAS experiment
    print("\n8. Running EAS experiment...")
    eas_metrics = evaluator.evaluate_with_eas(datasets['evaluation'], num_iterations=50)
    print("EAS experiment completed")
    
    # 9. Log comparison
    print("\n9. Logging comparisons...")
    log_experiment_comparison(
        baseline_results.get('baseline', {}).get('metrics', {}),
        eas_metrics,
        logger
    )
    
    # 10. Log final results and save visualizations
    print("\n10. Saving results and visualizations...")
    logger.log_final_results()
    logger.save_visualizations()
    
    # 11. Print summary
    print("\n=== EXPERIMENT SUMMARY ===")
    final_accuracy = eas_metrics['accuracy'][-1] if eas_metrics['accuracy'] else 0
    avg_latency = sum(eas_metrics['latency']) / len(eas_metrics['latency']) if eas_metrics['latency'] else 0
    
    baseline_acc = baseline_results.get('baseline', {}).get('metrics', {}).get('accuracy', [])
    baseline_final = baseline_acc[-1] if baseline_acc else 0
    
    print(f"Baseline Accuracy: {baseline_final:.4f}")
    print(f"EAS Final Accuracy: {final_accuracy:.4f}")
    print(f"Improvement: {final_accuracy - baseline_final:.4f}")
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Total Interventions: {len(eas_metrics['intervention_frequency'])}")
    print(f"Final Attractor Stability: {eas_metrics['attractor_stability'][-1] if eas_metrics['attractor_stability'] else 0:.4f}")
    
    # 12. Success criteria evaluation
    print("\n=== SUCCESS CRITERIA EVALUATION ===")
    accuracy_improved = (final_accuracy - baseline_final) >= 0.15  # 15% improvement threshold for small model
    latency_acceptable = avg_latency < 0.1  # Less strict for small model
    print(f"Accuracy Improvement ≥15%: {accuracy_improved}")
    print(f"Latency Overhead <0.1s: {latency_acceptable}")
    
    success = accuracy_improved and latency_acceptable
    print(f"Overall Success: {success}")
    
    return {
        'success': success,
        'accuracy_improvement': final_accuracy - baseline_final,
        'final_accuracy': final_accuracy,
        'baseline_accuracy': baseline_final,
        'avg_latency': avg_latency,
        'metrics': eas_metrics
    }


def main():
    """Main function to run the EAS validation experiment."""
    print("Emergent Activation Snapping (EAS) Validation Experiment")
    print("=" * 60)
    
    try:
        results = run_small_scale_eas_experiment()
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED")
        print(f"Success: {results['success']}")
        print(f"Accuracy Improvement: {results['accuracy_improvement']:.4f}")
        print(f"Final Accuracy: {results['final_accuracy']:.4f}")
        print(f"Baseline Accuracy: {results['baseline_accuracy']:.4f}")
        
        # Exit with appropriate code
        exit_code = 0 if results['success'] else 1
        return exit_code
        
    except Exception as e:
        print(f"\nEXPERIMENT FAILED with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)