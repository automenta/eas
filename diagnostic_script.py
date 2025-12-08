#!/usr/bin/env python3
"""
Diagnostic script to identify why datasets show identical results in EAS validation
"""
import os
import sys
import random
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from eas.src.models.tokenizer import create_small_tokenizer
from eas.src.models.transformer import create_small_model
from eas.src.watcher import EmergentWatcher
from eas.src.datasets import LogicCorpusGenerator
from eas.src.experiments import EASEvaluator
from eas.src.experiments.baselines import BaseEvaluator


def create_diverse_datasets():
    """Create genuinely diverse datasets"""
    print("Creating diverse testing datasets...")
    generator = LogicCorpusGenerator()
    
    datasets = {}
    
    # Dataset 1: Basic syllogisms
    datasets["syllogisms_basic"] = generator.generate_dataset(20, include_invalid=True)
    
    # Dataset 2: Complex logic with more invalid problems
    challenging_samples = []
    for _ in range(20):
        # 70% chance of making it invalid/challenging
        sample = generator.generate_challenging_sample()
        if random.random() < 0.7:
            sample["validity"] = False  # Force invalid
        challenging_samples.append(sample)
    datasets["challenging_logic"] = challenging_samples
    
    # Dataset 3: Propositional logic focus
    propositional_samples = []
    for _ in range(20):
        if random.random() < 0.6:
            # More propositional logic samples
            problem, is_valid = generator.generate_propositional_logic()
            # Format into standard structure
            parts = problem.split(" -> ")
            if len(parts) == 2:
                premises_text = parts[0].strip()
                conclusion = parts[1].strip()
                premises = [p.strip() for p in premises_text.split(".") if p.strip()]
            else:
                premises = []
                conclusion = problem
            propositional_samples.append({
                "premise1": premises[0] if len(premises) > 0 else "",
                "premise2": premises[1] if len(premises) > 1 else "",
                "conclusion": conclusion,
                "validity": is_valid,
                "logical_type": "propositional_logic",
                "problem_text": problem
            })
        else:
            propositional_samples.append(generator.generate_sample())
    datasets["propositional_focus"] = propositional_samples
    
    # Dataset 4: Mixed difficulty with different characteristics
    mixed_samples = []
    for _ in range(20):
        if random.random() < 0.3:
            # Easy problems (syllogisms)
            problem, is_valid = generator.generate_classic_syllogism()
        elif random.random() < 0.6:
            # Medium problems (propositional)
            problem, is_valid = generator.generate_propositional_logic()
        else:
            # Hard problems 
            sample = generator.generate_challenging_sample()
            problem = sample["problem_text"]
            is_valid = sample["validity"]
        
        # Format consistently
        parts = problem.split(" -> ")
        if len(parts) == 2:
            premises_text = parts[0].strip()
            conclusion = parts[1].strip()
            premises = [p.strip() for p in premises_text.split(".") if p.strip()]
        else:
            premises = []
            conclusion = problem
        
        mixed_samples.append({
            "premise1": premises[0] if len(premises) > 0 else "",
            "premise2": premises[1] if len(premises) > 1 else "",
            "conclusion": conclusion,
            "validity": is_valid,
            "problem_text": problem
        })
    datasets["mixed_difficulty"] = mixed_samples
    
    # Dataset 5: Very high challenge ratio
    high_challenge_samples = []
    for _ in range(20):
        # 90% chance of making it invalid/challenging
        sample = generator.generate_challenging_sample()
        if random.random() < 0.9:
            sample["validity"] = False  # Force invalid to make it challenging
        high_challenge_samples.append(sample)
    datasets["high_challenge"] = high_challenge_samples
    
    return datasets


def test_single_dataset_performance(dataset_name, dataset, trial_seed):
    """Test performance on a single dataset with unique seed"""
    # Set unique seed for this trial
    torch.manual_seed(trial_seed)
    random.seed(trial_seed)
    np.random.seed(trial_seed)
    
    print(f"Testing on {dataset_name}...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create new model and tokenizer for this trial
    tokenizer = create_small_tokenizer(vocab_size=200)
    model = create_small_model(vocab_size=tokenizer.get_vocab_size())
    model.to(device)

    # Train model
    evaluator = EASEvaluator(model, tokenizer, device=device)
    evaluator.train_base_model(dataset, epochs=1)

    # Run baseline
    baseline_eval = BaseEvaluator(model, tokenizer, device=device)
    baseline_eval.model_trained = True
    baseline_results = baseline_eval.evaluate_baseline(dataset, num_iterations=25)
    baseline_acc = baseline_results['accuracy'][-1] if baseline_results['accuracy'] else 0.5

    # Create new watcher
    watcher = EmergentWatcher(
        dim=128,
        k=5,
        alpha_base=0.3,
        max_delta=0.3,
        update_frequency=3
    ).to(device)

    # Run EAS
    eas_eval = EASEvaluator(model, tokenizer, watcher, device=device)
    eas_eval.model_trained = True
    eas_results = eas_eval.evaluate_with_eas(dataset, num_iterations=25)
    eas_acc = eas_results['accuracy'][-1] if eas_results['accuracy'] else 0.5
    
    print(f"  {dataset_name}: Baseline={baseline_acc:.4f}, EAS={eas_acc:.4f}, Diff={eas_acc - baseline_acc:.4f}")
    
    return baseline_acc, eas_acc


def run_diagnostic():
    """Run full diagnostic to identify the issue"""
    print("=" * 80)
    print("EAS VALIDATION DIAGNOSTIC")
    print("Identifying why all datasets show identical results")
    print("=" * 80)
    
    # Create diverse datasets
    datasets = create_diverse_datasets()
    
    print(f"Created {len(datasets)} diverse datasets:")
    for name, data in datasets.items():
        valid_count = sum(1 for s in data if s.get('validity', False))
        total_count = len(data)
        print(f"  {name}: {total_count} samples, {valid_count} valid, {total_count-valid_count} invalid")
    print()
    
    # Test each dataset separately with different seeds
    results = {}
    for i, (name, dataset) in enumerate(datasets.items(), 1):
        # Use a different seed for each dataset
        baseline_acc, eas_acc = test_single_dataset_performance(name, dataset, trial_seed=1000+i)
        results[name] = {
            'baseline': baseline_acc,
            'eas': eas_acc,
            'improvement': eas_acc - baseline_acc
        }
        print(f"  Result: Improvement = {results[name]['improvement']:.4f}")
        print()
    
    print("=" * 80)
    print("DIAGNOSTIC RESULTS")
    print("=" * 80)
    
    print("Dataset Results:")
    for name, data in results.items():
        print(f"{name:20s}: Baseline={data['baseline']:6.4f}, EAS={data['eas']:6.4f}, Diff={data['improvement']:6.4f}")
    
    print()
    # Check if results are identical
    improvements = [data['improvement'] for data in results.values()]
    baseline_accs = [data['baseline'] for data in results.values()]
    eas_accs = [data['eas'] for data in results.values()]
    
    improvements_identical = len(set([round(x, 4) for x in improvements])) == 1
    baseline_identical = len(set([round(x, 4) for x in baseline_accs])) == 1
    eas_identical = len(set([round(x, 4) for x in eas_accs])) == 1
    
    print(f"Are all improvements identical? {improvements_identical}")
    print(f"Are all baseline accuracies identical? {baseline_identical}")  
    print(f"Are all EAS accuracies identical? {eas_identical}")
    
    if improvements_identical and baseline_identical and eas_identical:
        print("\nPROBLEM IDENTIFIED: All datasets showing identical results.")
        print("This suggests:")
        print("1. Model behavior is too deterministic")
        print("2. Problem generation may not be varied enough")
        print("3. Same model/training conditions across datasets")
        print("4. Intervention effects are uniform regardless of dataset")
    else:
        print("\nGOOD: Datasets show varying results as expected.")
    
    print("\nImprovements:", [f"{x:.4f}" for x in improvements])
    print("Baselines:", [f"{x:.4f}" for x in baseline_accs])
    print("EAS:", [f"{x:.4f}" for x in eas_accs])


if __name__ == "__main__":
    run_diagnostic()