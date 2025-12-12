"""
Breakthrough Validation Suite
Comprehensive validation for the new EAS research directions:
1. Contrastive Learning validation
2. Self-Supervised Learning validation  
3. Cross-model Transfer experiments
"""
import torch
import numpy as np
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Import watchers
from eas.src.watcher import EmergentWatcher
from eas.src.watcher.contrastive_watcher import ContrastiveWatcher, create_contrastive_watcher
from eas.src.watcher.self_supervised_watcher import SelfSupervisedWatcher, create_self_supervised_watcher

# Import datasets
from eas.src.datasets.paired_dataset import PairedDatasetGenerator, create_nli_dataset

# Import models
try:
    from eas.src.models.transformer import PretrainedTransformer
    HAS_PRETRAINED = True
except ImportError:
    HAS_PRETRAINED = False
    print("Warning: PretrainedTransformer not available, using toy model fallback")

# Import interpretability (may fail if matplotlib has issues)
try:
    from eas.advanced_validation.interpretability import AttractorVisualizer
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    AttractorVisualizer = None
    print("Warning: Visualization tools not available")


@dataclass
class ExperimentResult:
    """Structured result from a single experiment run"""
    experiment_name: str
    model_name: str
    watcher_type: str
    baseline_accuracy: float
    eas_accuracy: float
    improvement: float
    num_samples: int
    duration_seconds: float
    additional_metrics: Dict[str, Any]
    

class BreakthroughValidationSuite:
    """
    Unified validation suite for all breakthrough research directions.
    """
    def __init__(self, 
                 model_name: str = "EleutherAI/pythia-70m",
                 device: str = "cpu",
                 results_dir: str = "eas/advanced_validation/results/breakthrough"):
        
        self.model_name = model_name
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize model
        if HAS_PRETRAINED:
            self.model = PretrainedTransformer(model_name, device)
            self.hidden_dim = self.model.model.config.hidden_size
            self.tokenizer = self.model.tokenizer
        else:
            self.hidden_dim = 512
            self.model = None
            self.tokenizer = None
        
        # Initialize data generators
        self.paired_gen = PairedDatasetGenerator()
        
        # Initialize visualizer (optional)
        if HAS_VISUALIZATION and AttractorVisualizer is not None:
            self.visualizer = AttractorVisualizer(save_dir=os.path.join(results_dir, "visualizations"))
        else:
            self.visualizer = None
        
        # Results storage
        self.results: List[ExperimentResult] = []
        
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to input IDs"""
        if self.tokenizer:
            return self.tokenizer(text, return_tensors="pt", 
                                  truncation=True, max_length=128).input_ids.to(self.device)
        else:
            # Fallback: random encoding
            return torch.randint(0, 1000, (1, 20)).to(self.device)
    
    def _get_model_prediction(self, input_ids: torch.Tensor, 
                               watcher: Optional[torch.nn.Module] = None,
                               intervention_layer: int = 3) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Get model prediction with optional watcher intervention.
        Returns: (predicted_token, hidden_states, logits)
        """
        if self.model is None:
            # Fallback for no model
            hidden = torch.randn(1, 20, self.hidden_dim)
            logits = torch.randn(1, 20, 50000)
            return "yes", hidden, logits
        
        # Register intervention hook if watcher provided
        if watcher is not None:
            self.model.register_intervention_hook(intervention_layer, watcher.snap)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model.forward(input_ids)
        
        # Get hidden states from intervention layer
        hidden_states = self.model.get_layer_activation(intervention_layer)
        
        # Remove hook
        if watcher is not None:
            self.model.remove_intervention_hook(intervention_layer)
        
        # Get prediction
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        predicted_id = logits[0, -1].argmax().item()
        predicted_token = self.tokenizer.decode([predicted_id]).strip().lower()
        
        return predicted_token, hidden_states, logits
    
    def run_contrastive_experiment(self, 
                                    num_samples: int = 100,
                                    warmup_pairs: int = 20,
                                    intervention_layer: int = 3) -> ExperimentResult:
        """
        Validate Contrastive Attractor Learning.
        
        Key test: Does learning from success/failure pairs outperform standard EAS?
        """
        print(f"\n{'='*60}")
        print("CONTRASTIVE LEARNING EXPERIMENT")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate paired data
        pairs = self.paired_gen.generate_dataset(num_samples + warmup_pairs)
        warmup_data = pairs[:warmup_pairs]
        test_data = pairs[warmup_pairs:]
        
        # Initialize watchers
        contrastive_watcher = create_contrastive_watcher(dim=self.hidden_dim, k=10)
        standard_watcher = EmergentWatcher(dim=self.hidden_dim, k=10)
        
        # --- WARMUP PHASE ---
        print(f"\nWarmup with {warmup_pairs} contrastive pairs...")
        for pair in warmup_data:
            success_text, failure_text = self.paired_gen.format_for_training(pair)
            
            # Get activations for success and failure
            success_ids = self._encode_text(success_text)
            failure_ids = self._encode_text(failure_text)
            
            _, success_hidden, _ = self._get_model_prediction(success_ids)
            _, failure_hidden, _ = self._get_model_prediction(failure_ids)
            
            # Update contrastive watcher with pairs
            if success_hidden is not None and failure_hidden is not None:
                contrastive_watcher.update_contrastive(success_hidden, failure_hidden)
                standard_watcher.update(success_hidden)  # Standard only sees success
        
        # --- TEST PHASE ---
        print(f"\nTesting on {len(test_data)} samples...")
        nli_data = self.paired_gen.generate_entailment_pairs(len(test_data))
        
        baseline_correct = 0
        standard_correct = 0
        contrastive_correct = 0
        
        success_activations = []
        failure_activations = []
        
        for sample in nli_data:
            prompt = f"Premise: {sample['premise']} Hypothesis: {sample['hypothesis']} Answer (yes/no):"
            target = sample['label'].lower()
            
            input_ids = self._encode_text(prompt)
            
            # Baseline (no intervention)
            pred_base, hidden_base, _ = self._get_model_prediction(input_ids)
            if target in pred_base:
                baseline_correct += 1
            
            # Standard EAS
            pred_std, hidden_std, _ = self._get_model_prediction(input_ids, standard_watcher)
            if target in pred_std:
                standard_correct += 1
            
            # Contrastive EAS
            pred_con, hidden_con, _ = self._get_model_prediction(input_ids, contrastive_watcher)
            if target in pred_con:
                contrastive_correct += 1
            
            # Collect activations for visualization
            if hidden_base is not None:
                if sample['is_correct']:
                    success_activations.append(hidden_base.mean(dim=1).cpu().numpy())
                else:
                    failure_activations.append(hidden_base.mean(dim=1).cpu().numpy())
        
        # Calculate accuracies
        n = len(nli_data)
        baseline_acc = baseline_correct / n
        standard_acc = standard_correct / n
        contrastive_acc = contrastive_correct / n
        
        duration = time.time() - start_time
        
        # Log results
        print(f"\nResults:")
        print(f"  Baseline:    {baseline_acc:.1%}")
        print(f"  Standard:    {standard_acc:.1%} ({standard_acc - baseline_acc:+.1%})")
        print(f"  Contrastive: {contrastive_acc:.1%} ({contrastive_acc - baseline_acc:+.1%})")
        
        # Create visualization if we have data and visualizer is available
        if success_activations and failure_activations and self.visualizer is not None:
            success_arr = np.vstack(success_activations)
            failure_arr = np.vstack(failure_activations)
            attractors = contrastive_watcher.attractor_memory.positive_attractors.detach().cpu().numpy()
            
            try:
                viz_path = self.visualizer.plot_success_failure_contrast(
                    success_arr, failure_arr, attractors,
                    filename=f"contrastive_{self.model_name.split('/')[-1]}.png"
                )
                print(f"  Visualization saved: {viz_path}")
            except Exception as e:
                print(f"  Visualization skipped: {e}")
        
        result = ExperimentResult(
            experiment_name="contrastive_learning",
            model_name=self.model_name,
            watcher_type="ContrastiveWatcher",
            baseline_accuracy=baseline_acc,
            eas_accuracy=contrastive_acc,
            improvement=contrastive_acc - baseline_acc,
            num_samples=n,
            duration_seconds=duration,
            additional_metrics={
                "standard_eas_accuracy": standard_acc,
                "standard_improvement": standard_acc - baseline_acc,
                "contrastive_vs_standard": contrastive_acc - standard_acc,
                "repulsion_ratio": contrastive_watcher.get_statistics()["repulsion_ratio"],
                "warmup_pairs": warmup_pairs
            }
        )
        
        self.results.append(result)
        return result
    
    def run_self_supervised_experiment(self,
                                        num_samples: int = 100,
                                        warmup_samples: int = 50,
                                        intervention_layer: int = 3) -> ExperimentResult:
        """
        Validate Self-Supervised Learning (no labels!).
        
        Key test: Can confidence-based learning match supervised performance?
        """
        print(f"\n{'='*60}")
        print("SELF-SUPERVISED LEARNING EXPERIMENT")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate NLI data
        nli_data = create_nli_dataset(num_samples + warmup_samples)
        warmup_data = nli_data[:warmup_samples]
        test_data = nli_data[warmup_samples:]
        
        # Initialize watchers
        ss_watcher = create_self_supervised_watcher(dim=self.hidden_dim, k=10)
        supervised_watcher = EmergentWatcher(dim=self.hidden_dim, k=10)
        
        # --- WARMUP PHASE ---
        print(f"\nWarmup with {warmup_samples} samples...")
        print("  Self-supervised: using model confidence only (NO LABELS)")
        print("  Supervised: using ground truth labels")
        
        for sample in warmup_data:
            prompt = f"Premise: {sample['premise']} Hypothesis: {sample['hypothesis']} Answer:"
            input_ids = self._encode_text(prompt)
            
            pred, hidden, logits = self._get_model_prediction(input_ids)
            
            if hidden is not None and logits is not None:
                # Self-supervised: just uses confidence
                ss_watcher.self_supervised_update(hidden, logits)
                
                # Supervised: uses ground truth
                target = sample['label'].lower()
                if target in pred:
                    supervised_watcher.update(hidden)
        
        # --- TEST PHASE ---
        print(f"\nTesting on {len(test_data)} samples...")
        
        baseline_correct = 0
        supervised_correct = 0
        self_supervised_correct = 0
        
        for sample in test_data:
            prompt = f"Premise: {sample['premise']} Hypothesis: {sample['hypothesis']} Answer (yes/no):"
            target = sample['label'].lower()
            input_ids = self._encode_text(prompt)
            
            # Baseline
            pred_base, _, _ = self._get_model_prediction(input_ids)
            if target in pred_base:
                baseline_correct += 1
            
            # Supervised EAS
            pred_sup, _, _ = self._get_model_prediction(input_ids, supervised_watcher)
            if target in pred_sup:
                supervised_correct += 1
            
            # Self-supervised EAS
            pred_ss, _, _ = self._get_model_prediction(input_ids, ss_watcher)
            if target in pred_ss:
                self_supervised_correct += 1
        
        # Calculate accuracies
        n = len(test_data)
        baseline_acc = baseline_correct / n
        supervised_acc = supervised_correct / n
        ss_acc = self_supervised_correct / n
        
        duration = time.time() - start_time
        
        # Log results
        print(f"\nResults:")
        print(f"  Baseline:        {baseline_acc:.1%}")
        print(f"  Supervised:      {supervised_acc:.1%} ({supervised_acc - baseline_acc:+.1%})")
        print(f"  Self-Supervised: {ss_acc:.1%} ({ss_acc - baseline_acc:+.1%})")
        print(f"  SS vs Supervised: {ss_acc - supervised_acc:+.1%}")
        
        ss_stats = ss_watcher.get_statistics()
        print(f"\nSelf-Supervised Stats:")
        print(f"  High confidence updates: {ss_stats['high_confidence_updates']}")
        print(f"  Low confidence skips: {ss_stats['low_confidence_skips']}")
        print(f"  Mean confidence: {ss_stats['mean_confidence']:.3f}")
        
        result = ExperimentResult(
            experiment_name="self_supervised_learning",
            model_name=self.model_name,
            watcher_type="SelfSupervisedWatcher",
            baseline_accuracy=baseline_acc,
            eas_accuracy=ss_acc,
            improvement=ss_acc - baseline_acc,
            num_samples=n,
            duration_seconds=duration,
            additional_metrics={
                "supervised_accuracy": supervised_acc,
                "supervised_improvement": supervised_acc - baseline_acc,
                "ss_vs_supervised": ss_acc - supervised_acc,
                "high_confidence_ratio": ss_stats["high_confidence_ratio"],
                "mean_confidence": ss_stats["mean_confidence"],
                "uncertainty_regions": ss_stats["uncertainty_regions"]
            }
        )
        
        self.results.append(result)
        return result
    
    def run_all_experiments(self) -> Dict[str, ExperimentResult]:
        """Run all breakthrough experiments and save results"""
        print(f"\n{'#'*60}")
        print(f"# BREAKTHROUGH VALIDATION SUITE")
        print(f"# Model: {self.model_name}")
        print(f"# Device: {self.device}")
        print(f"{'#'*60}")
        
        experiments = {}
        
        # Run contrastive experiment
        experiments['contrastive'] = self.run_contrastive_experiment()
        
        # Run self-supervised experiment
        experiments['self_supervised'] = self.run_self_supervised_experiment()
        
        # Save results
        self._save_results()
        
        # Print summary
        self._print_summary(experiments)
        
        return experiments
    
    def _save_results(self):
        """Save all results to JSON"""
        results_dict = [asdict(r) for r in self.results]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.results_dir, f"breakthrough_results_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {filepath}")
    
    def _print_summary(self, experiments: Dict[str, ExperimentResult]):
        """Print summary of all experiments"""
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for name, result in experiments.items():
            print(f"\n{name.upper()}:")
            print(f"  Baseline: {result.baseline_accuracy:.1%}")
            print(f"  EAS:      {result.eas_accuracy:.1%}")
            print(f"  Δ:        {result.improvement:+.1%}")


def run_quick_validation():
    """Quick smoke test with minimal samples"""
    print("Running quick breakthrough validation...")
    
    suite = BreakthroughValidationSuite(
        model_name="EleutherAI/pythia-70m",
        device="cpu"
    )
    
    # Run with minimal samples for quick testing
    suite.run_contrastive_experiment(num_samples=20, warmup_pairs=5)
    suite.run_self_supervised_experiment(num_samples=20, warmup_samples=10)
    
    print("\nQuick validation complete!")


if __name__ == "__main__":
    run_quick_validation()
