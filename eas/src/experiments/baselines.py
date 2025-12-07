"""
Baseline Conditions for EAS Experiment
Implements various baseline conditions for comparison with EAS approach
"""
import torch
import torch.nn.functional as F
import random
import time
from typing import List, Dict, Tuple
from ..models.transformer import AutoregressiveTransformer
from ..models.tokenizer import LogicTokenizer
from ..watcher import EmergentWatcher
from ..experiments import EASEvaluator


class BaseEvaluator(EASEvaluator):
    """Baseline evaluator without any Watcher intervention"""
    
    def evaluate_baseline(self, dataset: List[Dict], num_iterations: int = 200):
        """Evaluate base model without any watcher intervention."""
        if not self.model_trained:
            raise ValueError("Base model must be trained before evaluation")
        
        print(f"Starting baseline evaluation (no watcher) for {num_iterations} iterations...")
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        for iteration in range(num_iterations):
            sample = random.choice(dataset)
            input_ids, target_ids = self.encode_problem(sample)
            
            with torch.no_grad():
                output = self.model(input_ids)
                is_correct = self.check_correctness(output, target_ids, sample)
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
            
            # Record accuracy
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.metrics['accuracy'].append(current_accuracy)
            
            if (iteration + 1) % 25 == 0:
                print(f"Baseline Iteration {iteration + 1}/{num_iterations} - "
                      f"Accuracy: {current_accuracy:.4f}")
        
        print(f"Baseline evaluation completed. Final accuracy: {correct_predictions/total_predictions:.4f}")
        return self.metrics


class RandomControlEvaluator(EASEvaluator):
    """Random Control evaluator: Watcher enabled but update() disabled"""
    
    def evaluate_random_control(self, dataset: List[Dict], num_iterations: int = 200):
        """Evaluate with watcher enabled but update disabled (static random attractors)."""
        if not self.model_trained or self.watcher is None:
            raise ValueError("Base model and watcher must be initialized")
        
        print(f"Starting random control evaluation (watcher with update disabled) for {num_iterations} iterations...")
        
        # Temporarily disable the update method
        original_update = self.watcher.update
        
        def disabled_update(*args, **kwargs):
            # Do nothing - this disables attractor updates
            pass
        
        self.watcher.update = disabled_update
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        # Register intervention hook
        def intervention_func(activations):
            return self.watcher.snap(activations)
        
        self.model.register_intervention_hook(self.model.middle_layer_idx, intervention_func)
        
        for iteration in range(num_iterations):
            sample = random.choice(dataset)
            input_ids, target_ids = self.encode_problem(sample)
            
            with torch.no_grad():
                # Clear previous activations
                self.model.layer_activations.clear()
                
                output = self.model(input_ids)
                is_correct = self.check_correctness(output, target_ids, sample)
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
            
            # Record accuracy
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.metrics['accuracy'].append(current_accuracy)
            
            if (iteration + 1) % 25 == 0:
                print(f"Random Control Iteration {iteration + 1}/{num_iterations} - "
                      f"Accuracy: {current_accuracy:.4f}")
        
        # Restore the original update method
        self.watcher.update = original_update
        
        # Remove intervention hook
        self.model.remove_intervention_hook(self.model.middle_layer_idx)
        
        print(f"Random control evaluation completed. Final accuracy: {correct_predictions/total_predictions:.4f}")
        return self.metrics


class FixedSteeringEvaluator(EASEvaluator):
    """Fixed Steering evaluator: Constant alpha value (no adaptive strength)"""
    
    def __init__(self, model: AutoregressiveTransformer, tokenizer: LogicTokenizer,
                 fixed_alpha: float = 0.3, device: str = 'cpu', max_seq_len: int = 128):
        super().__init__(model, tokenizer, device=device, max_seq_len=max_seq_len)

        # Create a watcher with fixed alpha behavior
        if hasattr(self.model, 'd_model') and self.model.d_model:
            dim = self.model.d_model
        else:
            dim = 128  # Use smaller dimension for small model

        self.fixed_alpha_watcher = EmergentWatcher(
            dim=dim,
            k=5,  # Use fewer attractors for small model
            alpha_base=fixed_alpha,
            max_delta=0.5,
            update_frequency=3  # Update more frequently for small model
        ).to(device)

        # Override alpha behavior in the watcher
        self.fixed_alpha_watcher.alpha_base = fixed_alpha
    
    def snap_fixed_alpha(self, hidden_states):
        """Modified snap function with fixed alpha"""
        # Pool over sequence dimension to get sentence-level vector
        v_raw = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # Update whitening buffer with raw activations
        self.fixed_alpha_watcher.whitening_buffer.update(v_raw)
        
        # Apply whitening normalization
        v_norm = self.fixed_alpha_watcher.whitening_buffer.whiten(v_raw)
        
        # Ensure attractors are normalized
        self.fixed_alpha_watcher.attractor_memory.normalize_attractors()
        attractors_norm = F.normalize(self.fixed_alpha_watcher.attractor_memory.attractors, p=2, dim=1)
        
        # Compute cosine similarity between normalized activations and attractors
        cosine_similarities = torch.mm(v_norm, attractors_norm.t())  # [batch, k]
        
        # Find best matching attractor for each sample in the batch
        best_scores, best_indices = torch.max(cosine_similarities, dim=1)
        
        # Record which attractors were used
        self.fixed_alpha_watcher.snap_history.extend(best_indices.cpu().numpy())
        
        # Get the closest attractors
        closest_att = self.fixed_alpha_watcher.attractor_memory.attractors[best_indices]  # [batch, hidden_dim]
        
        # Use FIXED alpha (not adaptive)
        alpha_fixed = self.fixed_alpha_watcher.alpha_base  # Use the fixed value
        
        # Compute the nudge vector
        delta = closest_att - v_norm
        # Clamp the delta to prevent excessive changes (safety clamp)
        delta = torch.clamp(delta, -self.fixed_alpha_watcher.max_delta, self.fixed_alpha_watcher.max_delta)
        
        # Apply the nudge with fixed alpha
        v_snapped = v_norm + (alpha_fixed * delta)
        
        # Update intervention count
        self.fixed_alpha_watcher.intervention_count += hidden_states.size(0)
        
        # Broadcast back to sequence length (add as residual to original hidden states)
        v_diff = v_snapped.unsqueeze(1) - v_raw.unsqueeze(1)
        
        return hidden_states + v_diff
    
    def evaluate_fixed_steering(self, dataset: List[Dict], num_iterations: int = 200):
        """Evaluate with fixed steering (constant alpha)."""
        if not self.model_trained:
            raise ValueError("Base model must be trained before evaluation")
        
        print(f"Starting fixed steering evaluation (alpha={self.fixed_alpha_watcher.alpha_base}) for {num_iterations} iterations...")
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        # Register intervention hook with fixed alpha
        def intervention_func(activations):
            return self.snap_fixed_alpha(activations)
        
        self.model.register_intervention_hook(self.model.middle_layer_idx, intervention_func)
        
        for iteration in range(num_iterations):
            sample = random.choice(dataset)
            input_ids, target_ids = self.encode_problem(sample)
            
            with torch.no_grad():
                # Clear previous activations
                self.model.layer_activations.clear()
                
                output = self.model(input_ids)
                is_correct = self.check_correctness(output, target_ids, sample)
                
                if is_correct:
                    correct_predictions += 1
                    
                    # Update attractors (the fixed alpha watcher still adapts its attractors)
                    middle_layer_activation = self.model.get_layer_activation(self.model.middle_layer_idx)
                    if middle_layer_activation is not None:
                        self.fixed_alpha_watcher.update(middle_layer_activation)
                
                total_predictions += 1
            
            # Record accuracy
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.metrics['accuracy'].append(current_accuracy)
            
            if (iteration + 1) % 25 == 0:
                print(f"Fixed Steering Iteration {iteration + 1}/{num_iterations} - "
                      f"Accuracy: {current_accuracy:.4f}")
        
        # Remove intervention hook
        self.model.remove_intervention_hook(self.model.middle_layer_idx)
        
        print(f"Fixed steering evaluation completed. Final accuracy: {correct_predictions/total_predictions:.4f}")
        return self.metrics


class NoClampingEvaluator(EASEvaluator):
    """No Clamping evaluator: Without safety magnitude clamping"""
    
    def snap_no_clamping(self, hidden_states):
        """Modified snap function without clamping"""
        # Pool over sequence dimension to get sentence-level vector
        v_raw = hidden_states.mean(dim=1)  # [batch, hidden_dim]

        # Update whitening buffer with raw activations
        self.watcher.whitening_buffer.update(v_raw)

        # Apply whitening normalization
        v_norm = self.watcher.whitening_buffer.whiten(v_raw)

        # Ensure attractors are normalized
        self.watcher.attractor_memory.normalize_attractors()
        attractors_norm = F.normalize(self.watcher.attractor_memory.attractors, p=2, dim=1)

        # Compute cosine similarity between normalized activations and attractors
        cosine_similarities = torch.mm(v_norm, attractors_norm.t())  # [batch, k]

        # Find best matching attractor for each sample in the batch
        best_scores, best_indices = torch.max(cosine_similarities, dim=1)

        # Record which attractors were used
        self.watcher.snap_history.extend(best_indices.cpu().numpy())

        # Get the closest attractors
        closest_att = self.watcher.attractor_memory.attractors[best_indices]  # [batch, hidden_dim]

        # Compute adaptive alpha (dynamic strength)
        alpha_dyn = self.watcher.alpha_base * (1.0 - best_scores.unsqueeze(1))

        # Compute the nudge vector WITHOUT CLAMPING
        delta = closest_att - v_norm
        # NO CLAMPING applied here

        # Apply the nudge
        v_snapped = v_norm + (alpha_dyn * delta)

        # Update intervention count
        self.watcher.intervention_count += hidden_states.size(0)

        # Broadcast back to sequence length (add as residual to original hidden states)
        v_diff = v_snapped.unsqueeze(1) - v_raw.unsqueeze(1)

        return hidden_states + v_diff
    
    def evaluate_no_clamping(self, dataset: List[Dict], num_iterations: int = 200):
        """Evaluate without safety clamping."""
        if not self.model_trained or self.watcher is None:
            raise ValueError("Base model and watcher must be initialized")
        
        print(f"Starting no clamping evaluation for {num_iterations} iterations...")
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        # Register intervention hook without clamping
        def intervention_func(activations):
            return self.snap_no_clamping(activations)
        
        self.model.register_intervention_hook(self.model.middle_layer_idx, intervention_func)
        
        for iteration in range(num_iterations):
            sample = random.choice(dataset)
            input_ids, target_ids = self.encode_problem(sample)
            
            with torch.no_grad():
                # Clear previous activations
                self.model.layer_activations.clear()
                
                output = self.model(input_ids)
                is_correct = self.check_correctness(output, target_ids, sample)
                
                if is_correct:
                    correct_predictions += 1
                    
                    # Update attractors
                    middle_layer_activation = self.model.get_layer_activation(self.model.middle_layer_idx)
                    if middle_layer_activation is not None:
                        self.watcher.update(middle_layer_activation)
                
                total_predictions += 1
            
            # Record accuracy
            current_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            self.metrics['accuracy'].append(current_accuracy)
            
            if (iteration + 1) % 25 == 0:
                print(f"No Clamping Iteration {iteration + 1}/{num_iterations} - "
                      f"Accuracy: {current_accuracy:.4f}")
        
        # Remove intervention hook
        self.model.remove_intervention_hook(self.model.middle_layer_idx)
        
        print(f"No clamping evaluation completed. Final accuracy: {correct_predictions/total_predictions:.4f}")
        return self.metrics


def run_baseline_experiments():
    """Run all baseline experiments for comparison."""
    from ..datasets import LogicCorpusGenerator
    
    # Initialize components
    tokenizer = LogicTokenizer(vocab_size=200)  # Use the same tokenizer as in main
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate datasets
    generator = LogicCorpusGenerator()
    datasets = {
        'pretrain': generator.generate_pretraining_dataset(200),  # Use smaller set for baselines
        'evaluation': generator.generate_evaluation_dataset(50)
    }
    pretrain_data = datasets['pretrain']
    eval_data = datasets['evaluation']
    
    # Create a small model for rapid baseline testing
    small_model = AutoregressiveTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=128,
        num_layers=1,
        num_heads=4
    ).to(device)
    
    # Initialize evaluator
    evaluator = EASEvaluator(small_model, tokenizer, device=device)
    
    print("Training small base model for baselines...")
    evaluator.train_base_model(pretrain_data[:200], epochs=2)  # Use smaller dataset for speed
    
    results = {}
    
    # 1. Baseline: No Watcher (base model only)
    print("\n1. Running Baseline (No Watcher)...")
    baseline_evaluator = BaseEvaluator(small_model, tokenizer, device=device)
    baseline_evaluator.model_trained = True  # Mark as already trained
    baseline_metrics = baseline_evaluator.evaluate_baseline(eval_data[:50], num_iterations=50)
    results['baseline'] = baseline_metrics
    
    # 2. Random Control: Watcher enabled but update() disabled
    print("\n2. Running Random Control (Watcher with update disabled)...")
    watcher_for_control = EmergentWatcher(
        dim=128,
        k=5,
        alpha_base=0.3,
        max_delta=0.3
    ).to(device)
    
    random_control_evaluator = RandomControlEvaluator(
        small_model, tokenizer, watcher_for_control, device=device
    )
    random_control_evaluator.model_trained = True  # Mark as already trained
    random_control_metrics = random_control_evaluator.evaluate_random_control(
        eval_data[:50], num_iterations=50
    )
    results['random_control'] = random_control_metrics
    
    # 3. Fixed Steering: Constant alpha value
    print("\n3. Running Fixed Steering (Constant alpha)...")
    fixed_steering_evaluator = FixedSteeringEvaluator(
        small_model, tokenizer, fixed_alpha=0.3, device=device
    )
    fixed_steering_evaluator.model_trained = True  # Mark as already trained
    fixed_steering_metrics = fixed_steering_evaluator.evaluate_fixed_steering(
        eval_data[:50], num_iterations=50
    )
    results['fixed_steering'] = fixed_steering_metrics
    
    # 4. No Clamping: Without safety magnitude clamping
    print("\n4. Running No Clamping...")
    watcher_for_no_clamp = EmergentWatcher(
        dim=128,
        k=5,
        alpha_base=0.3,
        max_delta=0.3,  # This will be ignored in the no-clamping version
        update_frequency=3  # Use appropriate update frequency for small model
    ).to(device)

    no_clamping_evaluator = NoClampingEvaluator(
        small_model, tokenizer, watcher_for_no_clamp, device=device
    )
    no_clamping_evaluator.model_trained = True  # Mark as already trained
    no_clamping_metrics = no_clamping_evaluator.evaluate_no_clamping(
        eval_data[:50], num_iterations=50
    )
    results['no_clamping'] = no_clamping_metrics
    
    print("\nAll baseline experiments completed!")
    return results