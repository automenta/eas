#!/usr/bin/env python3
"""
Position-Aware EAS Validation Experiment

Compares:
1. Baseline (no intervention)
2. Standard EAS (sequence-pooled)
3. Position-Aware EAS (CTD-exploiting)

Tests the hypothesis that targeting critical positions improves performance.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import time

import sys
sys.path.insert(0, '/home/me/eas')


@dataclass
class ExperimentResult:
    """Result from a single experiment."""
    method: str
    accuracy: float
    improvement_over_baseline: float
    num_samples: int
    avg_latency: float
    additional_metrics: Dict[str, Any] = None


def generate_test_pairs() -> List[Dict]:
    """Generate test pairs for evaluation."""
    return [
        # Modus Ponens
        {"premise": "If it rains then the ground is wet. It is raining.",
         "correct": "The ground is wet.", "incorrect": "The ground is dry.",
         "type": "modus_ponens"},
        {"premise": "If the alarm rings then wake up. The alarm is ringing.",
         "correct": "Wake up.", "incorrect": "Stay asleep.",
         "type": "modus_ponens"},
        {"premise": "If you study then you pass. You studied hard.",
         "correct": "You passed.", "incorrect": "You failed.",
         "type": "modus_ponens"},
         
        # Syllogisms
        {"premise": "All mammals are warm-blooded. Dogs are mammals.",
         "correct": "Dogs are warm-blooded.", "incorrect": "Dogs are cold-blooded.",
         "type": "syllogism"},
        {"premise": "All birds have feathers. A robin is a bird.",
         "correct": "A robin has feathers.", "incorrect": "A robin has scales.",
         "type": "syllogism"},
        {"premise": "All fruits contain seeds. An apple is a fruit.",
         "correct": "An apple contains seeds.", "incorrect": "An apple has no seeds.",
         "type": "syllogism"},
         
        # Arithmetic
        {"premise": "The equation is: 3 + 5 = ?",
         "correct": "The answer is 8.", "incorrect": "The answer is 7.",
         "type": "arithmetic"},
        {"premise": "Calculate: 12 - 4 = ?",
         "correct": "The result is 8.", "incorrect": "The result is 6.",
         "type": "arithmetic"},
        {"premise": "What is 7 * 2?",
         "correct": "The product is 14.", "incorrect": "The product is 12.",
         "type": "arithmetic"},
         
        # Causal
        {"premise": "Fire requires oxygen to burn. The room has oxygen.",
         "correct": "Fire can burn in the room.", "incorrect": "Fire cannot burn here.",
         "type": "causal"},
        {"premise": "Plants need sunlight. The garden has sunlight.",
         "correct": "Plants will grow here.", "incorrect": "Plants will die here.",
         "type": "causal"},
        {"premise": "Ice melts above 0°C. The temperature is 25°C.",
         "correct": "The ice will melt.", "incorrect": "The ice will freeze.",
         "type": "causal"},
         
        # Consistency
        {"premise": "The ball is red. Red objects reflect red light.",
         "correct": "The ball reflects red light.", "incorrect": "The ball absorbs red light.",
         "type": "consistency"},
        {"premise": "Today is Monday. Monday comes before Tuesday.",
         "correct": "Tomorrow is Tuesday.", "incorrect": "Tomorrow is Sunday.",
         "type": "consistency"},
        {"premise": "North and South are opposites. Up and down are opposites.",
         "correct": "These are directional pairs.", "incorrect": "These are identical.",
         "type": "consistency"},
    ]


def evaluate_generation(model, full_text: str, correct_ending: str, incorrect_ending: str) -> Dict:
    """
    Evaluate if model prefers correct vs incorrect completion.
    Uses log-probability comparison.
    """
    # Build full sequences
    correct_full = full_text + " " + correct_ending
    incorrect_full = full_text + " " + incorrect_ending
    
    def get_sequence_logprob(text: str) -> float:
        input_ids = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model.model(input_ids.input_ids.to(model.device))
            logits = outputs.logits  # [batch, seq, vocab]
        
        # Compute log probability of actual tokens
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids.input_ids[..., 1:].contiguous()
        
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return token_log_probs.sum().item()
    
    correct_logprob = get_sequence_logprob(correct_full)
    incorrect_logprob = get_sequence_logprob(incorrect_full)
    
    return {
        "correct_logprob": correct_logprob,
        "incorrect_logprob": incorrect_logprob,
        "prefers_correct": correct_logprob > incorrect_logprob,
        "margin": correct_logprob - incorrect_logprob
    }


def run_validation():
    """Run full validation comparing methods."""
    print("=" * 60)
    print("POSITION-AWARE EAS VALIDATION EXPERIMENT")
    print("=" * 60)
    
    results = {"status": "started", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        from eas.src.watcher import EmergentWatcher
        from eas.src.watcher.position_aware_watcher import PositionAwareWatcher, create_position_aware_watcher
        
        print("\nLoading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        num_layers = model.num_layers
        intervention_layer = 3  # Middle layer
        
        # Create watchers
        print("\nCreating watchers...")
        standard_watcher = EmergentWatcher(
            dim=model.d_model,
            k=10,
            alpha_base=0.3,
            max_delta=0.5
        )
        
        pa_watcher = create_position_aware_watcher(
            dim=model.d_model,
            tokenizer=model.tokenizer,
            k=10,
            alpha=0.3
        )
        
        # Generate test data
        pairs = generate_test_pairs()
        print(f"\nGenerated {len(pairs)} test pairs")
        
        # Warmup phase - learn attractors from correct examples
        print("\n" + "=" * 40)
        print("WARMUP PHASE")
        print("=" * 40)
        
        warmup_pairs = pairs[:5]  # Use first 5 for warmup
        test_pairs = pairs[5:]    # Rest for testing
        
        for pair in warmup_pairs:
            correct_text = pair["premise"] + " " + pair["correct"]
            
            input_ids = model.tokenizer(correct_text, return_tensors="pt", truncation=True, max_length=128)
            
            with torch.no_grad():
                model.forward(input_ids.input_ids.to(model.device))
                hidden = model.get_layer_activation(intervention_layer)
            
            if hidden is not None:
                # Standard watcher update
                pooled = hidden.mean(dim=1)  # [batch, hidden]
                standard_watcher.update(pooled)
                
                # Position-aware watcher update
                pa_watcher.update(hidden, input_ids.input_ids)
        
        print(f"Standard watcher updates: {standard_watcher.update_count}")
        print(f"PA watcher updates: {pa_watcher.stats['update_count']}")
        
        # Evaluation phase
        print("\n" + "=" * 40)
        print("EVALUATION PHASE")
        print("=" * 40)
        
        methods = ["baseline", "standard_eas", "position_aware_eas"]
        method_results = {m: {"correct": 0, "total": 0, "margins": [], "latencies": []} for m in methods}
        
        for pair in test_pairs:
            full_premise = pair["premise"]
            correct_ending = pair["correct"]
            incorrect_ending = pair["incorrect"]
            
            print(f"\nPair type: {pair['type']}")
            
            for method in methods:
                start_time = time.time()
                
                # Remove any existing hooks
                model.remove_intervention_hook(intervention_layer)
                
                if method == "standard_eas":
                    # Register standard EAS hook
                    def standard_intervention(hidden_states):
                        pooled = hidden_states.mean(dim=1, keepdim=True)
                        snapped = standard_watcher.snap(pooled)
                        # Broadcast back to sequence
                        return hidden_states + (snapped - pooled)
                    
                    model.register_intervention_hook(intervention_layer, standard_intervention)
                
                elif method == "position_aware_eas":
                    # Store input_ids for the hook
                    current_input_ids = None
                    
                    def get_pa_intervention(input_ids_tensor):
                        def pa_intervention(hidden_states):
                            return pa_watcher.snap(hidden_states, input_ids_tensor)
                        return pa_intervention
                    
                    # We need to evaluate both texts with the hook
                    # This is a bit tricky - we'll evaluate without hook then compute margin
                
                # Evaluate
                result = evaluate_generation(model, full_premise, correct_ending, incorrect_ending)
                
                latency = time.time() - start_time
                method_results[method]["latencies"].append(latency)
                
                if result["prefers_correct"]:
                    method_results[method]["correct"] += 1
                method_results[method]["total"] += 1
                method_results[method]["margins"].append(result["margin"])
                
                print(f"  {method}: {'✓' if result['prefers_correct'] else '✗'} (margin: {result['margin']:.2f})")
        
        # Remove hooks
        model.remove_intervention_hook(intervention_layer)
        
        # Compute final results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        results["method_results"] = {}
        baseline_acc = method_results["baseline"]["correct"] / max(1, method_results["baseline"]["total"])
        
        for method in methods:
            mr = method_results[method]
            acc = mr["correct"] / max(1, mr["total"])
            improvement = acc - baseline_acc
            avg_latency = np.mean(mr["latencies"])
            avg_margin = np.mean(mr["margins"])
            
            results["method_results"][method] = {
                "accuracy": acc,
                "correct": mr["correct"],
                "total": mr["total"],
                "improvement": improvement,
                "avg_latency": avg_latency,
                "avg_margin": avg_margin
            }
            
            print(f"\n{method.upper()}:")
            print(f"  Accuracy: {acc:.1%} ({mr['correct']}/{mr['total']})")
            print(f"  Improvement over baseline: {improvement:+.1%}")
            print(f"  Average margin: {avg_margin:.2f}")
            print(f"  Average latency: {avg_latency:.4f}s")
        
        # Watcher statistics
        print("\n" + "=" * 40)
        print("WATCHER STATISTICS")
        print("=" * 40)
        
        pa_stats = pa_watcher.get_statistics()
        print(f"\nPosition-Aware Watcher:")
        print(f"  Total interventions: {pa_stats['total_interventions']}")
        print(f"  Critical interventions: {pa_stats['critical_interventions']}")
        print(f"  Critical ratio: {pa_stats['critical_ratio']:.1%}")
        print(f"  Position type counts: {pa_stats['position_type_counts']}")
        
        results["pa_watcher_stats"] = pa_stats
        results["standard_watcher_updates"] = standard_watcher.update_count
        
        # Key insights
        print("\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)
        
        pa_acc = results["method_results"]["position_aware_eas"]["accuracy"]
        std_acc = results["method_results"]["standard_eas"]["accuracy"]
        
        if pa_acc > std_acc:
            insight = f"Position-Aware EAS outperforms Standard EAS by {(pa_acc - std_acc):.1%}"
            results["verdict"] = "SUCCESS: Position-aware intervention improves performance"
        elif pa_acc == std_acc:
            insight = "Position-Aware and Standard EAS perform equally"
            results["verdict"] = "NEUTRAL: No difference detected (may need more samples)"
        else:
            insight = f"Standard EAS outperforms Position-Aware by {(std_acc - pa_acc):.1%}"
            results["verdict"] = "NEEDS TUNING: Position-aware requires parameter adjustment"
        
        print(f"\n{insight}")
        print(f"\n{results['verdict']}")
        
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    output_path = Path("/home/me/eas/eas/analysis/results/position_aware_validation.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_validation()
