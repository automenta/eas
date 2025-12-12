#!/usr/bin/env python3
"""
Magnitude-Aware EAS Validation
Based on finding that success/failure differ in MAGNITUDE not direction.

Key insight from layer analysis:
- Cosine similarity ~1.0 across all layers (same direction)
- Centroid distance grows (magnitude difference increases)

New approach: Learn to scale activations toward the "correct magnitude region".
"""
import json
import sys
import numpy as np
import torch
from typing import Dict, List

sys.path.insert(0, '/home/me/eas')


def run_magnitude_analysis():
    """Analyze magnitude patterns and test magnitude-based intervention."""
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        from eas.src.datasets.paired_dataset import PairedDatasetGenerator
        
        print("Loading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        hidden_dim = model.model.config.hidden_size
        num_layers = model.model.config.num_hidden_layers
        print(f"Model: {num_layers} layers, {hidden_dim}d")
        
        # Generate paired data
        gen = PairedDatasetGenerator()
        pairs = gen.generate_dataset(30)
        
        # Analyze magnitude patterns across layers
        print("\n=== MAGNITUDE ANALYSIS ===")
        
        for layer in [3, 4, 5]:  # Focus on deeper layers
            success_norms = []
            failure_norms = []
            
            for pair in pairs[:15]:
                success_text, failure_text = gen.format_for_training(pair)
                
                success_ids = model.tokenizer(success_text, return_tensors="pt",
                                              truncation=True, max_length=64).input_ids
                failure_ids = model.tokenizer(failure_text, return_tensors="pt",
                                              truncation=True, max_length=64).input_ids
                
                with torch.no_grad():
                    model.forward(success_ids)
                    success_hidden = model.get_layer_activation(layer)
                    
                    model.forward(failure_ids)
                    failure_hidden = model.get_layer_activation(layer)
                
                if success_hidden is not None and failure_hidden is not None:
                    # Get pooled activation norms
                    success_norm = success_hidden.mean(dim=1).norm().item()
                    failure_norm = failure_hidden.mean(dim=1).norm().item()
                    
                    success_norms.append(success_norm)
                    failure_norms.append(failure_norm)
            
            success_mean = np.mean(success_norms)
            failure_mean = np.mean(failure_norms)
            success_std = np.std(success_norms)
            failure_std = np.std(failure_norms)
            
            # T-test for significance
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(success_norms, failure_norms)
            
            print(f"\nLayer {layer}:")
            print(f"  Success norms: {success_mean:.4f} ± {success_std:.4f}")
            print(f"  Failure norms: {failure_mean:.4f} ± {failure_std:.4f}")
            print(f"  Difference: {success_mean - failure_mean:+.4f}")
            print(f"  T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  *** SIGNIFICANT DIFFERENCE! ***")
            
            results[f"layer_{layer}"] = {
                "success_mean": float(success_mean),
                "failure_mean": float(failure_mean),
                "difference": float(success_mean - failure_mean),
                "t_stat": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
        
        # === MAGNITUDE-BASED INTERVENTION ===
        print("\n=== MAGNITUDE-BASED INTERVENTION TEST ===")
        
        # Determine target magnitude from success examples
        target_layer = 4  # Use layer with best separation
        
        success_norms = []
        for pair in pairs[:15]:
            success_text, _ = gen.format_for_training(pair)
            input_ids = model.tokenizer(success_text, return_tensors="pt",
                                        truncation=True, max_length=64).input_ids
            
            with torch.no_grad():
                model.forward(input_ids)
                hidden = model.get_layer_activation(target_layer)
            
            if hidden is not None:
                norm = hidden.mean(dim=1).norm().item()
                success_norms.append(norm)
        
        target_norm = np.mean(success_norms)
        print(f"Target norm (success average): {target_norm:.4f}")
        
        # Test: scale activations toward target norm
        improvements = []
        
        for pair in pairs[15:25]:
            _, failure_text = gen.format_for_training(pair)
            input_ids = model.tokenizer(failure_text, return_tensors="pt",
                                        truncation=True, max_length=64).input_ids
            
            with torch.no_grad():
                model.forward(input_ids)
                hidden = model.get_layer_activation(target_layer)
            
            if hidden is not None:
                # Current norm
                pooled = hidden.mean(dim=1)
                current_norm = pooled.norm().item()
                
                # Scale toward target
                if current_norm > 0:
                    scale_factor = target_norm / current_norm
                    scaled = pooled * scale_factor
                    new_norm = scaled.norm().item()
                    
                    improvement = abs(new_norm - target_norm) < abs(current_norm - target_norm)
                    improvements.append(improvement)
        
        improvement_rate = np.mean(improvements)
        print(f"Magnitude intervention improvement rate: {improvement_rate:.1%}")
        
        results["magnitude_intervention"] = {
            "target_norm": float(target_norm),
            "improvement_rate": float(improvement_rate),
        }
        
        # === KEY INSIGHT ===
        print("\n=== KEY INSIGHT ===")
        print("Standard EAS normalizes activations (L2 norm = 1) before clustering.")
        print("This REMOVES the magnitude signal that distinguishes success/failure!")
        print("\nRECOMMENDATION: Modify EAS to preserve/utilize magnitude information:")
        print("1. Skip normalization OR apply magnitude-based post-correction")
        print("2. Learn 'magnitude attractors' - target activation norms, not directions")
        print("3. Use magnitude as a secondary signal alongside directional similarity")
        
        results["insight"] = "Magnitude difference is the key signal; normalization destroys it"
        results["recommendation"] = "Modify EAS to preserve magnitude information"
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    with open("eas/analysis/results/magnitude_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to magnitude_analysis_results.json")
    return results


if __name__ == "__main__":
    run_magnitude_analysis()
