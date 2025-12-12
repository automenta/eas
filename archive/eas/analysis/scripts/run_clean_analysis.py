#!/usr/bin/env python3
"""
Clean Premise-Conclusion Analysis
Test EAS on pairs where we KNOW the model must produce different activations:
1. Grammatical vs ungrammatical text
2. Semantically coherent vs incoherent continuations
3. Simple arithmetic correct vs incorrect

This establishes ground truth for whether EAS CAN work before testing complex logic.
"""
import json
import sys
import numpy as np
import torch
from typing import Dict, List
from scipy import stats

sys.path.insert(0, '/home/me/eas')


def generate_clean_pairs():
    """Generate pairs with clear right/wrong signal."""
    
    # Type 1: Grammatical vs ungrammatical
    grammar_pairs = [
        ("The cat sat on the mat.", "The cat sat on mat the."),
        ("She runs quickly through the park.", "She run quickly through park the."),
        ("They are going to the store.", "They is going to store the."),
        ("He writes beautiful poetry.", "He write beautiful poetry."),
        ("The children play in the garden.", "The children plays in garden the."),
    ]
    
    # Type 2: Semantically coherent vs incoherent
    semantic_pairs = [
        ("The sun rises in the east.", "The sun rises in the elephant."),
        ("Water is essential for life.", "Water is essential for purple."),
        ("Birds fly through the sky.", "Birds fly through the happiness."),
        ("She read the interesting book.", "She read the interesting mountain."),
        ("The train arrives at the station.", "The train arrives at the flavor."),
    ]
    
    # Type 3: Simple arithmetic (correct vs incorrect continuation)
    arithmetic_pairs = [
        ("2 + 2 = 4", "2 + 2 = 5"),
        ("3 x 3 = 9", "3 x 3 = 8"),
        ("10 - 5 = 5", "10 - 5 = 6"),
        ("8 / 2 = 4", "8 / 2 = 3"),
        ("1 + 1 = 2", "1 + 1 = 3"),
    ]
    
    return {
        "grammar": grammar_pairs,
        "semantic": semantic_pairs,
        "arithmetic": arithmetic_pairs,
    }


def run_clean_analysis():
    """Run analysis on clean pairs where model MUST differ."""
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        
        print("Loading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        hidden_dim = model.model.config.hidden_size
        num_layers = model.model.config.num_hidden_layers
        print(f"Model: {num_layers} layers, {hidden_dim}d")
        
        pairs_by_type = generate_clean_pairs()
        
        for pair_type, pairs in pairs_by_type.items():
            print(f"\n=== {pair_type.upper()} PAIRS ===")
            
            layer_results = {}
            
            for layer in range(num_layers):
                correct_acts = []
                incorrect_acts = []
                
                for correct, incorrect in pairs:
                    correct_ids = model.tokenizer(correct, return_tensors="pt").input_ids
                    incorrect_ids = model.tokenizer(incorrect, return_tensors="pt").input_ids
                    
                    with torch.no_grad():
                        model.forward(correct_ids)
                        correct_hidden = model.get_layer_activation(layer)
                        
                        model.forward(incorrect_ids)
                        incorrect_hidden = model.get_layer_activation(layer)
                    
                    if correct_hidden is not None and incorrect_hidden is not None:
                        correct_pooled = correct_hidden.mean(dim=1).cpu().numpy().squeeze()
                        incorrect_pooled = incorrect_hidden.mean(dim=1).cpu().numpy().squeeze()
                        
                        correct_acts.append(correct_pooled)
                        incorrect_acts.append(incorrect_pooled)
                
                correct_arr = np.vstack(correct_acts)
                incorrect_arr = np.vstack(incorrect_acts)
                
                # Compute similarity
                correct_centroid = correct_arr.mean(axis=0)
                incorrect_centroid = incorrect_arr.mean(axis=0)
                
                cosine_sim = np.dot(correct_centroid, incorrect_centroid) / (
                    np.linalg.norm(correct_centroid) * np.linalg.norm(incorrect_centroid) + 1e-8
                )
                
                euclidean_dist = np.linalg.norm(correct_centroid - incorrect_centroid)
                
                # Norm differences
                correct_norms = [np.linalg.norm(a) for a in correct_arr]
                incorrect_norms = [np.linalg.norm(a) for a in incorrect_arr]
                t_stat, p_value = stats.ttest_ind(correct_norms, incorrect_norms)
                
                layer_results[layer] = {
                    "cosine_similarity": float(cosine_sim),
                    "euclidean_distance": float(euclidean_dist),
                    "norm_t_stat": float(t_stat),
                    "norm_p_value": float(p_value),
                }
            
            # Find best layer for this pair type
            best_layer = min(layer_results.keys(), 
                           key=lambda l: layer_results[l]["cosine_similarity"])
            
            print(f"  Best layer (lowest similarity): {best_layer}")
            print(f"  Cosine similarity: {layer_results[best_layer]['cosine_similarity']:.6f}")
            print(f"  Euclidean distance: {layer_results[best_layer]['euclidean_distance']:.4f}")
            print(f"  Norm p-value: {layer_results[best_layer]['norm_p_value']:.4f}")
            
            # Show all layers
            print(f"  Layer similarities: ", end="")
            for l in range(num_layers):
                sim = layer_results[l]["cosine_similarity"]
                print(f"L{l}:{sim:.4f} ", end="")
            print()
            
            results[pair_type] = {
                "layer_results": layer_results,
                "best_layer": best_layer,
            }
        
        # Overall assessment
        print("\n=== OVERALL ASSESSMENT ===")
        
        max_divergence = 0
        best_type = None
        best_layer_overall = None
        
        for pair_type in pairs_by_type.keys():
            for layer, res in results[pair_type]["layer_results"].items():
                divergence = 1 - res["cosine_similarity"]
                if divergence > max_divergence:
                    max_divergence = divergence
                    best_type = pair_type
                    best_layer_overall = layer
        
        print(f"Maximum divergence: {max_divergence:.6f}")
        print(f"Found in: {best_type} pairs, layer {best_layer_overall}")
        
        if max_divergence > 0.01:
            verdict = "SIGNAL FOUND: EAS can leverage this contrast"
        elif max_divergence > 0.001:
            verdict = "WEAK SIGNAL: Some differentiation exists"
        else:
            verdict = "NO SIGNAL: Model representations nearly identical"
        
        print(f"\nVERDICT: {verdict}")
        
        results["max_divergence"] = float(max_divergence)
        results["best_type"] = best_type
        results["best_layer"] = best_layer_overall
        results["verdict"] = verdict
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    with open("eas/analysis/results/clean_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to clean_analysis_results.json")
    return results


if __name__ == "__main__":
    run_clean_analysis()
