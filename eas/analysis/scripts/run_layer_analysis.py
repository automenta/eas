#!/usr/bin/env python3
"""
Multi-Layer Activation Analysis
Find where success/failure activations DIVERGE the most across layers.
This identifies the optimal intervention layer for EAS.

Key insight: If layer 3 shows 0.98 similarity, we need to find where
the reasoning "goes wrong" - that's where EAS should intervene.
"""
import json
import sys
import numpy as np
import torch
from typing import Dict, List

sys.path.insert(0, '/home/me/eas')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_layer_analysis():
    """Analyze activation divergence across all layers."""
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        from eas.src.datasets.paired_dataset import PairedDatasetGenerator
        
        print("Loading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        
        # Get number of layers
        num_layers = model.model.config.num_hidden_layers
        hidden_dim = model.model.config.hidden_size
        print(f"Model: {num_layers} layers, {hidden_dim}d")
        
        results["num_layers"] = num_layers
        results["hidden_dim"] = hidden_dim
        
        # Generate paired data
        gen = PairedDatasetGenerator()
        pairs = gen.generate_dataset(20)
        
        # For each layer, track success/failure separation
        layer_stats = {}
        
        print("\nAnalyzing activation divergence per layer...")
        
        for layer in range(num_layers):
            success_acts = []
            failure_acts = []
            
            for pair in pairs[:10]:
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
                    success_pooled = success_hidden.mean(dim=1).cpu().numpy().squeeze()
                    failure_pooled = failure_hidden.mean(dim=1).cpu().numpy().squeeze()
                    
                    success_acts.append(success_pooled)
                    failure_acts.append(failure_pooled)
            
            success_arr = np.vstack(success_acts)
            failure_arr = np.vstack(failure_acts)
            
            # Compute metrics for this layer
            success_centroid = success_arr.mean(axis=0)
            failure_centroid = failure_arr.mean(axis=0)
            
            # 1. Centroid divergence (how far apart are success/failure centers?)
            centroid_distance = np.linalg.norm(success_centroid - failure_centroid)
            centroid_similarity = cosine_similarity(success_centroid, failure_centroid)
            
            # 2. Within-class coherence (are successes similar to each other?)
            success_coherence = np.mean([cosine_similarity(s, success_centroid) for s in success_arr])
            failure_coherence = np.mean([cosine_similarity(f, failure_centroid) for f in failure_arr])
            
            # 3. Cross-class similarity (do success/failure examples overlap?)
            cross_similarity = np.mean([
                cosine_similarity(s, failure_centroid) for s in success_arr
            ])
            
            # 4. Separation score: divergence / overlap
            separation = (1 - centroid_similarity) / (cross_similarity + 0.01)
            
            layer_stats[layer] = {
                "centroid_distance": float(centroid_distance),
                "centroid_similarity": float(centroid_similarity),
                "success_coherence": float(success_coherence),
                "failure_coherence": float(failure_coherence),
                "cross_similarity": float(cross_similarity),
                "separation_score": float(separation),
            }
            
            print(f"  Layer {layer:2d}: dist={centroid_distance:.4f}, "
                  f"sim={centroid_similarity:.4f}, sep={separation:.4f}")
        
        # Find best intervention layer
        best_layer = max(layer_stats.keys(), key=lambda l: layer_stats[l]["separation_score"])
        worst_layer = min(layer_stats.keys(), key=lambda l: layer_stats[l]["separation_score"])
        
        print(f"\n=== LAYER ANALYSIS SUMMARY ===")
        print(f"Best intervention layer: {best_layer} (separation={layer_stats[best_layer]['separation_score']:.4f})")
        print(f"Worst intervention layer: {worst_layer} (separation={layer_stats[worst_layer]['separation_score']:.4f})")
        
        results["layer_stats"] = layer_stats
        results["best_intervention_layer"] = best_layer
        results["worst_intervention_layer"] = worst_layer
        results["status"] = "success"
        
        # Visualization-ready data
        layers = list(range(num_layers))
        separations = [layer_stats[l]["separation_score"] for l in layers]
        distances = [layer_stats[l]["centroid_distance"] for l in layers]
        
        print(f"\nSeparation scores by layer: {[f'{s:.3f}' for s in separations]}")
        print(f"Centroid distances by layer: {[f'{d:.3f}' for d in distances]}")
        
        results["separation_profile"] = separations
        results["distance_profile"] = distances
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    with open("eas/analysis/results/layer_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to layer_analysis_results.json")
    return results


if __name__ == "__main__":
    run_layer_analysis()
