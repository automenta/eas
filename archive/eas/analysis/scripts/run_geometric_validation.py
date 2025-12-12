#!/usr/bin/env python3
"""
Geometric Validation for EAS
Measures the GEOMETRIC effectiveness of EAS independent of model task accuracy.

Key insight: A 70M model can't do NLI well, but EAS can still create
geometric structure that WOULD improve a more capable model.

We measure:
1. Activation separation: Do success/failure activations become more separable?
2. Attractor alignment: Do attractors align with the contrast direction?
3. Snapping effectiveness: Does snapping move activations toward correct regions?
"""
import json
import sys
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cosine

sys.path.insert(0, '/home/me/eas')


def compute_separation_score(success_acts: np.ndarray, failure_acts: np.ndarray) -> float:
    """
    Compute how well-separated success and failure activations are.
    Uses silhouette score (higher = better separation).
    """
    if len(success_acts) < 2 or len(failure_acts) < 2:
        return 0.0
    
    # Combine with labels
    X = np.vstack([success_acts, failure_acts])
    labels = np.array([1] * len(success_acts) + [0] * len(failure_acts))
    
    try:
        score = silhouette_score(X, labels)
        return float(score)
    except:
        return 0.0


def compute_attractor_alignment(attractors: np.ndarray, 
                                  success_acts: np.ndarray,
                                  failure_acts: np.ndarray) -> Dict[str, float]:
    """
    Measure how well attractors align with the correct reasoning direction.
    """
    # Compute centroids
    success_centroid = success_acts.mean(axis=0)
    failure_centroid = failure_acts.mean(axis=0)
    
    # The "correct direction" is from failure to success
    correct_direction = success_centroid - failure_centroid
    correct_direction = correct_direction / (np.linalg.norm(correct_direction) + 1e-8)
    
    # Measure alignment of each attractor with the correct direction
    alignments = []
    for attractor in attractors:
        att_norm = attractor / (np.linalg.norm(attractor) + 1e-8)
        alignment = np.dot(att_norm, correct_direction)
        alignments.append(alignment)
    
    return {
        "max_alignment": float(np.max(alignments)),
        "mean_alignment": float(np.mean(alignments)),
        "best_attractor": int(np.argmax(alignments)),
        "positive_count": int(np.sum(np.array(alignments) > 0)),
    }


def compute_snapping_improvement(before_acts: np.ndarray,
                                   after_acts: np.ndarray, 
                                   success_centroid: np.ndarray) -> Dict[str, float]:
    """
    Measure whether snapping moves activations TOWARD the success region.
    """
    # Distances before and after snapping
    before_distances = []
    after_distances = []
    
    for before, after in zip(before_acts, after_acts):
        before_dist = 1 - cosine(before, success_centroid)  # Similarity
        after_dist = 1 - cosine(after, success_centroid)
        before_distances.append(before_dist)
        after_distances.append(after_dist)
    
    before_mean = np.mean(before_distances)
    after_mean = np.mean(after_distances)
    
    return {
        "before_similarity": float(before_mean),
        "after_similarity": float(after_mean),
        "improvement": float(after_mean - before_mean),
        "improved_fraction": float(np.mean(np.array(after_distances) > np.array(before_distances))),
    }


def run_geometric_validation():
    """Run geometric validation with Pythia-70m."""
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        from eas.src.watcher.contrastive_watcher import ContrastiveWatcher
        from eas.src.watcher import EmergentWatcher
        from eas.src.datasets.paired_dataset import PairedDatasetGenerator
        
        print("Loading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        hidden_dim = model.model.config.hidden_size
        print(f"Model loaded: {hidden_dim}d")
        
        # Create watchers
        contrastive = ContrastiveWatcher(dim=hidden_dim, k=10)
        standard = EmergentWatcher(dim=hidden_dim, k=10)
        
        # Generate data
        gen = PairedDatasetGenerator()
        pairs = gen.generate_dataset(30)
        
        # Collect activations during warmup
        success_activations = []
        failure_activations = []
        
        print("Collecting activations during warmup...")
        for pair in pairs[:15]:
            success_text, failure_text = gen.format_for_training(pair)
            
            success_ids = model.tokenizer(success_text, return_tensors="pt",
                                          truncation=True, max_length=64).input_ids
            failure_ids = model.tokenizer(failure_text, return_tensors="pt",
                                          truncation=True, max_length=64).input_ids
            
            with torch.no_grad():
                model.forward(success_ids)
                success_hidden = model.get_layer_activation(3)
                
                model.forward(failure_ids)
                failure_hidden = model.get_layer_activation(3)
            
            if success_hidden is not None and failure_hidden is not None:
                success_pooled = success_hidden.mean(dim=1).cpu().numpy()
                failure_pooled = failure_hidden.mean(dim=1).cpu().numpy()
                
                success_activations.append(success_pooled.squeeze())
                failure_activations.append(failure_pooled.squeeze())
                
                # Update watchers
                contrastive.update_contrastive(success_hidden, failure_hidden)
                standard.update(success_hidden)
        
        success_acts = np.vstack(success_activations)
        failure_acts = np.vstack(failure_activations)
        
        print(f"Collected {len(success_acts)} success and {len(failure_acts)} failure activations")
        
        # ------ GEOMETRIC METRICS ------
        
        # 1. Separation score (before any clustering would affect it)
        raw_separation = compute_separation_score(success_acts, failure_acts)
        print(f"\n1. Raw activation separation: {raw_separation:.4f}")
        results["raw_separation"] = raw_separation
        
        # 2. Attractor alignment
        contrastive_attractors = contrastive.attractor_memory.positive_attractors.detach().cpu().numpy()
        alignment = compute_attractor_alignment(contrastive_attractors, success_acts, failure_acts)
        print(f"2. Attractor alignment with correct direction:")
        print(f"   Max alignment: {alignment['max_alignment']:.4f}")
        print(f"   Mean alignment: {alignment['mean_alignment']:.4f}")
        print(f"   Attractors pointing toward success: {alignment['positive_count']}/10")
        results["attractor_alignment"] = alignment
        
        # 3. Snapping effectiveness test
        print("\n3. Testing snapping effectiveness...")
        success_centroid = success_acts.mean(axis=0)
        
        before_snaps = []
        after_snaps_contrastive = []
        after_snaps_standard = []
        
        for pair in pairs[15:25]:  # Test on new pairs
            test_text, _ = gen.format_for_training(pair)
            input_ids = model.tokenizer(test_text, return_tensors="pt",
                                        truncation=True, max_length=64).input_ids
            
            with torch.no_grad():
                model.forward(input_ids)
                hidden = model.get_layer_activation(3)
            
            if hidden is not None:
                # Before snapping
                before = hidden.mean(dim=1).detach().cpu().numpy().squeeze()
                before_snaps.append(before)
                
                # After contrastive snapping
                snapped_c = contrastive.snap(hidden)
                after_c = snapped_c.mean(dim=1).detach().cpu().numpy().squeeze()
                after_snaps_contrastive.append(after_c)
                
                # After standard snapping
                snapped_s = standard.snap(hidden)
                after_s = snapped_s.mean(dim=1).detach().cpu().numpy().squeeze()
                after_snaps_standard.append(after_s)
        
        before_arr = np.vstack(before_snaps)
        after_contrastive_arr = np.vstack(after_snaps_contrastive)
        after_standard_arr = np.vstack(after_snaps_standard)
        
        contrastive_improvement = compute_snapping_improvement(
            before_arr, after_contrastive_arr, success_centroid
        )
        standard_improvement = compute_snapping_improvement(
            before_arr, after_standard_arr, success_centroid
        )
        
        print(f"   Contrastive snapping:")
        print(f"     Before similarity to success: {contrastive_improvement['before_similarity']:.4f}")
        print(f"     After similarity to success:  {contrastive_improvement['after_similarity']:.4f}")
        print(f"     Improvement: {contrastive_improvement['improvement']:+.4f}")
        
        print(f"   Standard snapping:")
        print(f"     Before similarity to success: {standard_improvement['before_similarity']:.4f}")
        print(f"     After similarity to success:  {standard_improvement['after_similarity']:.4f}")
        print(f"     Improvement: {standard_improvement['improvement']:+.4f}")
        
        results["contrastive_snapping"] = contrastive_improvement
        results["standard_snapping"] = standard_improvement
        
        # 4. Anti-attractor effectiveness (contrastive only)
        anti_attractors = contrastive.attractor_memory.anti_attractors.detach().cpu().numpy()
        anti_alignment = compute_attractor_alignment(anti_attractors, failure_acts, success_acts)
        print(f"\n4. Anti-attractor alignment (should point toward failures):")
        print(f"   Max alignment: {anti_alignment['max_alignment']:.4f}")
        print(f"   Attractors pointing toward failure: {anti_alignment['positive_count']}/10")
        results["anti_attractor_alignment"] = anti_alignment
        
        # 5. Summary metrics
        contrastive_stats = contrastive.get_statistics()
        results["contrastive_stats"] = contrastive_stats
        
        # Compute overall score
        overall_score = (
            0.3 * (1 + alignment['mean_alignment']) / 2 +  # Alignment: [-1,1] -> [0,1]
            0.3 * max(0, contrastive_improvement['improvement'] * 10) +  # Snapping improvement
            0.2 * (1 + anti_alignment['max_alignment']) / 2 +  # Anti-alignment
            0.2 * contrastive_stats['attractor_entropy']  # Diversity
        )
        
        print(f"\n=== OVERALL GEOMETRIC SCORE: {overall_score:.4f} ===")
        results["overall_geometric_score"] = float(overall_score)
        
        # Verdict
        if overall_score > 0.6:
            verdict = "STRONG: EAS creates effective geometric structure"
        elif overall_score > 0.4:
            verdict = "MODERATE: EAS shows promising geometric effects"
        else:
            verdict = "WEAK: Geometric effects not yet significant"
        
        print(f"VERDICT: {verdict}")
        results["verdict"] = verdict
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    with open("eas/analysis/results/geometric_validation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nResults saved to geometric_validation_results.json")
    return results


if __name__ == "__main__":
    run_geometric_validation()
