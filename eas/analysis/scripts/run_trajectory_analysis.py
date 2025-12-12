#!/usr/bin/env python3
"""
Cross-Layer Trajectory Analysis

KEY INSIGHT: Even if endpoints are similar, the PATH through activation space
may differ between correct and incorrect reasoning.

This analysis tracks the geometric trajectory of activations through layers,
computing curvature, velocity, and other path properties.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from scipy import stats
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/me/eas')


@dataclass
class TrajectoryMetrics:
    """Metrics describing a trajectory through activation space."""
    arc_length: float  # Total distance traveled
    displacement: float  # Straight-line distance start to end
    curvature: float  # How curved is the path
    mean_velocity: float  # Average speed per layer
    velocity_variance: float  # Consistency of velocity
    direction_changes: float  # How much does direction change
    layer_magnitudes: List[float]  # Magnitude at each layer


def generate_trajectory_pairs() -> List[Dict]:
    """Generate pairs for trajectory analysis."""
    return [
        {
            "correct": "Given that all birds fly. A sparrow is a bird. Therefore sparrows fly.",
            "incorrect": "Given that all birds fly. A sparrow is a bird. Therefore sparrows swim.",
            "type": "syllogism"
        },
        {
            "correct": "If the ground is wet then it rained. The ground is wet. It rained.",
            "incorrect": "If the ground is wet then it rained. The ground is wet. It did not rain.",
            "type": "modus_ponens"
        },
        {
            "correct": "The temperature is 100C. Water boils at 100C. The water is boiling.",
            "incorrect": "The temperature is 100C. Water boils at 100C. The water is freezing.",
            "type": "physical_reasoning"
        },
        {
            "correct": "1 plus 1 equals 2. This is mathematically true.",
            "incorrect": "1 plus 1 equals 3. This is mathematically true.",
            "type": "arithmetic"
        },
        {
            "correct": "Mammals are warm-blooded. Dogs are mammals. Dogs are warm-blooded.",
            "incorrect": "Mammals are warm-blooded. Dogs are mammals. Dogs are cold-blooded.",
            "type": "category_inheritance"
        },
        {
            "correct": "If A implies B and B implies C then A implies C. This is valid logic.",
            "incorrect": "If A implies B and B implies C then A implies not C. This is valid logic.",
            "type": "transitivity"
        },
        {
            "correct": "Plants need sunlight. The room has sunlight. Plants can grow here.",
            "incorrect": "Plants need sunlight. The room has sunlight. Plants cannot grow here.",
            "type": "requirements"
        },
        {
            "correct": "Parallel lines never meet. These lines are parallel. They never meet.",
            "incorrect": "Parallel lines never meet. These lines are parallel. They will meet.",
            "type": "geometry"
        }
    ]


def get_layer_trajectory(model, text: str) -> List[torch.Tensor]:
    """Extract activation at each layer for a text."""
    input_ids = model.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128
    )
    
    with torch.no_grad():
        model.forward(input_ids.input_ids.to(model.device))
    
    trajectory = []
    for layer_idx in range(model.num_layers):
        hidden = model.get_layer_activation(layer_idx)
        if hidden is not None:
            # Pool to single vector per layer
            pooled = hidden.mean(dim=1).squeeze(0).cpu()  # [hidden_dim]
            trajectory.append(pooled)
    
    return trajectory


def compute_trajectory_metrics(trajectory: List[torch.Tensor]) -> TrajectoryMetrics:
    """Compute geometric metrics for a trajectory."""
    if len(trajectory) < 2:
        return None
    
    # Arc length: sum of distances between consecutive points
    arc_length = 0.0
    velocities = []
    direction_changes = []
    magnitudes = []
    
    prev_direction = None
    
    for i, point in enumerate(trajectory):
        magnitudes.append(torch.norm(point).item())
        
        if i > 0:
            # Velocity (distance to previous point)
            velocity = torch.norm(point - trajectory[i-1]).item()
            velocities.append(velocity)
            arc_length += velocity
            
            # Direction change
            if i > 1:
                prev_step = trajectory[i-1] - trajectory[i-2]
                curr_step = point - trajectory[i-1]
                
                # Cosine of angle between steps
                cos_angle = torch.nn.functional.cosine_similarity(
                    prev_step.unsqueeze(0), curr_step.unsqueeze(0)
                ).item()
                # Direction change = 1 - cos (0 = same direction, 2 = opposite)
                direction_changes.append(1 - cos_angle)
    
    # Displacement: distance from start to end
    displacement = torch.norm(trajectory[-1] - trajectory[0]).item()
    
    # Curvature: arc_length / displacement (1 = straight line, > 1 = curved)
    curvature = arc_length / displacement if displacement > 0 else float('inf')
    
    return TrajectoryMetrics(
        arc_length=arc_length,
        displacement=displacement,
        curvature=curvature,
        mean_velocity=np.mean(velocities) if velocities else 0,
        velocity_variance=np.var(velocities) if velocities else 0,
        direction_changes=np.mean(direction_changes) if direction_changes else 0,
        layer_magnitudes=magnitudes
    )


def compare_trajectories(correct_traj: List[torch.Tensor],
                         incorrect_traj: List[torch.Tensor]) -> Dict:
    """Compare two trajectories."""
    
    c_metrics = compute_trajectory_metrics(correct_traj)
    ic_metrics = compute_trajectory_metrics(incorrect_traj)
    
    if c_metrics is None or ic_metrics is None:
        return None
    
    # Point-by-point comparison
    point_distances = []
    point_cosines = []
    
    min_len = min(len(correct_traj), len(incorrect_traj))
    
    for i in range(min_len):
        c_point = correct_traj[i]
        ic_point = incorrect_traj[i]
        
        dist = torch.norm(c_point - ic_point).item()
        point_distances.append(dist)
        
        cos_sim = torch.nn.functional.cosine_similarity(
            c_point.unsqueeze(0), ic_point.unsqueeze(0)
        ).item()
        point_cosines.append(cos_sim)
    
    return {
        "correct_metrics": {
            "arc_length": c_metrics.arc_length,
            "displacement": c_metrics.displacement,
            "curvature": c_metrics.curvature,
            "mean_velocity": c_metrics.mean_velocity,
            "velocity_variance": c_metrics.velocity_variance,
            "direction_changes": c_metrics.direction_changes, 
            "final_magnitude": c_metrics.layer_magnitudes[-1] if c_metrics.layer_magnitudes else 0
        },
        "incorrect_metrics": {
            "arc_length": ic_metrics.arc_length,
            "displacement": ic_metrics.displacement,
            "curvature": ic_metrics.curvature,
            "mean_velocity": ic_metrics.mean_velocity,
            "velocity_variance": ic_metrics.velocity_variance,
            "direction_changes": ic_metrics.direction_changes,
            "final_magnitude": ic_metrics.layer_magnitudes[-1] if ic_metrics.layer_magnitudes else 0
        },
        "differences": {
            "arc_length_diff": c_metrics.arc_length - ic_metrics.arc_length,
            "curvature_diff": c_metrics.curvature - ic_metrics.curvature,
            "velocity_diff": c_metrics.mean_velocity - ic_metrics.mean_velocity,
            "direction_changes_diff": c_metrics.direction_changes - ic_metrics.direction_changes
        },
        "trajectory_comparison": {
            "mean_point_distance": np.mean(point_distances),
            "max_point_distance": np.max(point_distances),
            "mean_point_cosine": np.mean(point_cosines),
            "min_point_cosine": np.min(point_cosines),
            "point_distances_by_layer": point_distances,
            "point_cosines_by_layer": point_cosines
        }
    }


def run_trajectory_analysis():
    """Main analysis function."""
    print("=" * 60)
    print("CROSS-LAYER TRAJECTORY ANALYSIS")
    print("=" * 60)
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        
        print("\nLoading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        num_layers = model.num_layers
        print(f"Model: {num_layers} layers")
        
        pairs = generate_trajectory_pairs()
        print(f"Generated {len(pairs)} pairs for trajectory analysis")
        
        all_comparisons = []
        
        for pair in pairs:
            print(f"\nAnalyzing: {pair['type']}")
            
            correct_traj = get_layer_trajectory(model, pair["correct"])
            incorrect_traj = get_layer_trajectory(model, pair["incorrect"])
            
            if correct_traj and incorrect_traj:
                comparison = compare_trajectories(correct_traj, incorrect_traj)
                if comparison:
                    comparison["pair_type"] = pair["type"]
                    all_comparisons.append(comparison)
                    
                    cm = comparison["correct_metrics"]
                    im = comparison["incorrect_metrics"]
                    td = comparison["differences"]
                    tc = comparison["trajectory_comparison"]
                    
                    print(f"  Correct curvature: {cm['curvature']:.4f}, Incorrect: {im['curvature']:.4f}")
                    print(f"  Curvature difference: {td['curvature_diff']:.4f}")
                    print(f"  Mean trajectory distance: {tc['mean_point_distance']:.4f}")
        
        # Aggregate analysis
        print("\n" + "=" * 60)
        print("AGGREGATE TRAJECTORY FINDINGS")
        print("=" * 60)
        
        curvature_diffs = []
        arc_length_diffs = []
        direction_diffs = []
        mean_distances = []
        min_cosines = []
        
        # Layer-by-layer divergence
        layer_distances = {i: [] for i in range(num_layers)}
        layer_cosines = {i: [] for i in range(num_layers)}
        
        for comp in all_comparisons:
            curvature_diffs.append(comp["differences"]["curvature_diff"])
            arc_length_diffs.append(comp["differences"]["arc_length_diff"])
            direction_diffs.append(comp["differences"]["direction_changes_diff"])
            mean_distances.append(comp["trajectory_comparison"]["mean_point_distance"])
            min_cosines.append(comp["trajectory_comparison"]["min_point_cosine"])
            
            for i, dist in enumerate(comp["trajectory_comparison"]["point_distances_by_layer"]):
                if i < num_layers:
                    layer_distances[i].append(dist)
            for i, cos in enumerate(comp["trajectory_comparison"]["point_cosines_by_layer"]):
                if i < num_layers:
                    layer_cosines[i].append(cos)
        
        # Summary statistics
        results["aggregate"] = {
            "mean_curvature_diff": np.mean(curvature_diffs),
            "std_curvature_diff": np.std(curvature_diffs),
            "mean_arc_length_diff": np.mean(arc_length_diffs),
            "mean_direction_changes_diff": np.mean(direction_diffs),
            "mean_trajectory_distance": np.mean(mean_distances),
            "min_trajectory_cosine": np.min(min_cosines),
            "mean_min_cosine": np.mean(min_cosines)
        }
        
        print(f"Mean curvature difference: {results['aggregate']['mean_curvature_diff']:.4f}")
        print(f"Mean trajectory distance: {results['aggregate']['mean_trajectory_distance']:.4f}")
        print(f"Minimum cosine similarity: {results['aggregate']['min_trajectory_cosine']:.4f}")
        
        # Layer-by-layer summary
        print("\nLayer-by-layer trajectory divergence:")
        results["layer_divergence"] = {}
        
        max_layer_divergence = 0
        most_divergent_layer = 0
        
        for i in range(num_layers):
            if layer_distances[i]:
                mean_dist = np.mean(layer_distances[i])
                mean_cos = np.mean(layer_cosines[i])
                divergence = 1 - mean_cos
                
                results["layer_divergence"][str(i)] = {
                    "mean_distance": mean_dist,
                    "mean_cosine": mean_cos,
                    "divergence": divergence
                }
                
                print(f"  Layer {i}: dist={mean_dist:.4f}, cos={mean_cos:.4f}, div={divergence:.4f}")
                
                if divergence > max_layer_divergence:
                    max_layer_divergence = divergence
                    most_divergent_layer = i
        
        # Statistical tests on curvature
        print("\nStatistical Analysis:")
        
        correct_curvatures = [c["correct_metrics"]["curvature"] for c in all_comparisons]
        incorrect_curvatures = [c["incorrect_metrics"]["curvature"] for c in all_comparisons]
        
        t_stat, p_value = stats.ttest_rel(correct_curvatures, incorrect_curvatures)
        results["curvature_ttest"] = {"t_stat": t_stat, "p_value": p_value}
        print(f"Curvature t-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Cohen's d for curvature
        cohens_d = np.mean(curvature_diffs) / np.std(curvature_diffs) if np.std(curvature_diffs) > 0 else 0
        results["curvature_cohens_d"] = cohens_d
        print(f"Curvature Cohen's d: {cohens_d:.4f}")
        
        # KEY FINDING: Trajectory divergence profile
        results["max_layer_divergence"] = max_layer_divergence
        results["most_divergent_layer"] = most_divergent_layer
        
        print(f"\nMost divergent layer: {most_divergent_layer} (divergence={max_layer_divergence:.4f})")
        
        # Verdict
        if max_layer_divergence > 0.01:  # 1% divergence
            results["verdict"] = "BREAKTHROUGH: Trajectory analysis reveals layer-specific divergence patterns"
            results["finding"] = (
                f"Layer {most_divergent_layer} shows {max_layer_divergence*100:.2f}% divergence - "
                "a potential intervention target"
            )
        elif abs(cohens_d) > 0.5:
            results["verdict"] = "PROMISING: Curvature differences have medium effect size"
            results["finding"] = f"Path curvature differs with Cohen's d = {cohens_d:.2f}"
        else:
            results["verdict"] = "NEGATIVE: Trajectories are geometrically similar"
            results["finding"] = "Both correct and incorrect reasoning follow similar paths"
        
        print(f"\n{results['verdict']}")
        print(f"Finding: {results['finding']}")
        
        results["status"] = "success"
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    output_path = Path("/home/me/eas/eas/analysis/results/trajectory_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_trajectory_analysis()
