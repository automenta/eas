#!/usr/bin/env python3
"""
david_vs_goliath.py - Complete David vs Goliath Benchmark

Tests whether smaller models enhanced with EAS (Emergent Activation Snapping)
can achieve equal or better accuracy than larger baseline models.

Experiment Structure:
1. Load multiple models (small "Davids" and large "Goliaths")
2. For each model, test with and without EAS enhancement
3. Compare raw accuracy improvements
4. Determine whether EAS enables small models to match/beat large ones

Usage:
    python david_vs_goliath.py                    # Default benchmark
    python david_vs_goliath.py --quick            # Quick test (30 samples)
    python david_vs_goliath.py --samples 200      # Full benchmark
    python david_vs_goliath.py --models pythia-70m gpt2  # Specific models
"""

import argparse
import torch
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from copy import deepcopy

from eas_core import EASConfig, EASIntervener, wrap_model_with_eas
from utils import (
    load_model, load_logiqa, load_arc_challenge,
    evaluate_model, print_results_table, save_results,
    BenchmarkResult, EvalSample, get_device, MODEL_REGISTRY
)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    model_key: str
    config_name: str
    use_eas: bool
    eas_config: Optional[EASConfig] = None
    random_control: bool = False  # Random perturbation instead of EAS


def create_experiment_configs(
    model_keys: List[str],
    include_baselines: bool = True,
    include_eas: bool = True,
    include_random: bool = True
) -> List[ExperimentConfig]:
    """Create all experiment configurations to run."""
    configs = []
    
    for model_key in model_keys:
        model_info = MODEL_REGISTRY[model_key]
        hidden_dim = model_info["hidden_dim"]
        
        if include_baselines:
            # Baseline: no intervention
            configs.append(ExperimentConfig(
                model_key=model_key,
                config_name="baseline",
                use_eas=False
            ))
        
        if include_eas:
            # EAS with default settings
            configs.append(ExperimentConfig(
                model_key=model_key,
                config_name="eas",
                use_eas=True,
                eas_config=EASConfig(
                    hidden_dim=hidden_dim,
                    num_attractors=10,
                    base_alpha=0.3,
                    warmup_samples=15
                )
            ))
            
            # EAS with stronger intervention
            configs.append(ExperimentConfig(
                model_key=model_key,
                config_name="eas_strong",
                use_eas=True,
                eas_config=EASConfig(
                    hidden_dim=hidden_dim,
                    num_attractors=10,
                    base_alpha=0.5,
                    warmup_samples=15
                )
            ))
        
        if include_random:
            # Random control: same perturbation magnitude but random direction
            configs.append(ExperimentConfig(
                model_key=model_key,
                config_name="random_control",
                use_eas=False,
                random_control=True
            ))
    
    return configs


class RandomPerturbationIntervener:
    """Control: random perturbation with same magnitude as EAS."""
    
    def __init__(self, hidden_dim: int, magnitude: float = 0.1):
        self.hidden_dim = hidden_dim
        self.magnitude = magnitude
        self.intervention_count = 0
        self.total_samples = 0
        self.successful_samples = 0
        self.last_hidden_state = None
    
    def intervene(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply random perturbation."""
        self.last_hidden_state = hidden_state.detach().clone()
        noise = torch.randn_like(hidden_state) * self.magnitude
        self.intervention_count += 1
        return hidden_state + noise
    
    def record_sample(self):
        self.total_samples += 1
    
    def update_on_success(self, hidden_state=None):
        self.successful_samples += 1
    
    def get_stats(self):
        return {
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "intervention_count": self.intervention_count,
            "attractor_entropy": 0.0,
            "warmup_complete": True,
        }


def run_single_experiment(
    experiment: ExperimentConfig,
    samples: List[EvalSample],
    device: str,
    verbose: bool = True
) -> BenchmarkResult:
    """Run a single experiment configuration."""
    
    model_info = MODEL_REGISTRY[experiment.model_key]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {experiment.model_key} + {experiment.config_name}")
        print(f"{'='*60}")
    
    # Load model
    model, tokenizer, info = load_model(experiment.model_key, device)
    
    intervener = None
    
    if experiment.use_eas:
        # Wrap with EAS
        model, intervener = wrap_model_with_eas(
            model,
            hidden_dim=model_info["hidden_dim"],
            config=experiment.eas_config
        )
        intervener.to(device)
        
        if verbose:
            print(f"EAS enabled: {experiment.eas_config.num_attractors} attractors, "
                  f"alpha={experiment.eas_config.base_alpha}")
    
    elif experiment.random_control:
        # Random perturbation control
        intervener = RandomPerturbationIntervener(model_info["hidden_dim"])
        if verbose:
            print("Random perturbation control enabled")
    
    # Evaluate
    result = evaluate_model(
        model, tokenizer, samples,
        device=device,
        show_progress=verbose,
        intervener=intervener
    )
    
    result.model_name = f"{experiment.model_key} ({model_info['size']})"
    result.config_name = experiment.config_name
    
    if verbose:
        print(f"Result: {result.accuracy:.1%} accuracy ({result.num_correct}/{result.num_samples})")
        if intervener:
            stats = intervener.get_stats()
            print(f"Interventions: {stats['intervention_count']}")
    
    # Clean up model to free memory
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


def run_benchmark(
    model_keys: Optional[List[str]] = None,
    num_samples: int = 100,
    dataset: str = "logiqa",
    device: Optional[str] = None,
    include_random: bool = True,
    verbose: bool = True
) -> List[BenchmarkResult]:
    """
    Run the complete David vs Goliath benchmark.
    
    Args:
        model_keys: Models to test (default: pythia-70m, gpt2, gpt2-medium)
        num_samples: Number of samples to evaluate
        dataset: Dataset to use ("logiqa" or "arc")
        device: Device to run on (auto-detected if None)
        include_random: Include random perturbation control
        verbose: Print progress
    
    Returns:
        List of BenchmarkResults
    """
    device = device or get_device()
    
    if model_keys is None:
        # Default: compare small Davids with medium Goliath
        model_keys = ["pythia-70m", "gpt2", "gpt2-medium"]
    
    if verbose:
        print("\n" + "="*60)
        print("DAVID VS GOLIATH BENCHMARK")
        print("="*60)
        print(f"Device: {device}")
        print(f"Models: {', '.join(model_keys)}")
        print(f"Samples: {num_samples}")
        print(f"Dataset: {dataset}")
        print("="*60)
    
    # Load dataset
    if dataset == "logiqa":
        samples = load_logiqa(max_samples=num_samples)
    elif dataset == "arc":
        samples = load_arc_challenge(max_samples=num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Create experiment configurations
    experiments = create_experiment_configs(
        model_keys,
        include_baselines=True,
        include_eas=True,
        include_random=include_random
    )
    
    if verbose:
        print(f"\nTotal experiments: {len(experiments)}")
    
    # Run all experiments
    results = []
    for exp in experiments:
        try:
            result = run_single_experiment(exp, samples, device, verbose)
            results.append(result)
        except Exception as e:
            print(f"Error running {exp.model_key} + {exp.config_name}: {e}")
            continue
    
    return results


def analyze_results(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """
    Analyze benchmark results to determine if David beats Goliath.
    """
    analysis = {
        "eas_helps": False,
        "david_beats_goliath": False,
        "best_david": None,
        "best_goliath": None,
        "improvements": []
    }
    
    # Group by model
    by_model = {}
    for r in results:
        model_key = r.model_name.split()[0]  # Extract model name
        if model_key not in by_model:
            by_model[model_key] = {}
        by_model[model_key][r.config_name] = r
    
    # Check if EAS improves accuracy for each model
    for model_key, configs in by_model.items():
        if "baseline" in configs and "eas" in configs:
            baseline_acc = configs["baseline"].accuracy
            eas_acc = configs["eas"].accuracy
            improvement = eas_acc - baseline_acc
            
            analysis["improvements"].append({
                "model": model_key,
                "baseline": baseline_acc,
                "eas": eas_acc,
                "improvement": improvement,
                "improved": improvement > 0
            })
            
            if improvement > 0:
                analysis["eas_helps"] = True
    
    # Find best David and Goliath
    davids = [r for r in results if "70m" in r.model_name or "gpt2 " in r.model_name.lower()]
    goliaths = [r for r in results if "medium" in r.model_name.lower() or "large" in r.model_name.lower()]
    
    if davids:
        best_david = max(davids, key=lambda r: r.accuracy)
        analysis["best_david"] = {
            "model": best_david.model_name,
            "config": best_david.config_name,
            "accuracy": best_david.accuracy
        }
    
    if goliaths:
        best_goliath = max(goliaths, key=lambda r: r.accuracy)
        analysis["best_goliath"] = {
            "model": best_goliath.model_name,
            "config": best_goliath.config_name,
            "accuracy": best_goliath.accuracy
        }
    
    # David beats Goliath?
    if analysis["best_david"] and analysis["best_goliath"]:
        if analysis["best_david"]["accuracy"] >= analysis["best_goliath"]["accuracy"]:
            analysis["david_beats_goliath"] = True
    
    return analysis


def print_analysis(analysis: Dict[str, Any]):
    """Print analysis results."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # EAS improvements
    print("\n📊 EAS Effect on Accuracy:")
    for imp in analysis["improvements"]:
        emoji = "✅" if imp["improved"] else "❌"
        sign = "+" if imp["improvement"] >= 0 else ""
        print(f"  {emoji} {imp['model']}: {imp['baseline']:.1%} → {imp['eas']:.1%} "
              f"({sign}{imp['improvement']:.1%})")
    
    # David vs Goliath
    print("\n🏆 David vs Goliath:")
    if analysis["best_david"]:
        d = analysis["best_david"]
        print(f"  Best David: {d['model']} + {d['config']} = {d['accuracy']:.1%}")
    if analysis["best_goliath"]:
        g = analysis["best_goliath"]
        print(f"  Best Goliath: {g['model']} + {g['config']} = {g['accuracy']:.1%}")
    
    if analysis["david_beats_goliath"]:
        print("\n  🎉 DAVID WINS! Small model matches or beats larger model!")
    else:
        print("\n  👑 Goliath still leads (larger model is more accurate)")
    
    # Verdict
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    
    if analysis["eas_helps"] and analysis["david_beats_goliath"]:
        print("✅ EAS VALIDATED: Improves accuracy AND enables small models to compete!")
        print("   Recommendation: CONTINUE research, prepare publication")
    elif analysis["eas_helps"]:
        print("🔶 EAS PARTIALLY VALIDATED: Improves accuracy but not enough to beat larger models")
        print("   Recommendation: TUNE further or try larger Davids (Pythia-410m)")
    else:
        print("❌ EAS NOT VALIDATED: No consistent accuracy improvement observed")
        print("   Recommendation: INVESTIGATE or PIVOT research direction")


def main():
    parser = argparse.ArgumentParser(description="David vs Goliath EAS Benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to test (default: pythia-70m, gpt2, gpt2-medium)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--dataset", choices=["logiqa", "arc"], default="logiqa",
                       help="Dataset to use")
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cuda/cpu/mps, auto-detected if not specified)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with 30 samples and fewer models")
    parser.add_argument("--no-random", action="store_true",
                       help="Skip random control experiments")
    parser.add_argument("--output", type=str, default="results.json",
                       help="Output file for results")
    parser.add_argument("--quiet", action="store_true",
                       help="Minimal output")
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.samples = 30
        if args.models is None:
            args.models = ["pythia-70m", "gpt2"]
    
    # Available models check
    if args.models:
        for m in args.models:
            if m not in MODEL_REGISTRY:
                print(f"Unknown model: {m}")
                print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
                return
    
    # Run benchmark
    start_time = time.time()
    
    results = run_benchmark(
        model_keys=args.models,
        num_samples=args.samples,
        dataset=args.dataset,
        device=args.device,
        include_random=not args.no_random,
        verbose=not args.quiet
    )
    
    total_time = time.time() - start_time
    
    # Print and save results
    print_results_table(results)
    
    analysis = analyze_results(results)
    print_analysis(analysis)
    
    # Save results
    save_results(results, args.output)
    
    print(f"\n⏱️ Total benchmark time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
