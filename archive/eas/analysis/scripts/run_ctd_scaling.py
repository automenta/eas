#!/usr/bin/env python3
"""
CTD Scaling Hypothesis Validation

Tests whether Critical Token Divergence (CTD) increases with model size.

Prediction: CTD ratio should scale monotonically with model capacity.

Models to test (if available):
- Pythia-70m (tested, CTD=109x)
- Pythia-160m
- Pythia-410m
- Pythia-1B (if GPU available)

This is key evidence for publication - demonstrating scale invariance.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from scipy import stats
import time

import sys
sys.path.insert(0, '/home/me/eas')


def generate_ctd_pairs() -> List[Dict]:
    """Generate pairs for CTD measurement."""
    return [
        {
            "correct": "If A then B . A is true . Therefore B is true .",
            "incorrect": "If A then B . A is true . Therefore B is false .",
        },
        {
            "correct": "All dogs bark . Rex is a dog . Rex barks .",
            "incorrect": "All dogs bark . Rex is a dog . Rex meows .",
        },
        {
            "correct": "The sky is blue . This is correct .",
            "incorrect": "The sky is green . This is correct .",
        },
        {
            "correct": "2 + 2 = 4 . Math is right .",
            "incorrect": "2 + 2 = 5 . Math is right .",
        },
        {
            "correct": "Fire needs oxygen . There is oxygen . Fire burns .",
            "incorrect": "Fire needs oxygen . There is oxygen . Fire freezes .",
        },
        {
            "correct": "Mammals are warm-blooded . Whales are mammals . Whales are warm-blooded .",
            "incorrect": "Mammals are warm-blooded . Whales are mammals . Whales are cold-blooded .",
        },
        {
            "correct": "If rain then wet . It rained . Ground is wet .",
            "incorrect": "If rain then wet . It rained . Ground is dry .",
        },
        {
            "correct": "Birds have wings . Eagles are birds . Eagles have wings .",
            "incorrect": "Birds have wings . Eagles are birds . Eagles have fins .",
        },
    ]


def compute_ctd_for_model(model, pairs: List[Dict], target_layer: int) -> Dict:
    """Compute CTD metrics for a single model."""
    
    all_critical_divs = []
    all_non_critical_divs = []
    
    for pair in pairs:
        correct_text = pair["correct"]
        incorrect_text = pair["incorrect"]
        
        # Tokenize
        correct_enc = model.tokenizer(correct_text, return_tensors="pt", truncation=True, max_length=128)
        incorrect_enc = model.tokenizer(incorrect_text, return_tensors="pt", truncation=True, max_length=128)
        
        correct_tokens = model.tokenizer.convert_ids_to_tokens(correct_enc.input_ids[0])
        incorrect_tokens = model.tokenizer.convert_ids_to_tokens(incorrect_enc.input_ids[0])
        
        # Get activations
        with torch.no_grad():
            model.forward(correct_enc.input_ids.to(model.device))
            correct_hidden = model.get_layer_activation(target_layer)
            
            model.forward(incorrect_enc.input_ids.to(model.device))
            incorrect_hidden = model.get_layer_activation(target_layer)
        
        if correct_hidden is None or incorrect_hidden is None:
            continue
        
        correct_hidden = correct_hidden.squeeze(0).cpu()
        incorrect_hidden = incorrect_hidden.squeeze(0).cpu()
        
        min_len = min(len(correct_tokens), len(incorrect_tokens),
                      correct_hidden.shape[0], incorrect_hidden.shape[0])
        
        for i in range(min_len):
            c_vec = correct_hidden[i]
            ic_vec = incorrect_hidden[i]
            
            # Cosine divergence
            cos_sim = torch.nn.functional.cosine_similarity(
                c_vec.unsqueeze(0), ic_vec.unsqueeze(0)
            ).item()
            divergence = 1 - cos_sim
            
            # Check if tokens differ (critical position)
            is_critical = correct_tokens[i] != incorrect_tokens[i]
            
            if is_critical:
                all_critical_divs.append(divergence)
            else:
                all_non_critical_divs.append(divergence)
    
    # Compute statistics
    if not all_critical_divs or not all_non_critical_divs:
        return None
    
    mean_critical = np.mean(all_critical_divs)
    mean_non_critical = np.mean(all_non_critical_divs)
    ctd_ratio = mean_critical / mean_non_critical if mean_non_critical > 0 else float('inf')
    
    t_stat, p_value = stats.ttest_ind(all_critical_divs, all_non_critical_divs)
    
    # Cohen's d
    pooled_std = np.sqrt((np.var(all_critical_divs) + np.var(all_non_critical_divs)) / 2)
    cohens_d = (mean_critical - mean_non_critical) / pooled_std if pooled_std > 0 else 0
    
    return {
        "ctd_mean": mean_critical,
        "ctd_max": np.max(all_critical_divs),
        "non_critical_mean": mean_non_critical,
        "ctd_ratio": ctd_ratio,
        "cohens_d": cohens_d,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_critical": len(all_critical_divs),
        "n_non_critical": len(all_non_critical_divs)
    }


def run_scaling_validation():
    """Test CTD across model sizes."""
    print("=" * 60)
    print("CTD SCALING HYPOTHESIS VALIDATION")
    print("=" * 60)
    print("\nPrediction: CTD ratio increases with model size")
    
    results = {"status": "started", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    
    # Models to test (in order of size)
    model_configs = [
        ("EleutherAI/pythia-70m", 70),
        ("EleutherAI/pythia-160m", 160),
        ("EleutherAI/pythia-410m", 410),
        # Larger models would need GPU
    ]
    
    pairs = generate_ctd_pairs()
    print(f"\nGenerated {len(pairs)} test pairs")
    
    model_results = {}
    
    for model_name, param_count in model_configs:
        print(f"\n{'='*40}")
        print(f"Testing: {model_name} ({param_count}M params)")
        print(f"{'='*40}")
        
        try:
            from eas.src.models.transformer import PretrainedTransformer
            
            start_time = time.time()
            model = PretrainedTransformer(model_name, device="cpu")
            load_time = time.time() - start_time
            
            print(f"Loaded in {load_time:.1f}s")
            print(f"Layers: {model.num_layers}, Hidden dim: {model.d_model}")
            
            # Test multiple layers
            layer_results = {}
            
            for layer_idx in range(model.num_layers):
                ctd_result = compute_ctd_for_model(model, pairs, layer_idx)
                if ctd_result:
                    layer_results[layer_idx] = ctd_result
                    print(f"  Layer {layer_idx}: CTD={ctd_result['ctd_mean']:.4f}, Ratio={ctd_result['ctd_ratio']:.1f}x, d={ctd_result['cohens_d']:.2f}")
            
            # Find best layer
            if layer_results:
                best_layer = max(layer_results, key=lambda l: layer_results[l]['ctd_ratio'])
                best_ctd = layer_results[best_layer]
                
                model_results[model_name] = {
                    "param_count_millions": param_count,
                    "num_layers": model.num_layers,
                    "hidden_dim": model.d_model,
                    "best_layer": best_layer,
                    "best_ctd_ratio": best_ctd['ctd_ratio'],
                    "best_ctd_mean": best_ctd['ctd_mean'],
                    "best_ctd_max": best_ctd['ctd_max'],
                    "best_cohens_d": best_ctd['cohens_d'],
                    "best_p_value": best_ctd['p_value'],
                    "all_layers": {str(k): v for k, v in layer_results.items()}
                }
                
                print(f"\n  Best: Layer {best_layer} with CTD ratio = {best_ctd['ctd_ratio']:.1f}x")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"  Error: {e}")
            model_results[model_name] = {"error": str(e)}
            continue
    
    # Analyze scaling trend
    print("\n" + "=" * 60)
    print("SCALING ANALYSIS")
    print("=" * 60)
    
    valid_results = {k: v for k, v in model_results.items() if "error" not in v}
    
    if len(valid_results) >= 2:
        # Sort by parameter count
        sorted_models = sorted(valid_results.items(), 
                              key=lambda x: x[1]["param_count_millions"])
        
        print("\nCTD Ratio by Model Size:")
        print("-" * 50)
        
        sizes = []
        ratios = []
        
        for model_name, data in sorted_models:
            sizes.append(data["param_count_millions"])
            ratios.append(data["best_ctd_ratio"])
            print(f"  {model_name}: {data['best_ctd_ratio']:.1f}x (Layer {data['best_layer']})")
        
        # Compute correlation
        if len(sizes) >= 2:
            correlation, p_value = stats.pearsonr(sizes, ratios)
            results["scaling_correlation"] = correlation
            results["scaling_p_value"] = p_value
            
            print(f"\nCorrelation (size vs CTD ratio): r = {correlation:.3f}, p = {p_value:.4f}")
            
            if correlation > 0.5 and p_value < 0.1:
                scaling_verdict = "CONFIRMED: CTD ratio increases with model size"
            elif correlation > 0:
                scaling_verdict = "PARTIAL: Positive trend but not significant"
            else:
                scaling_verdict = "UNEXPECTED: No positive correlation found"
            
            print(f"\n{scaling_verdict}")
            results["scaling_verdict"] = scaling_verdict
    
    results["model_results"] = model_results
    results["status"] = "success"
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print("\n| Model | Params | Best Layer | CTD Ratio | Cohen's d |")
    print("|-------|--------|------------|-----------|-----------|")
    
    for model_name, data in sorted(valid_results.items(), 
                                   key=lambda x: x[1]["param_count_millions"]):
        print(f"| {model_name.split('/')[-1]} | {data['param_count_millions']}M | "
              f"{data['best_layer']} | {data['best_ctd_ratio']:.1f}x | "
              f"{data['best_cohens_d']:.2f} |")
    
    # Save results
    output_path = Path("/home/me/eas/eas/analysis/results/ctd_scaling_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_scaling_validation()
