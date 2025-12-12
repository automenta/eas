#!/usr/bin/env python3
"""
Token-Level Divergence Analysis

KEY INSIGHT: Sequence pooling may hide important signals. 
Specific token positions (logical connectives, answers) may show 
dramatically higher divergence even in small models.

This analysis looks for "divergence hotspots" at the token level.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from scipy import stats

import sys
sys.path.insert(0, '/home/me/eas')


def generate_logic_pairs() -> List[Dict]:
    """Generate matched pairs with clear token structure."""
    pairs = [
        # Modus Ponens - correct vs incorrect conclusion
        {
            "correct": "If it rains, the ground is wet. It rains. Therefore, the ground is wet.",
            "incorrect": "If it rains, the ground is wet. It rains. Therefore, the ground is dry.",
            "type": "modus_ponens",
            "critical_position": "conclusion"  # The last phrase is critical
        },
        {
            "correct": "All cats are mammals. Fluffy is a cat. Therefore, Fluffy is a mammal.",
            "incorrect": "All cats are mammals. Fluffy is a cat. Therefore, Fluffy is a reptile.",
            "type": "syllogism",
            "critical_position": "conclusion"
        },
        # Negation pairs
        {
            "correct": "The sky is blue. This statement is true.",
            "incorrect": "The sky is blue. This statement is false.",
            "type": "truth_value",
            "critical_position": "judgment"
        },
        {
            "correct": "2 + 2 = 4. The calculation is correct.",
            "incorrect": "2 + 2 = 5. The calculation is correct.",
            "type": "arithmetic",
            "critical_position": "equation"
        },
        # Semantic consistency
        {
            "correct": "The sun rises in the east and sets in the west.",
            "incorrect": "The sun rises in the west and sets in the east.",
            "type": "semantic",
            "critical_position": "direction"
        },
        # Quantifier logic
        {
            "correct": "All birds have feathers. A robin is a bird. Robins have feathers.",
            "incorrect": "All birds have feathers. A robin is a bird. Robins have scales.",
            "type": "quantifier",
            "critical_position": "property"
        },
        # Conditional chains
        {
            "correct": "If A then B. If B then C. A is true. Therefore C is true.",
            "incorrect": "If A then B. If B then C. A is true. Therefore C is false.",
            "type": "chain",
            "critical_position": "final_conclusion"
        },
        # Contradiction detection
        {
            "correct": "The ball is red. The ball is not blue. This is consistent.",
            "incorrect": "The ball is red. The ball is blue. This is consistent.",
            "type": "contradiction",
            "critical_position": "consistency"
        },
    ]
    return pairs


def get_token_activations(model, text: str, layer_idx: int) -> Tuple[torch.Tensor, List[str]]:
    """Get per-token activations for a text at a specific layer."""
    input_ids = model.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    tokens = model.tokenizer.convert_ids_to_tokens(input_ids.input_ids[0])
    
    with torch.no_grad():
        model.forward(input_ids.input_ids.to(model.device))
        hidden = model.get_layer_activation(layer_idx)
    
    if hidden is None:
        return None, tokens
    
    # Return shape: [seq_len, hidden_dim]
    return hidden.squeeze(0).cpu(), tokens


def compute_token_divergence(correct_hidden: torch.Tensor, 
                             incorrect_hidden: torch.Tensor,
                             correct_tokens: List[str],
                             incorrect_tokens: List[str]) -> Dict:
    """Compute divergence at each token position."""
    
    # Ensure same length (pad shorter if needed)
    min_len = min(len(correct_tokens), len(incorrect_tokens), 
                  correct_hidden.shape[0], incorrect_hidden.shape[0])
    
    correct_hidden = correct_hidden[:min_len]
    incorrect_hidden = incorrect_hidden[:min_len]
    
    # Per-position metrics
    cosine_sims = []
    euclidean_dists = []
    magnitude_diffs = []
    
    for i in range(min_len):
        c_vec = correct_hidden[i]
        ic_vec = incorrect_hidden[i]
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            c_vec.unsqueeze(0), ic_vec.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)
        
        # Euclidean distance
        euclid = torch.norm(c_vec - ic_vec).item()
        euclidean_dists.append(euclid)
        
        # Magnitude difference
        mag_diff = abs(torch.norm(c_vec).item() - torch.norm(ic_vec).item())
        magnitude_diffs.append(mag_diff)
    
    return {
        "cosine_sims": cosine_sims,
        "euclidean_dists": euclidean_dists,
        "magnitude_diffs": magnitude_diffs,
        "tokens": correct_tokens[:min_len],
        "num_tokens": min_len
    }


def analyze_divergence_hotspots(token_data: List[Dict]) -> Dict:
    """Identify positions with maximum divergence across examples."""
    
    # Aggregate by relative position (beginning, middle, end)
    all_divisions = {"first_quarter": [], "middle_half": [], "last_quarter": []}
    
    all_cosine_divergences = []
    position_divergences = {}
    
    for data in token_data:
        n = data["num_tokens"]
        cosines = data["cosine_sims"]
        divergences = [1 - c for c in cosines]
        
        all_cosine_divergences.extend(divergences)
        
        # Categorize by position
        for i, div in enumerate(divergences):
            rel_pos = i / n  # Relative position [0, 1]
            
            if rel_pos < 0.25:
                all_divisions["first_quarter"].append(div)
            elif rel_pos < 0.75:
                all_divisions["middle_half"].append(div)
            else:
                all_divisions["last_quarter"].append(div)
            
            # Track max divergence by absolute position
            if i not in position_divergences:
                position_divergences[i] = []
            position_divergences[i].append(div)
    
    # Statistical tests
    results = {}
    
    # Compare first_quarter vs last_quarter
    if all_divisions["first_quarter"] and all_divisions["last_quarter"]:
        t_stat, p_value = stats.ttest_ind(
            all_divisions["first_quarter"], 
            all_divisions["last_quarter"]
        )
        results["first_vs_last_tstat"] = t_stat
        results["first_vs_last_pvalue"] = p_value
    
    # Mean divergence by region
    for region, values in all_divisions.items():
        if values:
            results[f"{region}_mean_divergence"] = np.mean(values)
            results[f"{region}_std_divergence"] = np.std(values)
            results[f"{region}_max_divergence"] = np.max(values)
    
    # Overall stats
    results["overall_mean_divergence"] = np.mean(all_cosine_divergences)
    results["overall_std_divergence"] = np.std(all_cosine_divergences)
    results["overall_max_divergence"] = np.max(all_cosine_divergences)
    
    # Find position with maximum average divergence
    pos_means = {pos: np.mean(divs) for pos, divs in position_divergences.items()}
    if pos_means:
        max_pos = max(pos_means, key=pos_means.get)
        results["max_divergence_position"] = int(max_pos)
        results["max_position_divergence"] = pos_means[max_pos]
    
    return results


def run_token_divergence_analysis():
    """Main analysis function."""
    print("=" * 60)
    print("TOKEN-LEVEL DIVERGENCE ANALYSIS")
    print("=" * 60)
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        
        print("\nLoading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        num_layers = model.num_layers
        hidden_dim = model.d_model
        print(f"Model: {num_layers} layers, {hidden_dim}d")
        
        pairs = generate_logic_pairs()
        print(f"\nGenerated {len(pairs)} logic pairs")
        
        # Analyze each layer
        layer_results = {}
        
        for layer_idx in range(num_layers):
            print(f"\nAnalyzing Layer {layer_idx}...")
            
            token_data = []
            
            for pair in pairs:
                correct_hidden, correct_tokens = get_token_activations(
                    model, pair["correct"], layer_idx
                )
                incorrect_hidden, incorrect_tokens = get_token_activations(
                    model, pair["incorrect"], layer_idx
                )
                
                if correct_hidden is not None and incorrect_hidden is not None:
                    div_data = compute_token_divergence(
                        correct_hidden, incorrect_hidden,
                        correct_tokens, incorrect_tokens
                    )
                    div_data["pair_type"] = pair["type"]
                    div_data["critical_position"] = pair["critical_position"]
                    token_data.append(div_data)
            
            hotspot_analysis = analyze_divergence_hotspots(token_data)
            layer_results[layer_idx] = hotspot_analysis
            
            print(f"  Mean divergence: {hotspot_analysis.get('overall_mean_divergence', 0):.6f}")
            print(f"  Max divergence: {hotspot_analysis.get('overall_max_divergence', 0):.6f}")
            print(f"  Last quarter mean: {hotspot_analysis.get('last_quarter_mean_divergence', 0):.6f}")
        
        results["layer_results"] = {str(k): v for k, v in layer_results.items()}
        
        # Key insight: Compare early vs late tokens
        print("\n" + "=" * 60)
        print("KEY FINDINGS: Token Position Analysis")
        print("=" * 60)
        
        for layer_idx, lr in layer_results.items():
            first_q = lr.get("first_quarter_mean_divergence", 0)
            last_q = lr.get("last_quarter_mean_divergence", 0)
            ratio = last_q / first_q if first_q > 0 else 0
            p_val = lr.get("first_vs_last_pvalue", 1.0)
            
            significance = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            
            print(f"Layer {layer_idx}: Last/First ratio = {ratio:.2f}x {significance}")
            print(f"         First quarter: {first_q:.6f}, Last quarter: {last_q:.6f}")
            print(f"         p-value: {p_val:.4f}")
        
        # Aggregate finding
        all_first = []
        all_last = []
        for lr in layer_results.values():
            if "first_quarter_mean_divergence" in lr:
                all_first.append(lr["first_quarter_mean_divergence"])
            if "last_quarter_mean_divergence" in lr:
                all_last.append(lr["last_quarter_mean_divergence"])
        
        if all_first and all_last:
            avg_first = np.mean(all_first)
            avg_last = np.mean(all_last)
            results["aggregate_first_quarter_mean"] = avg_first
            results["aggregate_last_quarter_mean"] = avg_last
            results["aggregate_last_first_ratio"] = avg_last / avg_first if avg_first > 0 else 0
        
        # Breakthrough criterion
        max_div = max(lr.get("overall_max_divergence", 0) for lr in layer_results.values())
        results["max_token_level_divergence"] = max_div
        
        if max_div > 0.01:  # 1% divergence at token level
            results["verdict"] = "BREAKTHROUGH: Token-level analysis reveals hidden signal"
        elif max_div > 0.005:
            results["verdict"] = "PROMISING: Moderate token-level divergence found"
        else:
            results["verdict"] = "NEGATIVE: Token-level analysis mirrors sequence-level"
        
        results["status"] = "success"
        
        print(f"\n{results['verdict']}")
        print(f"Maximum token-level divergence: {max_div:.6f}")
        
    except Exception as e:
        results["status"] = "error"
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
        print(f"Error: {e}")
    
    # Save results
    output_path = Path("/home/me/eas/eas/analysis/results/token_divergence_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_token_divergence_analysis()
