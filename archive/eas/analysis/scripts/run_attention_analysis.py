#!/usr/bin/env python3
"""
Attention Pattern Fingerprinting

KEY INSIGHT: Even if hidden states are similar, attention patterns may differ.
Small and large models may share similar attention "fingerprints" for correct 
reasoning - an architectural invariant.

This analysis extracts and compares attention patterns during reasoning.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from scipy import stats
from collections import defaultdict

import sys
sys.path.insert(0, '/home/me/eas')


def generate_reasoning_pairs() -> List[Dict]:
    """Generate pairs designed to test attention behavior."""
    return [
        {
            "correct": "If A then B. A is true. Therefore B is true.",
            "incorrect": "If A then B. A is true. Therefore B is false.",
            "type": "modus_ponens"
        },
        {
            "correct": "All dogs bark. Rex is a dog. Rex barks.",
            "incorrect": "All dogs bark. Rex is a dog. Rex meows.",
            "type": "syllogism"
        },
        {
            "correct": "The number 7 is odd. Odd numbers are not divisible by 2. 7 is not divisible by 2.",
            "incorrect": "The number 7 is odd. Odd numbers are not divisible by 2. 7 is divisible by 2.",
            "type": "property_chain"
        },
        {
            "correct": "Fire needs oxygen. The room has oxygen. Fire can burn in the room.",
            "incorrect": "Fire needs oxygen. The room has oxygen. Fire cannot burn in the room.",
            "type": "causal"
        },
        {
            "correct": "Either it is day or night. It is not day. Therefore it is night.",
            "incorrect": "Either it is day or night. It is not day. Therefore it is day.",
            "type": "disjunction"
        },
        {
            "correct": "If rain then wet. If wet then slippery. Rain means slippery.",
            "incorrect": "If rain then wet. If wet then slippery. Rain means dry.",
            "type": "transitive"
        },
        {
            "correct": "Premise: All humans are mortal. Socrates is human. Conclusion: Socrates is mortal.",
            "incorrect": "Premise: All humans are mortal. Socrates is human. Conclusion: Socrates is immortal.",
            "type": "classic_syllogism"
        },
        {
            "correct": "Input: 5 + 3. Output: 8. This is correct arithmetic.",
            "incorrect": "Input: 5 + 3. Output: 7. This is correct arithmetic.",
            "type": "arithmetic"
        }
    ]


class AttentionExtractor:
    """Extract attention patterns from transformer models."""
    
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.hooks = []
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Register hooks to capture attention weights."""
        # Find attention modules
        for name, module in self.model.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                # Check if this is the actual attention computation
                if 'self_attn' in name or 'attn' in name:
                    layer_idx = self._extract_layer_idx(name)
                    if layer_idx is not None:
                        hook = module.register_forward_hook(
                            self._make_attention_hook(layer_idx, name)
                        )
                        self.hooks.append(hook)
    
    def _extract_layer_idx(self, name: str) -> Optional[int]:
        """Extract layer index from module name."""
        import re
        # Match patterns like "layers.0", "h.5", "layer.12"
        match = re.search(r'(?:layers?|h)\.(\d+)', name)
        if match:
            return int(match.group(1))
        return None
    
    def _make_attention_hook(self, layer_idx: int, name: str):
        """Create hook for capturing attention."""
        def hook(module, input, output):
            # Different models return attention differently
            if isinstance(output, tuple) and len(output) >= 2:
                # Many models return (hidden_states, attention_weights, ...)
                attn_weights = output[1] if output[1] is not None else None
                if attn_weights is not None:
                    self.attention_maps[layer_idx] = attn_weights.detach().cpu()
        return hook
    
    def get_attention(self, text: str, output_attentions: bool = True) -> Dict[int, torch.Tensor]:
        """Get attention patterns for input text."""
        self.attention_maps = {}
        
        input_ids = self.model.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        )
        
        with torch.no_grad():
            # Try to get attention through model forward
            try:
                outputs = self.model.model(
                    input_ids.input_ids.to(self.model.device),
                    output_attentions=True
                )
                
                # If model returns attentions directly
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    for i, attn in enumerate(outputs.attentions):
                        self.attention_maps[i] = attn.detach().cpu()
            except:
                # Fall back to hooks
                self.model.forward(input_ids.input_ids.to(self.model.device))
        
        return self.attention_maps
    
    def cleanup(self):
        """Remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def compute_attention_entropy(attn: torch.Tensor) -> float:
    """Compute entropy of attention distribution."""
    # attn shape: [batch, heads, seq, seq] or [heads, seq, seq]
    if attn.dim() == 4:
        attn = attn.squeeze(0)  # [heads, seq, seq]
    
    # Average across heads and positions for overall entropy
    attn_flat = attn.mean(dim=0).view(-1)  # Flatten [seq*seq]
    attn_flat = attn_flat + 1e-10  # Avoid log(0)
    entropy = -torch.sum(attn_flat * torch.log(attn_flat)).item()
    
    return entropy


def compute_attention_focus(attn: torch.Tensor) -> Dict[str, float]:
    """Compute attention focus metrics."""
    if attn.dim() == 4:
        attn = attn.squeeze(0)  # [heads, seq, seq]
    
    num_heads = attn.shape[0]
    seq_len = attn.shape[1]
    
    # Per-head entropy
    head_entropies = []
    for h in range(num_heads):
        head_attn = attn[h].view(-1)
        head_attn = head_attn + 1e-10
        ent = -torch.sum(head_attn * torch.log(head_attn)).item()
        head_entropies.append(ent)
    
    # Attention to last token (often important for causal reasoning)
    last_token_attention = attn[:, :, -1].mean().item()
    
    # Self-attention diagonal (identity attention)
    diag_attention = torch.diagonal(attn.mean(dim=0)).mean().item()
    
    # Max attention weight (sharpness)
    max_attention = attn.max().item()
    
    return {
        "mean_head_entropy": np.mean(head_entropies),
        "std_head_entropy": np.std(head_entropies),
        "last_token_attention": last_token_attention,
        "diagonal_attention": diag_attention,
        "max_attention": max_attention,
        "head_entropy_variance": np.var(head_entropies)
    }


def compare_attention_patterns(correct_attn: Dict[int, torch.Tensor],
                               incorrect_attn: Dict[int, torch.Tensor]) -> Dict:
    """Compare attention patterns between correct and incorrect reasoning."""
    
    results = {}
    
    common_layers = set(correct_attn.keys()) & set(incorrect_attn.keys())
    
    for layer_idx in sorted(common_layers):
        c_attn = correct_attn[layer_idx]
        ic_attn = incorrect_attn[layer_idx]
        
        # Ensure same shape
        min_seq = min(c_attn.shape[-1], ic_attn.shape[-1])
        c_attn = c_attn[..., :min_seq, :min_seq]
        ic_attn = ic_attn[..., :min_seq, :min_seq]
        
        # Compute metrics for each
        c_metrics = compute_attention_focus(c_attn)
        ic_metrics = compute_attention_focus(ic_attn)
        
        # Differences
        layer_result = {
            "correct": c_metrics,
            "incorrect": ic_metrics,
            "differences": {}
        }
        
        for key in c_metrics:
            diff = c_metrics[key] - ic_metrics[key]
            layer_result["differences"][f"{key}_diff"] = diff
        
        # Frobenius norm of attention difference
        attn_diff = torch.norm(c_attn.float() - ic_attn.float()).item()
        layer_result["attention_frobenius_diff"] = attn_diff
        
        # Cosine similarity of flattened attention
        c_flat = c_attn.view(-1)
        ic_flat = ic_attn.view(-1)
        cos_sim = torch.nn.functional.cosine_similarity(
            c_flat.unsqueeze(0), ic_flat.unsqueeze(0)
        ).item()
        layer_result["attention_cosine_similarity"] = cos_sim
        
        results[layer_idx] = layer_result
    
    return results


def run_attention_analysis():
    """Main analysis function."""
    print("=" * 60)
    print("ATTENTION PATTERN FINGERPRINTING")
    print("=" * 60)
    
    results = {"status": "started"}
    
    try:
        from eas.src.models.transformer import PretrainedTransformer
        
        print("\nLoading Pythia-70m...")
        model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
        num_layers = model.num_layers
        print(f"Model: {num_layers} layers")
        
        pairs = generate_reasoning_pairs()
        print(f"Generated {len(pairs)} reasoning pairs")
        
        extractor = AttentionExtractor(model)
        
        all_comparisons = []
        
        for pair in pairs:
            print(f"\nAnalyzing: {pair['type']}")
            
            correct_attn = extractor.get_attention(pair["correct"])
            incorrect_attn = extractor.get_attention(pair["incorrect"])
            
            if correct_attn and incorrect_attn:
                comparison = compare_attention_patterns(correct_attn, incorrect_attn)
                comparison["pair_type"] = pair["type"]
                all_comparisons.append(comparison)
                
                # Print some stats
                for layer_idx in sorted(comparison.keys()):
                    if isinstance(layer_idx, int):
                        cos_sim = comparison[layer_idx].get("attention_cosine_similarity", 0)
                        frob_diff = comparison[layer_idx].get("attention_frobenius_diff", 0)
                        print(f"  Layer {layer_idx}: cos_sim={cos_sim:.4f}, frob_diff={frob_diff:.4f}")
        
        extractor.cleanup()
        
        # Aggregate findings
        print("\n" + "=" * 60)
        print("AGGREGATE FINDINGS")
        print("=" * 60)
        
        layer_stats = defaultdict(lambda: {
            "cos_sims": [], "frob_diffs": [], 
            "entropy_diffs": [], "last_token_diffs": []
        })
        
        for comp in all_comparisons:
            for layer_idx in comp:
                if isinstance(layer_idx, int):
                    layer_stats[layer_idx]["cos_sims"].append(
                        comp[layer_idx].get("attention_cosine_similarity", 0)
                    )
                    layer_stats[layer_idx]["frob_diffs"].append(
                        comp[layer_idx].get("attention_frobenius_diff", 0)
                    )
                    if "differences" in comp[layer_idx]:
                        diffs = comp[layer_idx]["differences"]
                        if "mean_head_entropy_diff" in diffs:
                            layer_stats[layer_idx]["entropy_diffs"].append(
                                diffs["mean_head_entropy_diff"]
                            )
                        if "last_token_attention_diff" in diffs:
                            layer_stats[layer_idx]["last_token_diffs"].append(
                                diffs["last_token_attention_diff"]
                            )
        
        results["layer_summary"] = {}
        
        for layer_idx in sorted(layer_stats.keys()):
            stats_dict = layer_stats[layer_idx]
            
            summary = {
                "mean_cos_sim": np.mean(stats_dict["cos_sims"]) if stats_dict["cos_sims"] else 0,
                "mean_frob_diff": np.mean(stats_dict["frob_diffs"]) if stats_dict["frob_diffs"] else 0,
                "mean_entropy_diff": np.mean(stats_dict["entropy_diffs"]) if stats_dict["entropy_diffs"] else 0,
                "mean_last_token_diff": np.mean(stats_dict["last_token_diffs"]) if stats_dict["last_token_diffs"] else 0
            }
            
            results["layer_summary"][str(layer_idx)] = summary
            
            print(f"Layer {layer_idx}:")
            print(f"  Mean attention cosine similarity: {summary['mean_cos_sim']:.4f}")
            print(f"  Mean Frobenius difference: {summary['mean_frob_diff']:.4f}")
            print(f"  Mean entropy difference (correct - incorrect): {summary['mean_entropy_diff']:.6f}")
            print(f"  Mean last-token attention difference: {summary['mean_last_token_diff']:.6f}")
        
        # Key insight: Do attention patterns diverge more than hidden states?
        min_cos_sim = min(
            summary.get("mean_cos_sim", 1.0) 
            for summary in results["layer_summary"].values()
        )
        attention_divergence = 1 - min_cos_sim
        
        results["max_attention_divergence"] = attention_divergence
        results["min_attention_cos_sim"] = min_cos_sim
        
        print(f"\nMaximum attention divergence: {attention_divergence:.4f}")
        print(f"(Compare to hidden state divergence of ~0.005)")
        
        # Verdict
        if attention_divergence > 0.1:  # 10% attention divergence
            results["verdict"] = "BREAKTHROUGH: Attention patterns diverge significantly more than hidden states"
            results["finding"] = "Attention fingerprints may be a scale-invariant indicator of reasoning quality"
        elif attention_divergence > 0.01:
            results["verdict"] = "PROMISING: Moderate attention divergence found"
            results["finding"] = "Attention patterns show more differentiation than hidden states"
        else:
            results["verdict"] = "NEGATIVE: Attention patterns are also highly similar"
            results["finding"] = "Both hidden states and attention converge on similar patterns"
        
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
    output_path = Path("/home/me/eas/eas/analysis/results/attention_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    run_attention_analysis()
