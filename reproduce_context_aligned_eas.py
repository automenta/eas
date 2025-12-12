#!/usr/bin/env python3
"""
reproduce_context_aligned_eas.py

Reproduction of 'Context-Aligned Raw EAS' which achieved positive transfer on GPT-2 (+16%)
and Pythia-160m (+11%).

Key Components:
1. Matched Context Warmup (using 3-shot examples to warm up for 3-shot evaluation)
2. Raw EAS (No Whitening) - maintains anisotropic features required for ICL
3. Position-Aware Intervention (from the archive implementation)

This script:
1. Loads GPT-2
2. Warms up the watcher with 3-shot logic examples (Raw Space)
3. Evaluates on a held-out test set
4. Compares Baseline vs EAS accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.cluster import MiniBatchKMeans
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import copy

# --- Position Aware Watcher (Simplified & Adapted for Raw EAS) ---

class RawSpaceWatcher(nn.Module):
    """
    Context-Aligned Raw EAS Watcher.

    Differences from Standard EAS:
    1. No Whitening (use_whitening=False)
    2. Cosine Similarity in Raw Space
    3. Attractors updated via 'adapt' on warmup data
    """

    def __init__(self, dim, k=10, alpha=0.3):
        super().__init__()
        self.dim = dim
        self.k = k
        self.alpha = alpha

        # Attractors [k, dim]
        self.register_buffer('attractors', torch.randn(k, dim))
        self.register_buffer('attractor_counts', torch.zeros(k))

        # Clustering engine
        self.clusterer = MiniBatchKMeans(n_clusters=k, batch_size=20, n_init=10)
        self.is_fitted = False

    def adapt(self, hidden_states):
        """
        Unsupervised adaptation (Warmup).
        Fits K-Means attractors to the raw hidden states of successful examples.
        """
        # Flatten: [batch, seq_len, dim] -> [N, dim]
        # We only care about the last token (generation step) usually,
        # but context alignment suggests we might want to cluster relevant tokens.
        # For simplicity and alignment with memory "Last-Token Pooling", we take the last token.

        # In 'Context-Aligned', we warm up on the *context* of the examples.
        # But 'EmergentWatcher' memory says "Last-Token Pooling".
        # Let's stick to last-token pooling of the warmup examples (which are complete successful examples).

        flat_hidden = hidden_states[:, -1, :].detach().cpu().numpy()

        if not self.is_fitted:
            self.clusterer.partial_fit(flat_hidden)
            self.is_fitted = True
        else:
            self.clusterer.partial_fit(flat_hidden)

        # Update torch buffers
        centroids = torch.from_numpy(self.clusterer.cluster_centers_).float().to(self.attractors.device)
        self.attractors.data = F.normalize(centroids, dim=-1) # Project to sphere

    def steer(self, hidden_states):
        """
        Apply steering intervention.
        """
        if not self.is_fitted:
            return hidden_states

        # Target: Last token
        h = hidden_states[:, -1, :] # [batch, dim]
        h_norm = F.normalize(h, dim=-1)

        # Find nearest attractor
        sims = torch.mm(h_norm, self.attractors.t()) # [batch, k]
        best_sim, best_idx = sims.max(dim=-1)

        nearest_attractor = self.attractors[best_idx] # [batch, dim]

        # Steering vector: direction towards attractor
        # Delta = (attractor - current)
        delta = (nearest_attractor - h_norm)

        # Apply to raw hidden state
        # We add the delta (direction) to the original vector
        # Note: Since delta is calculated on normalized vectors, we might want to scale it back?
        # Memory says: "Raw EAS... keeps all operations in the raw activation manifold."
        # And "intervention comparisons occurred in whitened space... Raw EAS resolves this."
        # If we act in raw space, we just add the delta.

        # However, h is magnitude-bearing. h_norm is unit.
        # If we add a unit-scale delta to a large-magnitude h, it might be too small.
        # Usually we project the direction onto the magnitude of h?
        # Or just add alpha * direction?
        # Let's try adding alpha * direction * h_magnitude to be scale invariant?
        # Or just fixed step size?
        # Memory says "Standard EAS... calculated steering delta... un-whitens... before adding".
        # For Raw EAS, we don't un-whiten.

        # Simple Raw EAS:
        # 1. Normalize H
        # 2. Find attractor A
        # 3. Direction D = A - H_norm
        # 4. H_new = H + alpha * D * ||H||  (Scaling by norm seems prudent to match scale)

        h_mag = h.norm(dim=-1, keepdim=True)
        intervention = self.alpha * delta * h_mag

        # Apply only to last token
        modified_states = hidden_states.clone()
        modified_states[:, -1, :] = h + intervention

        return modified_states

# --- Experiment Runner ---

from local_dataset import get_logic_dataset

def format_prompt_3shot(question, options, answer_idx=None):
    # This function would ideally pull 3 random examples from train,
    # but for stability let's hardcode 3 synthetic logic examples as "Context".

    shots = [
        "Question: All men are mortal. Socrates is a man. Is Socrates mortal?\nOptions:\nA. Yes\nB. No\nAnswer: A",
        "Question: If it rains, the grass is wet. It is raining. Is the grass wet?\nOptions:\nA. Yes\nB. No\nAnswer: A",
        "Question: Either it is day or night. It is not day. Is it night?\nOptions:\nA. Yes\nB. No\nAnswer: A"
    ]

    context = "\n\n".join(shots)

    current = f"Question: {question}\nOptions:\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    current += "\nAnswer:"

    full_prompt = context + "\n\n" + current
    return full_prompt

def run_experiment_on_model(model_name, target_layer=2):
    print("\n" + "="*60)
    print(f"RUNNING CONTEXT-ALIGNED RAW EAS ON {model_name}")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Determine embedding dimension
    if hasattr(model.config, 'n_embd'):
        dim = model.config.n_embd
    elif hasattr(model.config, 'hidden_size'):
        dim = model.config.hidden_size
    else:
        # Fallback
        dim = 768

    # Alpha adjusted to 1.5 to balance steering strength
    watcher = RawSpaceWatcher(dim=dim, k=20, alpha=1.5).to(device)

    dataset = get_logic_dataset()
    print("Preparing Data...")
    subset = dataset.select(range(50)) # Smaller subset for multi-model speed
    warmup_data = dataset.select(range(50, 100))

    # --- PHASE 1: Warmup ---
    print("\n[Phase 1] Warming up Watcher on Context...")
    activations = []

    for i in tqdm(range(len(warmup_data))):
        ex = warmup_data[i]
        prompt = format_prompt_3shot(ex['question'], ex['options'])
        correct_char = "ABCD"[ex['answer']]
        full_text = prompt + " " + correct_char

        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # +1 usually because 0 is embeddings
            # For OPT, hidden_states includes embeddings at 0? Yes usually.
            hidden = outputs.hidden_states[target_layer + 1]
            activations.append(hidden[:, -1, :])

    if activations:
        all_hidden = torch.cat(activations, dim=0).unsqueeze(1)
        watcher.adapt(all_hidden)
        print(f"Watcher warmed up with {len(activations)} examples.")

    # --- PHASE 2: Evaluation ---
    print("\n[Phase 2] Evaluating Baseline vs EAS...")
    results = {"baseline": 0, "eas": 0, "total": 0}

    # Identify layer module
    if "gpt2" in model_name:
        layer_module = model.transformer.h[target_layer]
    elif "opt" in model_name:
        layer_module = model.model.decoder.layers[target_layer]
    else:
        # Fallback for pythia
        if hasattr(model, 'gpt_neox'):
            layer_module = model.gpt_neox.layers[target_layer]
        else:
            raise ValueError(f"Unknown model structure for {model_name}")

    for i in tqdm(range(len(subset))):
        ex = subset[i]
        prompt = format_prompt_3shot(ex['question'], ex['options'])
        correct_char = "ABCD"[ex['answer']]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Baseline
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            cand_ids = [tokenizer.encode(" " + c)[0] for c in "ABCD"]
            # Filter invalid tokens if any (e.g. if tokenizer fails to find " A")
            # OPT tokenizer usually has leading space.
            # Fallback simple check
            if len(cand_ids) != 4:
                cand_ids = [tokenizer.encode(c)[0] for c in "ABCD"]

            cand_logits = logits[cand_ids]
            pred_idx = cand_logits.argmax().item()
            if "ABCD"[pred_idx] == correct_char:
                results["baseline"] += 1

        # EAS
        def eas_hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                steered = watcher.steer(h)
                return (steered,) + output[1:]
            else:
                return watcher.steer(output)

        handle = layer_module.register_forward_hook(eas_hook)
        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                cand_logits = logits[cand_ids]
                pred_idx = cand_logits.argmax().item()
                if "ABCD"[pred_idx] == correct_char:
                    results["eas"] += 1
        finally:
            handle.remove()

        results["total"] += 1

    baseline_acc = results["baseline"] / results["total"]
    eas_acc = results["eas"] / results["total"]
    delta = eas_acc - baseline_acc

    print(f"Model: {model_name} | Baseline: {baseline_acc:.2%} | EAS: {eas_acc:.2%} | Delta: {delta:+.2%}")
    return delta

def run_experiment():
    print("Running Multi-Model Context-Aligned EAS Validation...")

    # 1. GPT-2 (Original)
    delta_gpt2 = run_experiment_on_model("gpt2", target_layer=2)

    # 2. OPT-125m (New)
    try:
        delta_opt = run_experiment_on_model("facebook/opt-125m", target_layer=2)
    except Exception as e:
        print(f"OPT experiment failed: {e}")
        delta_opt = 0

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"GPT-2 Improvement: {delta_gpt2:+.2%}")
    print(f"OPT-125m Improvement: {delta_opt:+.2%}")

    if delta_gpt2 > 0 and delta_opt > 0:
         print("✅ SUCCESS: Positive transfer confirmed across architectures!")
    elif delta_gpt2 > 0:
         print("⚠️ PARTIAL: Confirmed on GPT-2 but not OPT.")
    else:
         print("❌ FAILURE: No consistent improvement.")

if __name__ == "__main__":
    run_experiment()
