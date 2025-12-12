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

def run_experiment():
    print("="*60)
    print("REPRODUCING CONTEXT-ALIGNED RAW EAS (+16% Claim)")
    print("="*60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2" # Using standard GPT-2 as requested (or small for speed if needed)
    # Memory mentions "GPT-2" (likely 124M)

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Configuration from memory
    # Layer: "Layer 2 is preferred for deeper models like GPT-2... in Few-Shot settings"
    target_layer = 2
    watcher = RawSpaceWatcher(dim=model.config.n_embd, k=20, alpha=0.5).to(device)

    dataset = get_logic_dataset()
    # Split into Warmup (Context) and Test
    # "Warmup" examples must be successful ones.

    print("Preparing Data...")
    subset = dataset.select(range(100)) # Small subset for PoC speed
    warmup_data = dataset.select(range(100, 150))

    # --- PHASE 1: Warmup (Context Alignment) ---
    print("\n[Phase 1] Warming up Watcher on Context...")
    # We run the model on warmup examples, collect activations at target layer for *correct* answers

    activations = []

    for i in tqdm(range(len(warmup_data))):
        ex = warmup_data[i]
        prompt = format_prompt_3shot(ex['question'], ex['options'])
        correct_char = "ABCD"[ex['answer']]
        # Append correct answer to prompt to get its activation
        full_text = prompt + " " + correct_char

        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get activation of the *last token* (the answer token) at target layer
            hidden = outputs.hidden_states[target_layer + 1] # +1 because 0 is embeddings
            activations.append(hidden[:, -1, :]) # [1, dim]

    # Batch update
    if activations:
        all_hidden = torch.cat(activations, dim=0).unsqueeze(1) # [N, 1, dim]
        watcher.adapt(all_hidden)
        print(f"Watcher warmed up with {len(activations)} examples.")

    # --- PHASE 2: Evaluation ---
    print("\n[Phase 2] Evaluating Baseline vs EAS...")

    results = {"baseline": 0, "eas": 0, "total": 0}

    for i in tqdm(range(len(subset))):
        ex = subset[i]
        prompt = format_prompt_3shot(ex['question'], ex['options'])
        correct_char = "ABCD"[ex['answer']]

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 1. Baseline Run
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Simple greedy decode of A, B, C, D
            cand_ids = [tokenizer.encode(" " + c)[0] for c in "ABCD"]
            cand_logits = logits[cand_ids]
            pred_idx = cand_logits.argmax().item()
            pred_char = "ABCD"[pred_idx]

            if pred_char == correct_char:
                results["baseline"] += 1

        # 2. EAS Run
        # We need to hook the model to intervene at target_layer
        # Using a simple forward hook

        def eas_hook(module, input, output):
            # GPT-2 output is a tuple where the first element is the hidden state
            # but sometimes it's just the tensor depending on config.
            # However, modifying the tuple directly can cause issues if downstream expects specific types.

            if isinstance(output, tuple):
                h = output[0]
                print(f"Hook input shape: {h.shape}")
                steered = watcher.steer(h)
                print(f"Hook output shape: {steered.shape}")
                # Reconstruct tuple carefully
                return (steered,) + output[1:]
            else:
                return watcher.steer(output)

        # Register hook
        layer_module = model.transformer.h[target_layer]
        handle = layer_module.register_forward_hook(eas_hook)

        try:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]

                cand_logits = logits[cand_ids]
                pred_idx = cand_logits.argmax().item()
                pred_char = "ABCD"[pred_idx]

                if pred_char == correct_char:
                    results["eas"] += 1
        finally:
            handle.remove()

        results["total"] += 1

    # --- Report ---
    baseline_acc = results["baseline"] / results["total"]
    eas_acc = results["eas"] / results["total"]
    delta = eas_acc - baseline_acc

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Total Samples: {results['total']}")
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    print(f"EAS Accuracy:      {eas_acc:.2%}")
    print(f"Improvement:       {delta:+.2%}")
    print("="*60)

    if delta > 0.05:
        print("✅ SUCCESS: Significant positive transfer observed!")
    elif delta > 0:
        print("⚠️ MARGINAL: Slight positive transfer.")
    else:
        print("❌ FAILURE: No improvement or degradation.")

if __name__ == "__main__":
    run_experiment()
