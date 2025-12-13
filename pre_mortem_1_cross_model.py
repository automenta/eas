#!/usr/bin/env python3
"""
pre_mortem_1_cross_model.py

Tests whether GPT-2 and Pythia-70M have similar activation geometry.
PASS: Average correlation > 0.3 after learned projection
FAIL: Correlation < 0.1
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
from datetime import datetime


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_activation(model, tokenizer, text, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


def load_prompts(n=1000):
    """Load diverse prompts from Wikitext."""
    print("Loading Wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    prompts = [t.strip() for t in dataset["text"] if len(t.strip()) > 50]
    return prompts[:n]


def run_pre_mortem_1(n_samples=500, layer=-1):
    start_time = datetime.now()
    
    print("=" * 60)
    print("PRE-MORTEM TEST 1: Cross-Model Activation Similarity")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Samples: {n_samples}")
    print(f"Layer: {layer}")
    
    # Load models
    print("\nLoading GPT-2...")
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    
    print("Loading Pythia-70M...")
    pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-70m").to(device).eval()
    pythia_tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    pythia_tok.pad_token = pythia_tok.eos_token
    
    # Load prompts
    prompts = load_prompts(n=n_samples)
    print(f"Loaded {len(prompts)} prompts")
    
    # Extract activations
    print("\nExtracting activations...")
    gpt2_acts = []
    pythia_acts = []
    
    for prompt in tqdm(prompts, desc="Processing prompts"):
        try:
            g_act = extract_activation(gpt2, gpt2_tok, prompt, layer)
            p_act = extract_activation(pythia, pythia_tok, prompt, layer)
            gpt2_acts.append(g_act)
            pythia_acts.append(p_act)
        except Exception as e:
            continue
    
    X = np.vstack(gpt2_acts)   # [n_samples, gpt2_dim=768]
    Y = np.vstack(pythia_acts)  # [n_samples, pythia_dim=512]
    
    print(f"\nGPT-2 activations: {X.shape}")
    print(f"Pythia activations: {Y.shape}")
    
    # Train/test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    
    # Learn projection from GPT-2 -> Pythia space
    print("\nLearning projection (GPT-2 -> Pythia)...")
    projector = Ridge(alpha=1.0)
    projector.fit(X_train, Y_train)
    
    # Predict and correlate
    Y_pred = projector.predict(X_test)
    
    # Compute per-dimension correlation
    correlations = []
    for i in range(min(Y_test.shape[1], Y_pred.shape[1])):
        try:
            r, _ = pearsonr(Y_test[:, i], Y_pred[:, i])
            if not np.isnan(r):
                correlations.append(r)
        except:
            continue
    
    avg_correlation = np.mean(correlations)
    std_correlation = np.std(correlations)
    
    # Also test reverse direction (Pythia -> GPT-2)
    print("Learning projection (Pythia -> GPT-2)...")
    projector_rev = Ridge(alpha=1.0)
    projector_rev.fit(Y_train, X_train)
    X_pred = projector_rev.predict(Y_test)
    
    correlations_rev = []
    for i in range(min(X_test.shape[1], X_pred.shape[1])):
        try:
            r, _ = pearsonr(X_test[:, i], X_pred[:, i])
            if not np.isnan(r):
                correlations_rev.append(r)
        except:
            continue
    
    avg_correlation_rev = np.mean(correlations_rev)
    
    # Bidirectional average
    bidirectional_corr = (avg_correlation + avg_correlation_rev) / 2
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"GPT-2 -> Pythia correlation: {avg_correlation:.3f} (±{std_correlation:.3f})")
    print(f"Pythia -> GPT-2 correlation: {avg_correlation_rev:.3f}")
    print(f"Bidirectional average: {bidirectional_corr:.3f}")
    print(f"Dimensions with r > 0.3: {sum(1 for c in correlations if c > 0.3)}/{len(correlations)}")
    print(f"Dimensions with r > 0.5: {sum(1 for c in correlations if c > 0.5)}/{len(correlations)}")
    print(f"Duration: {duration:.1f} minutes")
    
    # Pass/Fail
    PASS_THRESHOLD = 0.3
    FAIL_THRESHOLD = 0.1
    
    if bidirectional_corr > PASS_THRESHOLD:
        result = "PASS"
        msg = "Cross-model activation structure exists. Proceed with transfer experiments."
    elif bidirectional_corr < FAIL_THRESHOLD:
        result = "FAIL"
        msg = "No cross-model structure detected. Abandon this direction."
    else:
        result = "INCONCLUSIVE"
        msg = "Weak signal. Consider more data or different models."
    
    print(f"\n>>> {result}: {msg}")
    print("=" * 60)
    
    # Save results
    results = {
        "date": datetime.now().isoformat(),
        "duration_minutes": duration,
        "n_samples": len(gpt2_acts),
        "layer": layer,
        "gpt2_pythia_correlation": float(avg_correlation),
        "pythia_gpt2_correlation": float(avg_correlation_rev),
        "bidirectional_correlation": float(bidirectional_corr),
        "std_correlation": float(std_correlation),
        "result": result,
        "pass_threshold": PASS_THRESHOLD,
        "fail_threshold": FAIL_THRESHOLD
    }
    
    with open("results/pre_mortems/pm1_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--layer", type=int, default=-1)
    args = parser.parse_args()
    
    import os
    os.makedirs("results/pre_mortems", exist_ok=True)
    
    run_pre_mortem_1(n_samples=args.n_samples, layer=args.layer)
