#!/usr/bin/env python3
"""
pre_mortem_2_error_prediction.py

Tests whether activations can predict if the model will be correct.
PASS: Probe accuracy > 55%
FAIL: Probe accuracy ≈ 50% (random chance)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import argparse
import json
from datetime import datetime


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_activation(model, tokenizer, text, layer=-1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


def get_model_answer(model, tokenizer, prompt, n_options, device):
    """Get model's MCQ answer by comparing option probabilities."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    probs = []
    for i, opt in enumerate(["A", "B", "C", "D"][:n_options]):
        token_ids = tokenizer.encode(f" {opt}", add_special_tokens=False)
        if token_ids:
            prob = F.softmax(logits, dim=-1)[token_ids[0]].item()
        else:
            prob = 0.0
        probs.append(prob)
    
    return probs.index(max(probs)) if probs else 0


def load_hellaswag(n=500):
    """Load HellaSwag for testing."""
    print("Loading HellaSwag dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= n:
            break
        
        prompt = f"Context: {ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(ex['endings'])])
        prompt += "\nAnswer:"
        
        samples.append({
            "prompt": prompt,
            "correct_idx": int(ex["label"]),
            "activation_prompt": ex["ctx"]  # Use context for activation
        })
    
    return samples


def run_pre_mortem_2(n_samples=500, layer=-1, model_name="gpt2"):
    start_time = datetime.now()
    
    print("=" * 60)
    print("PRE-MORTEM TEST 2: Error Prediction from Activations")
    print("=" * 60)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Samples: {n_samples}")
    print(f"Layer: {layer}")
    
    # Load model
    print(f"\nLoading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    samples = load_hellaswag(n=n_samples)
    print(f"Loaded {len(samples)} samples")
    
    # Collect activations and correctness labels
    print("\nCollecting activations and predictions...")
    activations = []
    labels = []
    
    for sample in tqdm(samples, desc="Processing"):
        try:
            # Get activation BEFORE seeing options
            act = extract_activation(model, tokenizer, sample["activation_prompt"], layer)
            
            # Get model's answer
            pred_idx = get_model_answer(model, tokenizer, sample["prompt"], 4, device)
            is_correct = int(pred_idx == sample["correct_idx"])
            
            activations.append(act)
            labels.append(is_correct)
        except Exception as e:
            continue
    
    X = np.vstack(activations)
    y = np.array(labels)
    
    correct_rate = sum(y) / len(y)
    print(f"\nModel accuracy: {correct_rate:.1%} ({sum(y)}/{len(y)})")
    
    # Check if we have enough of both classes
    if sum(y) < 20 or sum(1-y) < 20:
        print("WARNING: Imbalanced classes, results may be unreliable")
    
    # Train probe with cross-validation
    print("\nTraining error prediction probe...")
    
    # Use pipeline with scaling
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Stratified K-fold to handle class imbalance
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    
    avg_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    
    # Also compute AUC if possible
    try:
        from sklearn.metrics import roc_auc_score, make_scorer
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        avg_auc = np.mean(auc_scores)
    except:
        avg_auc = None
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Probe accuracy: {avg_accuracy:.3f} (±{std_accuracy:.3f})")
    if avg_auc:
        print(f"Probe AUC: {avg_auc:.3f}")
    print(f"Baseline (random): 0.500")
    print(f"Lift over random: {(avg_accuracy - 0.5) * 100:.1f} percentage points")
    print(f"Model accuracy on task: {correct_rate:.1%}")
    print(f"Duration: {duration:.1f} minutes")
    
    # Pass/Fail
    PASS_THRESHOLD = 0.55
    INCONCLUSIVE_THRESHOLD = 0.52
    
    if avg_accuracy > PASS_THRESHOLD:
        result = "PASS"
        msg = "Activations encode correctness. Proceed with self-aware uncertainty."
    elif avg_accuracy > INCONCLUSIVE_THRESHOLD:
        result = "INCONCLUSIVE"
        msg = "Weak signal. May need more data or different layer."
    else:
        result = "FAIL"
        msg = "Activations don't predict correctness. Abandon this direction."
    
    print(f"\n>>> {result}: {msg}")
    print("=" * 60)
    
    # Save results
    results = {
        "date": datetime.now().isoformat(),
        "duration_minutes": duration,
        "model": model_name,
        "n_samples": len(activations),
        "layer": layer,
        "model_accuracy": float(correct_rate),
        "probe_accuracy": float(avg_accuracy),
        "probe_std": float(std_accuracy),
        "probe_auc": float(avg_auc) if avg_auc else None,
        "result": result,
        "pass_threshold": PASS_THRESHOLD
    }
    
    with open("results/pre_mortems/pm2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--model", type=str, default="gpt2")
    args = parser.parse_args()
    
    import os
    os.makedirs("results/pre_mortems", exist_ok=True)
    
    run_pre_mortem_2(n_samples=args.n_samples, layer=args.layer, model_name=args.model)
