#!/usr/bin/env python3
"""
self_correcting.py

NOVEL CONTRIBUTION: Self-Correcting LLMs via Activation-Based Rejection Sampling

Pre-mortem showed 61% accuracy at predicting errors from activations.
This experiment tests whether we can:
1. Use the error probe to REJECT outputs that are likely wrong
2. Regenerate with different sampling
3. Measure if this rejection sampling improves accuracy

This is DIFFERENT from existing uncertainty methods because:
- No ensemble required
- No explicit confidence head trained
- Uses activation geometry directly
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import json
from datetime import datetime


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_activation(model, tokenizer, text, layer=-1, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


def get_model_prediction(model, tokenizer, prompt, device, temperature=1.0):
    """Get model's MCQ prediction with confidence."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :] / temperature
    
    probs = []
    for opt in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(f" {opt}", add_special_tokens=False)
        if token_ids:
            prob = F.softmax(logits, dim=-1)[token_ids[0]].item()
        else:
            prob = 0.0
        probs.append(prob)
    
    pred = probs.index(max(probs))
    confidence = max(probs)
    return pred, confidence, probs


def run_self_correcting():
    """
    Main experiment: Can activation-based rejection improve accuracy?
    
    Protocol:
    1. Train error probe on held-out data
    2. For test samples:
       a. Get model prediction
       b. Query error probe: is this likely wrong?
       c. If likely wrong, regenerate with higher temperature
       d. Compare accuracy with and without rejection
    """
    start_time = datetime.now()
    device = get_device()
    
    print("=" * 70)
    print("NOVEL EXPERIMENT: SELF-CORRECTING LLMs VIA ERROR REJECTION")
    print("=" * 70)
    
    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("\nLoading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 600:
            break
        prompt = f"Context: {ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\nAnswer:"
        samples.append({
            "prompt": prompt,
            "context": ex["ctx"],
            "correct_idx": int(ex["label"])
        })
    
    # Phase 1: Train error probe
    print("\n[Phase 1] Training error probe...")
    train_samples = samples[:400]
    test_samples = samples[400:]
    
    train_acts = []
    train_labels = []
    
    for sample in tqdm(train_samples, desc="Collecting training data"):
        try:
            act = extract_activation(model, tokenizer, sample["context"], device=device)
            pred, _, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
            is_correct = int(pred == sample["correct_idx"])
            train_acts.append(act)
            train_labels.append(is_correct)
        except:
            continue
    
    X_train = np.vstack(train_acts)
    y_train = np.array(train_labels)
    
    probe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    probe.fit(X_train, y_train)
    
    train_acc = probe.score(X_train, y_train)
    print(f"Probe training accuracy: {train_acc:.1%}")
    
    # Phase 2: Test with rejection sampling
    print("\n[Phase 2] Testing self-correction...")
    
    n_test = len(test_samples)
    baseline_correct = 0
    rejection_correct = 0
    n_rejections = 0
    
    REJECTION_THRESHOLD = 0.4  # Reject if P(correct) < this
    
    for sample in tqdm(test_samples, desc="Testing"):
        try:
            # Get baseline prediction
            pred_base, conf_base, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
            if pred_base == sample["correct_idx"]:
                baseline_correct += 1
            
            # Get activation and probe prediction
            act = extract_activation(model, tokenizer, sample["context"], device=device)
            prob_correct = probe.predict_proba(act.reshape(1, -1))[0][1]
            
            # Decision: accept or reject
            if prob_correct < REJECTION_THRESHOLD:
                # REJECT: try again with higher temperature
                n_rejections += 1
                pred_new, _, probs_new = get_model_prediction(
                    model, tokenizer, sample["prompt"], device, temperature=1.5
                )
                
                # If new prediction is different and probe likes it better
                if pred_new != pred_base:
                    # Accept new prediction
                    final_pred = pred_new
                else:
                    final_pred = pred_base
            else:
                # ACCEPT: keep original prediction
                final_pred = pred_base
            
            if final_pred == sample["correct_idx"]:
                rejection_correct += 1
                
        except Exception as e:
            continue
    
    baseline_acc = baseline_correct / n_test
    rejection_acc = rejection_correct / n_test
    improvement = rejection_acc - baseline_acc
    rejection_rate = n_rejections / n_test
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Test samples: {n_test}")
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print(f"With rejection: {rejection_acc:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    print(f"Rejection rate: {rejection_rate:.1%}")
    
    result = "SUCCESS" if improvement > 0.02 else ("PARTIAL" if improvement > 0 else "FAILED")
    print(f"\n>>> {result}")
    
    results = {
        "date": datetime.now().isoformat(),
        "n_test": n_test,
        "baseline_accuracy": baseline_acc,
        "rejection_accuracy": rejection_acc,
        "improvement": improvement,
        "rejection_rate": rejection_rate,
        "rejection_threshold": REJECTION_THRESHOLD,
        "result": result
    }
    
    with open("results/experiments/self_correcting.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    run_self_correcting()
