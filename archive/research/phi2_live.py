#!/usr/bin/env python3
"""
phi2_live.py - Phi-2 experiment with LIVE animated feedback
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
import json
from datetime import datetime
import sys
import time


def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_progress(current, total, correct, label="Progress"):
    pct = current / total * 100
    acc = correct / current * 100 if current > 0 else 0
    bar_len = 40
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r{label}: [{bar}] {current}/{total} ({pct:.0f}%) | Accuracy: {correct}/{current} ({acc:.1f}%)", end="", flush=True)


def print_sample_result(prompt_preview, pred, correct, is_correct):
    emoji = "✓" if is_correct else "✗"
    color = "\033[92m" if is_correct else "\033[91m"
    reset = "\033[0m"
    print(f"\n  {color}{emoji}{reset} Predicted: {chr(65+pred)}, Correct: {chr(65+correct)} | \"{prompt_preview[:50]}...\"")


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActivationHook:
    def __init__(self, model, layer_idx):
        self.activation = None
        self.direction = None
        self.strength = 0.0
        self.mode = "collect"
        
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            layers = model.transformer.h
        
        self._hook = layers[layer_idx].register_forward_hook(self._fn)
    
    def _fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        if self.mode == "collect":
            self.activation = hidden.detach().mean(dim=1).squeeze().cpu().numpy()
        elif self.mode == "steer" and self.direction is not None:
            direction = self.direction.unsqueeze(0).unsqueeze(0)
            direction = direction.expand(hidden.shape[0], hidden.shape[1], -1)
            modified = hidden + self.strength * direction
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        return output
    
    def cleanup(self):
        self._hook.remove()


def get_pred(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = []
    for opt in ["A", "B", "C", "D"]:
        tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
        probs.append(F.softmax(logits, dim=-1)[tid[0]].item() if tid else 0)
    
    return probs.index(max(probs))


def main():
    device = get_device()
    
    print_banner("🧠 PHI-2 ACTIVATION STEERING EXPERIMENT 🧠")
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"💻 Device: {device}")
    
    # Load model with progress
    print_banner("LOADING PHI-2 MODEL", "-")
    print("📦 Downloading model weights... (this may take a minute)")
    
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print(f"✅ Model loaded in {time.time()-start:.1f}s")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    # Load data
    print_banner("LOADING DATASET", "-")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 300:
            break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({
            "prompt": prompt,
            "context": ex["ctx"],
            "correct_idx": int(ex["label"])
        })
    
    print(f"✅ Loaded {len(samples)} samples")
    
    train_samples = samples[:150]
    test_samples = samples[150:]
    
    # Phase 1: Learn patterns
    print_banner("🔬 PHASE 1: LEARNING PATTERNS", "-")
    print("Collecting activations from correct vs incorrect predictions...\n")
    
    n_layers = len(model.model.layers)
    layer_idx = n_layers // 2
    print(f"📍 Hooking layer {layer_idx}/{n_layers}")
    
    hook = ActivationHook(model, layer_idx)
    hook.mode = "collect"
    
    correct_acts = []
    incorrect_acts = []
    
    for i, sample in enumerate(train_samples):
        # Get activation
        inputs = tokenizer(sample["context"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)
        act = hook.activation
        
        # Get prediction
        pred = get_pred(model, tokenizer, sample["prompt"], device)
        is_correct = pred == sample["correct_idx"]
        
        if is_correct:
            correct_acts.append(act)
        else:
            incorrect_acts.append(act)
        
        # Live update
        total_correct = len(correct_acts)
        total_incorrect = len(incorrect_acts)
        acc = total_correct / (i+1) * 100
        
        bar_len = 30
        filled = int(bar_len * (i+1) / len(train_samples))
        bar = "█" * filled + "░" * (bar_len - filled)
        
        emoji = "✓" if is_correct else "✗"
        print(f"\r[{bar}] {i+1}/{len(train_samples)} | ✓{total_correct} ✗{total_incorrect} | Acc: {acc:.1f}% | Last: {emoji}", end="", flush=True)
    
    print(f"\n\n📊 Training stats:")
    print(f"   ✓ Correct predictions: {len(correct_acts)}")
    print(f"   ✗ Incorrect predictions: {len(incorrect_acts)}")
    print(f"   📈 Training accuracy: {len(correct_acts)/len(train_samples)*100:.1f}%")
    
    # Compute direction
    print_banner("🧮 COMPUTING STEERING DIRECTION", "-")
    
    correct_mean = np.mean(correct_acts, axis=0)
    incorrect_mean = np.mean(incorrect_acts, axis=0)
    direction = correct_mean - incorrect_mean
    magnitude = np.linalg.norm(direction)
    direction = direction / magnitude
    
    print(f"📐 Direction magnitude: {magnitude:.3f}")
    print(f"✅ Steering direction computed!")
    
    direction_tensor = torch.tensor(direction, device=device, dtype=torch.float16)
    
    # Phase 2: Test steering
    print_banner("🚀 PHASE 2: TESTING INTERVENTION", "-")
    print("Comparing baseline vs steered predictions...\n")
    
    baseline_correct = 0
    steered_correct = 0
    flipped_to_correct = 0
    flipped_to_wrong = 0
    
    hook.direction = direction_tensor
    hook.strength = 0.5
    
    for i, sample in enumerate(test_samples):
        # Baseline
        hook.mode = "collect"
        pred_base = get_pred(model, tokenizer, sample["prompt"], device)
        base_correct = pred_base == sample["correct_idx"]
        if base_correct:
            baseline_correct += 1
        
        # Steered
        hook.mode = "steer"
        pred_steer = get_pred(model, tokenizer, sample["prompt"], device)
        steer_correct = pred_steer == sample["correct_idx"]
        if steer_correct:
            steered_correct += 1
        
        # Track flips
        if not base_correct and steer_correct:
            flipped_to_correct += 1
        if base_correct and not steer_correct:
            flipped_to_wrong += 1
        
        # Live update
        n = i + 1
        base_acc = baseline_correct / n * 100
        steer_acc = steered_correct / n * 100
        diff = steer_acc - base_acc
        
        bar_len = 30
        filled = int(bar_len * n / len(test_samples))
        bar = "█" * filled + "░" * (bar_len - filled)
        
        diff_str = f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"
        emoji = "📈" if diff > 0 else ("📉" if diff < 0 else "➡️")
        
        print(f"\r[{bar}] {n}/{len(test_samples)} | Base: {base_acc:.1f}% | Steer: {steer_acc:.1f}% | Δ: {diff_str} {emoji}", end="", flush=True)
    
    hook.cleanup()
    
    # Final results
    n_test = len(test_samples)
    base_acc = baseline_correct / n_test
    steer_acc = steered_correct / n_test
    improvement = steer_acc - base_acc
    
    print_banner("📊 FINAL RESULTS", "=")
    
    print(f"  🎯 Baseline accuracy:    {base_acc:.1%} ({baseline_correct}/{n_test})")
    print(f"  🚀 Steered accuracy:     {steer_acc:.1%} ({steered_correct}/{n_test})")
    print(f"  ────────────────────────────────")
    
    if improvement > 0:
        print(f"  📈 IMPROVEMENT:          +{improvement:.1%}")
        print(f"\n  ✅ Flipped wrong→right:  {flipped_to_correct}")
        print(f"  ⚠️  Flipped right→wrong: {flipped_to_wrong}")
        print(f"  📊 Net flips:            +{flipped_to_correct - flipped_to_wrong}")
        
        if improvement >= 0.05:
            print_banner("🎉 SUCCESS! Significant improvement detected!", "🎉")
        else:
            print_banner("📈 Partial success - small improvement", "~")
    else:
        print(f"  📉 Change:               {improvement:.1%}")
        print_banner("❌ No improvement detected", "-")
    
    print(f"\n⏰ Completed: {datetime.now().strftime('%H:%M:%S')}")
    
    # Save
    results = {
        "baseline_accuracy": base_acc,
        "steered_accuracy": steer_acc,
        "improvement": improvement,
        "flipped_correct": flipped_to_correct,
        "flipped_wrong": flipped_to_wrong
    }
    
    with open("results/experiments/phi2_live.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
