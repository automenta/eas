#!/usr/bin/env python3
"""
phi2_sweep.py - Sweep steering strengths to find optimal
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
import json
from datetime import datetime


def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


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
    
    print_banner("🧪 PHI-2 STEERING STRENGTH SWEEP 🧪")
    
    # Load model
    print("📦 Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model loaded")
    
    # Load data
    print("📊 Loading data...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 500:
            break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({"prompt": prompt, "context": ex["ctx"], "correct_idx": int(ex["label"])})
    
    train_samples = samples[:200]
    test_samples = samples[200:400]
    print(f"✅ Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Setup hook
    n_layers = len(model.model.layers)
    layer_idx = n_layers // 2
    hook = ActivationHook(model, layer_idx)
    hook.mode = "collect"
    
    # Phase 1: Collect activations
    print_banner("🔬 PHASE 1: COLLECTING ACTIVATIONS", "-")
    
    correct_acts, incorrect_acts = [], []
    
    for i, sample in enumerate(train_samples):
        inputs = tokenizer(sample["context"], return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            _ = model(**inputs)
        act = hook.activation
        pred = get_pred(model, tokenizer, sample["prompt"], device)
        
        if pred == sample["correct_idx"]:
            correct_acts.append(act)
        else:
            incorrect_acts.append(act)
        
        pct = (i+1) / len(train_samples) * 100
        print(f"\r  Collecting: {i+1}/{len(train_samples)} ({pct:.0f}%) | ✓{len(correct_acts)} ✗{len(incorrect_acts)}", end="")
    
    print(f"\n\n📊 Training accuracy: {len(correct_acts)/len(train_samples)*100:.1f}%")
    
    # Compute direction
    direction = np.mean(correct_acts, axis=0) - np.mean(incorrect_acts, axis=0)
    direction = direction / np.linalg.norm(direction)
    direction_tensor = torch.tensor(direction, device=device, dtype=torch.float16)
    hook.direction = direction_tensor
    
    # Phase 2: Sweep strengths
    print_banner("🚀 PHASE 2: STRENGTH SWEEP", "-")
    
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    results = {}
    
    for strength in strengths:
        hook.strength = strength
        hook.mode = "steer" if strength > 0 else "collect"
        
        correct = 0
        for sample in test_samples:
            pred = get_pred(model, tokenizer, sample["prompt"], device)
            if pred == sample["correct_idx"]:
                correct += 1
        
        acc = correct / len(test_samples) * 100
        results[strength] = acc
        
        bar = "█" * int(acc / 2) + "░" * (50 - int(acc / 2))
        emoji = "🎯" if strength == 0 else "🚀"
        print(f"  {emoji} Strength {strength:.1f}: [{bar}] {acc:.1f}%")
    
    hook.cleanup()
    
    # Find best
    baseline = results[0.0]
    best_strength = max(results.keys(), key=lambda k: results[k] if k > 0 else -1)
    best_acc = results[best_strength]
    improvement = best_acc - baseline
    
    print_banner("📊 SUMMARY", "=")
    print(f"  🎯 Baseline (strength=0):     {baseline:.1f}%")
    print(f"  🏆 Best (strength={best_strength}):     {best_acc:.1f}%")
    print(f"  📈 Max improvement:           {improvement:+.1f}%")
    
    if improvement > 2:
        print_banner("🎉 SUCCESS! Found beneficial steering!", "🎉")
    elif improvement > 0:
        print_banner("📈 Modest improvement found", "~")
    else:
        print_banner("❌ No improvement from steering", "-")
    
    with open("results/experiments/phi2_sweep.json", "w") as f:
        json.dump({"results": results, "best_strength": best_strength, "improvement": improvement}, f, indent=2)


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
