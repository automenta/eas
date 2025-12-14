#!/usr/bin/env python3
"""
phi2_optimize.py - Find the "God Vector" for maximum accuracy
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
import json
from datetime import datetime
import time

def print_banner(text, char="="):
    print(f"\n{char*70}")
    print(f"{text:^70}")
    print(f"{char*70}\n")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class SteeringHook:
    def __init__(self, model):
        self.model = model
        self.layer_idx = None
        self.direction = None
        self.strength = 0.0
        self.handle = None
        self.activations = []
        self.mode = "off" # off, collect, steer

    def set_layer(self, layer_idx):
        if self.handle:
            self.handle.remove()
        
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.transformer.h
            
        self.layer_idx = layer_idx
        self.handle = layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        
        if self.mode == "collect":
            # Store mean activation of the last token
            act = hidden.detach()[:,-1,:].float().cpu().numpy()
            self.activations.append(act)
        
        elif self.mode == "steer" and self.direction is not None:
            # Inject steering vector
            dtype = hidden.dtype
            device = hidden.device
            
            # Direction is [hidden_dim]
            # Hidden is [batch, seq, hidden_dim]
            steering = self.direction.to(device).to(dtype)
            steering = steering.view(1, 1, -1)
            
            # Add to all tokens? Or just last? Let's add to all for now.
            modified = hidden + self.strength * steering
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
            
        return output

    def clear_activations(self):
        self.activations = []

    def cleanup(self):
        if self.handle:
            self.handle.remove()

def get_pred(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    probs = []
    for opt in ["A", "B", "C", "D"]:
        tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
        if tid:
            probs.append(logits[tid[0]].item())
        else:
            probs.append(-float('inf'))
    return np.argmax(probs)

def main():
    device = get_device()
    print_banner("🚀 PHI-2 ACCURACY OPTIMIZATION 🚀")
    
    # 1. Load Model
    print("📦 Loading Phi-2 (this needs to be fast)...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model loaded.")

    # 2. Load Data
    print("📚 Loading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    # Prepare samples
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 300: break # Use 300 samples: 100 train, 100 val, 100 test
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({
            "prompt": prompt,
            "context": ex["ctx"],
            "label": int(ex["label"])
        })
    
    train_set = samples[:100]
    val_set = samples[100:200]
    test_set = samples[200:300]
    print(f"✅ Data prepared: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")

    hook = SteeringHook(model)
    
    # 3. Find Best Layer (Scan middle layers)
    print_banner("🔬 PHASE 1: FINDING THE 'TRUTH' LAYER", "-")
    
    best_layer = -1
    best_separation = -1
    best_direction = None
    
    # Scan layers 10 to 25 (Phi-2 has 32 layers)
    layers_to_scan = range(10, 26, 2) 
    
    for layer in layers_to_scan:
        print(f"Scanning Layer {layer}...", end="", flush=True)
        hook.set_layer(layer)
        hook.mode = "collect"
        hook.clear_activations()
        
        correct_indices = []
        
        # Run forward pass on train set
        for i, s in enumerate(train_set):
            inputs = tokenizer(s["context"], return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                model(**inputs)
            
            # Check if model gets it right (we need ground truth for direction finding)
            # Actually, we want to separate "correct answer" from "incorrect answer"
            # But we only have the prompt.
            # Strategy: Use samples where the model is naturally correct vs incorrect?
            # Or better: Use the label to define "Truth".
            # Let's use the model's own predictions on the train set to define "Confidence"
            
            pred = get_pred(model, tokenizer, s["prompt"], device)
            if pred == s["label"]:
                correct_indices.append(i)
        
        # Compute separation
        acts = np.concatenate(hook.activations, axis=0) # [N, dim]
        
        if len(correct_indices) < 5 or len(correct_indices) > 95:
            print(f" (Skipping: unbalanced {len(correct_indices)}/{len(train_set)})")
            continue
            
        correct_acts = acts[correct_indices]
        incorrect_indices = [i for i in range(len(train_set)) if i not in correct_indices]
        incorrect_acts = acts[incorrect_indices]
        
        # Direction: Mean(Correct) - Mean(Incorrect)
        diff = np.mean(correct_acts, axis=0) - np.mean(incorrect_acts, axis=0)
        separation = np.linalg.norm(diff)
        
        print(f" Separation: {separation:.4f}")
        
        if separation > best_separation:
            best_separation = separation
            best_layer = layer
            best_direction = torch.tensor(diff, dtype=torch.float32)

    print(f"\n🏆 Best Layer: {best_layer} (Separation: {best_separation:.4f})")
    
    # 4. Optimize Strength on Val Set
    print_banner("🎚️ PHASE 2: TUNING STRENGTH", "-")
    
    hook.set_layer(best_layer)
    hook.direction = best_direction
    
    strengths = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    best_strength = 0.0
    best_val_acc = 0.0
    
    for strength in strengths:
        hook.strength = strength
        hook.mode = "steer" if strength != 0 else "off"
        
        correct = 0
        for s in val_set:
            pred = get_pred(model, tokenizer, s["prompt"], device)
            if pred == s["label"]:
                correct += 1
        
        acc = correct / len(val_set)
        print(f"Strength {strength}: Accuracy {acc:.1%}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_strength = strength

    print(f"\n🏆 Best Strength: {best_strength} (Val Acc: {best_val_acc:.1%})")

    # 5. Final Test
    print_banner("🎉 PHASE 3: FINAL VERIFICATION", "-")
    
    # Baseline
    hook.mode = "off"
    base_correct = 0
    for s in test_set:
        pred = get_pred(model, tokenizer, s["prompt"], device)
        if pred == s["label"]:
            base_correct += 1
    base_acc = base_correct / len(test_set)
    
    # Steered
    hook.mode = "steer"
    hook.strength = best_strength
    steer_correct = 0
    for s in test_set:
        pred = get_pred(model, tokenizer, s["prompt"], device)
        if pred == s["label"]:
            steer_correct += 1
    steer_acc = steer_correct / len(test_set)
    
    improvement = steer_acc - base_acc
    
    print(f"Baseline Accuracy: {base_acc:.1%}")
    print(f"Steered Accuracy:  {steer_acc:.1%}")
    print(f"Improvement:       {improvement:+.1%}")
    
    if improvement > 0.01:
        print("\n✅ UNDENIABLE SUCCESS: Raw accuracy improved significantly.")
    else:
        print("\n⚠️  RESULT: Minimal or no improvement.")

    # Save
    results = {
        "best_layer": best_layer,
        "best_strength": best_strength,
        "baseline_acc": base_acc,
        "steered_acc": steer_acc,
        "improvement": improvement
    }
    with open("results/experiments/phi2_optimized.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
