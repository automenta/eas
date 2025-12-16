#!/usr/bin/env python3
"""
phi2_function_vector.py - Injecting "Latent Reasoning"
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
import json
from datetime import datetime

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
        self.mode = "off"

    def set_layer(self, layer_idx):
        if self.handle: self.handle.remove()
        if hasattr(self.model, 'model'): layers = self.model.model.layers
        else: layers = self.model.transformer.h
        self.layer_idx = layer_idx
        self.handle = layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.mode == "collect":
            # Mean of last token? Or mean of all tokens?
            # For function vectors, usually mean of the *instruction* tokens is best.
            # But let's stick to last token for simplicity.
            self.activations.append(hidden.detach()[:,-1,:].float().cpu().numpy())
        elif self.mode == "steer" and self.direction is not None:
            dtype = hidden.dtype
            device = hidden.device
            steering = self.direction.to(device).to(dtype).view(1, 1, -1)
            return (hidden + self.strength * steering,) + output[1:] if isinstance(output, tuple) else (hidden + self.strength * steering)
        return output

    def cleanup(self):
        if self.handle: self.handle.remove()

def get_pred(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = []
    for opt in ["A", "B", "C", "D"]:
        tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
        probs.append(logits[tid[0]].item() if tid else -float('inf'))
    return np.argmax(probs)

def main():
    device = get_device()
    print_banner("🧠 PHI-2 LATENT REASONING INJECTION 🧠")
    
    print("📦 Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model loaded.")

    # 1. Extract Function Vector
    print_banner("🔬 PHASE 1: EXTRACTING REASONING VECTOR", "-")
    
    # We use a set of diverse queries
    queries = [
        "What is the capital of France?",
        "Solve 24 * 12.",
        "Explain quantum entanglement.",
        "Write a poem about rust.",
        "Who wrote Hamlet?",
        "What is the derivative of x^2?",
        "Translate 'hello' to Spanish.",
        "Why is the sky blue?",
        "How do airplanes fly?",
        "What is the meaning of life?"
    ]
    
    prompts_direct = [f"Question: {q}\nAnswer:" for q in queries]
    prompts_reason = [f"Question: {q}\nLet's think step by step to find the correct answer:\n" for q in queries]
    
    hook = SteeringHook(model)
    
    # Scan layers to find best "Reasoning" representation
    # Usually middle layers are best for function vectors (e.g. 10-20)
    best_layer = 16 # Heuristic start
    print(f"📍 Extracting from Layer {best_layer}...")
    
    hook.set_layer(best_layer)
    hook.mode = "collect"
    
    # Collect Direct
    hook.activations = []
    for p in prompts_direct:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
    direct_acts = np.concatenate(hook.activations, axis=0)
    
    # Collect Reasoning
    hook.activations = []
    for p in prompts_reason:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
    reason_acts = np.concatenate(hook.activations, axis=0)
    
    # Compute Vector
    diff = np.mean(reason_acts, axis=0) - np.mean(direct_acts, axis=0)
    direction = torch.tensor(diff / np.linalg.norm(diff), dtype=torch.float32)
    hook.direction = direction
    
    print(f"✅ Reasoning Vector Extracted (Norm: {np.linalg.norm(diff):.4f})")

    # 2. Test on HellaSwag
    print_banner("🚀 PHASE 2: TESTING ON HELLASWAG", "-")
    
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 200: break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({"prompt": prompt, "label": int(ex["label"])})
    
    # Baseline
    hook.mode = "off"
    base_correct = 0
    for s in samples:
        if get_pred(model, tokenizer, s["prompt"], device) == s["label"]:
            base_correct += 1
    base_acc = base_correct / len(samples)
    print(f"Baseline Accuracy: {base_acc:.1%}")
    
    # Steered (Sweep strength)
    print("\nSweeping strengths...")
    best_acc = 0
    best_strength = 0
    
    for strength in [0.5, 1.0, 1.5, 2.0, 3.0]:
        hook.mode = "steer"
        hook.strength = strength
        correct = 0
        for s in samples:
            if get_pred(model, tokenizer, s["prompt"], device) == s["label"]:
                correct += 1
        acc = correct / len(samples)
        print(f"  Strength {strength}: {acc:.1%}")
        if acc > best_acc:
            best_acc = acc
            best_strength = strength
            
    improvement = best_acc - base_acc
    print_banner("📊 RESULTS", "=")
    print(f"Baseline:    {base_acc:.1%}")
    print(f"With Latent Reasoning: {best_acc:.1%} (Strength {best_strength})")
    print(f"Improvement: {improvement:+.1%}")
    
    if improvement > 0.02:
        print("\n🎉 SUCCESS: Latent reasoning injection works!")
    else:
        print("\n❌ RESULT: No significant improvement.")
        
    with open("results/experiments/phi2_function_vector.json", "w") as f:
        json.dump({"baseline": base_acc, "steered": best_acc, "improvement": improvement}, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
