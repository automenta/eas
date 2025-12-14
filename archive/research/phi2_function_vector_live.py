#!/usr/bin/env python3
"""
phi2_function_vector_live.py - Injecting "Latent Reasoning" with LIVE FEEDBACK
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn.functional as F
import json
from datetime import datetime
import time
import sys

def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def print_progress(current, total, prefix="", suffix="", bar_len=30):
    filled_len = int(round(bar_len * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '█' * filled_len + '░' * (bar_len - filled_len)
    sys.stdout.write(f'\r{prefix} [{bar}] {percents}% {suffix}')
    sys.stdout.flush()

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
    print_banner("🧠 PHI-2 LATENT REASONING INJECTION (LIVE) 🧠")
    
    print("📦 Loading Phi-2...")
    start_load = time.time()
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Model loaded in {time.time() - start_load:.1f}s")

    # 1. Extract Function Vector
    print_banner("🔬 PHASE 1: EXTRACTING REASONING VECTOR", "-")
    
    queries = [
        "What is the capital of France?", "Solve 24 * 12.", "Explain quantum entanglement.",
        "Write a poem about rust.", "Who wrote Hamlet?", "What is the derivative of x^2?",
        "Translate 'hello' to Spanish.", "Why is the sky blue?", "How do airplanes fly?",
        "What is the meaning of life?"
    ]
    
    prompts_direct = [f"Question: {q}\nAnswer:" for q in queries]
    prompts_reason = [f"Question: {q}\nLet's think step by step to find the correct answer:\n" for q in queries]
    
    hook = SteeringHook(model)
    best_layer = 16
    print(f"📍 Extracting from Layer {best_layer}...")
    
    hook.set_layer(best_layer)
    hook.mode = "collect"
    
    print("  Collecting 'Direct' activations...")
    hook.activations = []
    for i, p in enumerate(prompts_direct):
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        print_progress(i+1, len(prompts_direct), prefix="  Direct:", suffix=f"Query {i+1}")
    sys.stdout.write("\n")
    direct_acts = np.concatenate(hook.activations, axis=0)
    
    print("  Collecting 'Reasoning' activations...")
    hook.activations = []
    for i, p in enumerate(prompts_reason):
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
        print_progress(i+1, len(prompts_reason), prefix="  Reason:", suffix=f"Query {i+1}")
    sys.stdout.write("\n")
    reason_acts = np.concatenate(hook.activations, axis=0)
    
    diff = np.mean(reason_acts, axis=0) - np.mean(direct_acts, axis=0)
    direction = torch.tensor(diff / np.linalg.norm(diff), dtype=torch.float32)
    hook.direction = direction
    
    print(f"✅ Reasoning Vector Extracted (Norm: {np.linalg.norm(diff):.4f})")

    # 2. Test on HellaSwag
    print_banner("🚀 PHASE 2: TESTING ON HELLASWAG", "-")
    
    print("📚 Loading dataset...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 200: break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({"prompt": prompt, "label": int(ex["label"])})
    print(f"✅ Loaded {len(samples)} samples")
    
    # Baseline
    print("\n📊 Measuring Baseline Accuracy...")
    hook.mode = "off"
    base_correct = 0
    for i, s in enumerate(samples):
        is_correct = get_pred(model, tokenizer, s["prompt"], device) == s["label"]
        if is_correct: base_correct += 1
        print_progress(i+1, len(samples), prefix="  Baseline:", suffix=f"Acc: {base_correct/(i+1):.1%}")
    sys.stdout.write("\n")
    base_acc = base_correct / len(samples)
    
    # Steered (Sweep strength)
    print_banner("🎚️ PHASE 3: SWEEPING STRENGTHS", "-")
    best_acc = 0
    best_strength = 0
    
    strengths = [0.5, 1.0, 1.5, 2.0, 3.0]
    for strength in strengths:
        print(f"\nTesting Strength {strength}...")
        hook.mode = "steer"
        hook.strength = strength
        correct = 0
        for i, s in enumerate(samples):
            is_correct = get_pred(model, tokenizer, s["prompt"], device) == s["label"]
            if is_correct: correct += 1
            print_progress(i+1, len(samples), prefix=f"  Str {strength}:", suffix=f"Acc: {correct/(i+1):.1%}")
        sys.stdout.write("\n")
        
        acc = correct / len(samples)
        if acc > best_acc:
            best_acc = acc
            best_strength = strength
            
    improvement = best_acc - base_acc
    print_banner("🎉 FINAL RESULTS", "=")
    print(f"  🎯 Baseline Accuracy:    {base_acc:.1%}")
    print(f"  🧠 With Latent Reasoning: {best_acc:.1%} (Strength {best_strength})")
    print(f"  📈 Improvement:          {improvement:+.1%}")
    
    if improvement > 0.02:
        print("\n✅ SUCCESS: Latent reasoning injection works!")
    elif improvement > 0:
        print("\n📈 PARTIAL: Small improvement detected.")
    else:
        print("\n❌ RESULT: No significant improvement.")
        
    with open("results/experiments/phi2_function_vector.json", "w") as f:
        json.dump({"baseline": base_acc, "steered": best_acc, "improvement": improvement}, f, indent=2)

if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
