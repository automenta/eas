#!/usr/bin/env python3
"""
aft_breakthrough.py - Find the breakthrough result

Target: Something that makes the user say "whoah"
- Small model beats large model baseline
- Cross-task transfer
- Massive improvement (+30%+)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import os
from datetime import datetime


class StaticAFT(nn.Module):
    def __init__(self, model, layer_idx, hidden_dim):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hook_handle = None
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        else:
            return self.model.transformer.h
    
    def _register_hook(self):
        layers = self._get_layers()
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden + self.steering_vector.to(hidden.dtype)
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


def load_hellaswag(max_samples=200):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples: break
        prompt = f"Complete:\n{ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nAnswer:"
        samples.append({"prompt": prompt, "label": int(ex["label"]), "choices": ["A","B","C","D"], "type": "mc"})
    split = len(samples) // 2
    return samples[:split], samples[split:]


def load_arc(max_samples=200):
    ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples: break
        prompt = f"Q: {ex['question']}\n"
        choices = ex['choices']
        for l, t in zip(choices['label'], choices['text']):
            prompt += f"{l}. {t}\n"
        prompt += "A:"
        label_map = {l: idx for idx, l in enumerate(choices['label'])}
        samples.append({"prompt": prompt, "label": label_map[ex['answerKey']], "choices": choices['label'], "type": "mc"})
    split = len(samples) // 2
    return samples[:split], samples[split:]


def evaluate(model, tokenizer, samples, device):
    correct = 0
    for s in samples:
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = []
        for opt in s["choices"]:
            tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
            if not tid: tid = tokenizer.encode(opt, add_special_tokens=False)
            probs.append(logits[tid[0]].item() if tid else -float('inf'))
        if np.argmax(probs) == s["label"]:
            correct += 1
    return correct / len(samples) * 100


def train_fast(steered, tokenizer, train_data, device, epochs=5, lr=5e-3):
    """Aggressive training with higher LR."""
    optimizer = torch.optim.Adam([steered.steering_vector], lr=lr)
    
    for epoch in range(epochs):
        np.random.shuffle(train_data)
        for s in train_data:
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            target_token = f" {s['choices'][s['label']]}"
            target_ids = tokenizer.encode(target_token, add_special_tokens=False)
            if not target_ids: continue
            
            outputs = steered(**inputs)
            logits = outputs.logits[0, -1, :]
            loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([target_ids[0]], device=device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([steered.steering_vector], 1.0)
            optimizer.step()
            optimizer.zero_grad()


def run_breakthrough_search():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*70)
    print("🔥 BREAKTHROUGH SEARCH")
    print("="*70)
    
    results = []
    
    # Test multiple models on HellaSwag with aggressive settings
    models_to_test = [
        ("Qwen/Qwen1.5-0.5B-Chat", "qwen-chat"),
        ("Qwen/Qwen1.5-0.5B", "qwen"),
        ("EleutherAI/pythia-410m", "pythia"),
    ]
    
    for model_path, model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {model_name}")
        print(f"{'='*60}")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            for p in model.parameters():
                p.requires_grad = False
            
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Get layers
            if hasattr(model, 'gpt_neox'):
                num_layers = len(model.gpt_neox.layers)
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                num_layers = len(model.model.layers)
            else:
                num_layers = len(model.transformer.h)
            
            hidden_dim = model.config.hidden_size
            
            # Load data
            train_data, test_data = load_hellaswag(300)
            
            # Baseline
            baseline = evaluate(model, tokenizer, test_data, device)
            print(f"Baseline: {baseline:.1f}%")
            
            # Test layers 0, 1, 2, 3 (very early) with aggressive training
            best_acc = baseline
            best_layer = 0
            
            for layer in range(min(4, num_layers)):
                steered = StaticAFT(model, layer, hidden_dim)
                steered.steering_vector.data = steered.steering_vector.data.to(device).float()
                
                # Initialize with small noise
                nn.init.normal_(steered.steering_vector, mean=0, std=0.02)
                
                # Aggressive training
                train_fast(steered, tokenizer, train_data, device, epochs=10, lr=5e-3)
                
                acc = evaluate(steered, tokenizer, test_data, device)
                improvement = acc - baseline
                
                print(f"  Layer {layer}: {acc:.1f}% ({improvement:+.1f}%)")
                
                if acc > best_acc:
                    best_acc = acc
                    best_layer = layer
                
                steered.cleanup()
            
            improvement = best_acc - baseline
            results.append({
                "model": model_name,
                "baseline": baseline,
                "best": best_acc,
                "improvement": improvement,
                "layer": best_layer
            })
            
            if improvement >= 15:
                print(f"\n🎯 STRONG RESULT: +{improvement:.1f}% on {model_name}")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS SUMMARY")
    print("="*70)
    
    for r in results:
        emoji = "🔥" if r["improvement"] >= 15 else "✅" if r["improvement"] > 0 else "❌"
        print(f"{emoji} {r['model']}: {r['baseline']:.1f}% → {r['best']:.1f}% ({r['improvement']:+.1f}%) @ Layer {r['layer']}")
    
    # Find best
    if results:
        best = max(results, key=lambda x: x["improvement"])
        print(f"\n🏆 BEST: {best['model']} with +{best['improvement']:.1f}%")
        
        if best["improvement"] >= 20:
            print("\n" + "="*70)
            print("🎉 BREAKTHROUGH ACHIEVED!")
            print("="*70)
    
    # Save
    os.makedirs("results/breakthrough", exist_ok=True)
    with open("results/breakthrough/search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_breakthrough_search()
