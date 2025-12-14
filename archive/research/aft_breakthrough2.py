#!/usr/bin/env python3
"""
aft_breakthrough2.py - Deeper sweep on Qwen to find the magic layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


class StaticAFT(nn.Module):
    def __init__(self, model, layer_idx, hidden_dim):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        return self.model.transformer.h
    
    def _register_hook(self):
        layers = self._get_layers()
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            modified = hidden + self.steering_vector.to(hidden.dtype)
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        self.hook = layers[self.layer_idx].register_forward_hook(hook)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        self.hook.remove()


def load_hellaswag(max_samples=400):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples: break
        prompt = f"{ex['ctx']}\n\nA. {ex['endings'][0]}\nB. {ex['endings'][1]}\nC. {ex['endings'][2]}\nD. {ex['endings'][3]}\n\nAnswer:"
        samples.append({"prompt": prompt, "label": int(ex["label"]), "choices": ["A","B","C","D"]})
    return samples[:len(samples)//2], samples[len(samples)//2:]


def evaluate(model, tokenizer, samples, device):
    correct = 0
    for s in samples:
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        probs = [logits[tokenizer.encode(f" {c}", add_special_tokens=False)[0]].item() for c in s["choices"]]
        if np.argmax(probs) == s["label"]:
            correct += 1
    return correct / len(samples) * 100


def train(steered, tokenizer, data, device, epochs=15, lr=3e-3):
    opt = torch.optim.Adam([steered.steering_vector], lr=lr)
    for _ in range(epochs):
        np.random.shuffle(data)
        for s in data:
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            tid = tokenizer.encode(f" {s['choices'][s['label']]}", add_special_tokens=False)[0]
            logits = steered(**inputs).logits[0, -1, :]
            loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([tid], device=device))
            loss.backward()
            opt.step()
            opt.zero_grad()


def main():
    device = "cuda"
    
    print("\n" + "="*70)
    print("🔥 DEEP SWEEP ON QWEN-0.5B - ALL 24 LAYERS")
    print("="*70)
    
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_layers = len(model.model.layers)
    hidden_dim = model.config.hidden_size
    
    train_data, test_data = load_hellaswag(400)
    baseline = evaluate(model, tokenizer, test_data, device)
    print(f"Baseline: {baseline:.1f}%")
    
    best_acc = baseline
    best_layer = 0
    
    for layer in range(num_layers):
        steered = StaticAFT(model, layer, hidden_dim)
        steered.steering_vector.data = steered.steering_vector.data.to(device).float()
        nn.init.normal_(steered.steering_vector, std=0.02)
        
        train(steered, tokenizer, train_data, device, epochs=15, lr=3e-3)
        acc = evaluate(steered, tokenizer, test_data, device)
        improvement = acc - baseline
        
        marker = ""
        if improvement >= 15:
            marker = " 🔥🔥🔥"
        elif improvement >= 10:
            marker = " 🔥🔥"
        elif improvement >= 5:
            marker = " 🔥"
        elif acc > best_acc:
            marker = " ← NEW BEST"
        
        print(f"Layer {layer:2d}: {acc:.1f}% ({improvement:+.1f}%){marker}")
        
        if acc > best_acc:
            best_acc = acc
            best_layer = layer
        
        steered.cleanup()
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST: Layer {best_layer} with {best_acc:.1f}% ({best_acc - baseline:+.1f}%)")
    print(f"{'='*70}")
    
    if best_acc - baseline >= 20:
        print("\n🎉🎉🎉 BREAKTHROUGH ACHIEVED! 🎉🎉🎉")


if __name__ == "__main__":
    main()
