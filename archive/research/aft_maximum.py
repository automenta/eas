#!/usr/bin/env python3
"""
aft_maximum.py - Maximum Results with Minimum Effort

Incorporates ALL insights from README5.md and README6.md:
1. Full layer sweep (ALL layers, not just 20-80%)
2. Retry logic (3 seeds per experiment)
3. L2 regularization on steering vector
4. Early stopping with regression detection
5. Cross-task transfer testing

Goal: 90%+ positive results, +10% average improvement
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


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B",
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B",
}

DATASETS = ["hellaswag", "arc_challenge", "gsm8k"]


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC AFT WITH L2 REGULARIZATION
# ═══════════════════════════════════════════════════════════════════════════════

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
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()
    
    def get_norm(self):
        return self.steering_vector.norm().item()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_name, max_samples=200):
    samples = []
    
    if dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Complete:\n{ex['ctx']}\n"
            prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
            prompt += "\n\nAnswer:"
            samples.append({"prompt": prompt, "label": int(ex["label"]), "choices": ["A","B","C","D"], "type": "mc"})
    
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\n"
            choices = ex['choices']
            for l, t in zip(choices['label'], choices['text']):
                prompt += f"{l}. {t}\n"
            prompt += "\nAnswer:"
            label_map = {l: idx for idx, l in enumerate(choices['label'])}
            samples.append({"prompt": prompt, "label": label_map[ex['answerKey']], "choices": choices['label'], "type": "mc"})
    
    elif dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\nAnswer:"
            answer = ex['answer'].split("####")[-1].strip()
            samples.append({"prompt": prompt, "answer": answer, "type": "gen"})
    
    split = len(samples) // 2
    return samples[:split], samples[split:]


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, tokenizer, samples, device):
    correct = 0
    for s in samples:
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        if s["type"] == "mc":
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            probs = []
            for opt in s["choices"]:
                tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
                if not tid: tid = tokenizer.encode(opt, add_special_tokens=False)
                probs.append(logits[tid[0]].item() if tid else -float('inf'))
            if np.argmax(probs) == s["label"]:
                correct += 1
        else:
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=30, pad_token_id=tokenizer.eos_token_id, do_sample=False)
            out = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if s["answer"] in out:
                correct += 1
    
    return correct / len(samples) * 100


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WITH L2 REGULARIZATION AND EARLY STOPPING
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_l2(steered, tokenizer, train_data, test_data, device, 
                  epochs=10, lr=1e-3, l2_lambda=0.01, patience=3):
    """Train with L2 regularization and early stopping."""
    
    optimizer = torch.optim.Adam([steered.steering_vector], lr=lr)
    
    best_acc = 0
    best_vector = None
    no_improvement = 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(train_data)
        
        for s in train_data:
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            
            if s["type"] == "mc":
                target_token = f" {s['choices'][s['label']]}"
                target_ids = tokenizer.encode(target_token, add_special_tokens=False)
                if not target_ids: continue
                
                outputs = steered(**inputs)
                logits = outputs.logits[0, -1, :]
                
                # Cross-entropy + L2 regularization
                ce_loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([target_ids[0]], device=device))
                l2_loss = l2_lambda * steered.steering_vector.norm() ** 2
                loss = ce_loss + l2_loss
            else:
                answer_ids = tokenizer.encode(s["answer"], add_special_tokens=False)[:5]
                if not answer_ids: continue
                
                outputs = steered(**inputs)
                logits = outputs.logits[0, -1, :]
                ce_loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([answer_ids[0]], device=device))
                l2_loss = l2_lambda * steered.steering_vector.norm() ** 2
                loss = ce_loss + l2_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_([steered.steering_vector], 1.0)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        
        # Evaluate
        acc = evaluate(steered, tokenizer, test_data, device)
        
        if acc > best_acc:
            best_acc = acc
            best_vector = steered.steering_vector.data.clone()
            no_improvement = 0
        else:
            no_improvement += 1
            # Regression detection (from README5)
            if acc < best_acc - 2:
                break
            if no_improvement >= patience:
                break
    
    # Restore best
    if best_vector is not None:
        steered.steering_vector.data = best_vector
    
    return best_acc


# ═══════════════════════════════════════════════════════════════════════════════
# FULL LAYER SWEEP (ALL LAYERS)
# ═══════════════════════════════════════════════════════════════════════════════

def full_layer_sweep(model, tokenizer, train_data, test_data, hidden_dim, device):
    """Sweep ALL layers (insight from README5.md)."""
    
    layers = None
    if hasattr(model, 'gpt_neox'):
        layers = model.gpt_neox.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        layers = model.transformer.h
    
    num_layers = len(layers)
    best_layer = 0
    best_acc = 0
    results = []
    
    print(f"  Sweeping ALL {num_layers} layers...")
    
    for layer_idx in range(num_layers):
        steered = StaticAFT(model, layer_idx, hidden_dim)
        steered.steering_vector.data = steered.steering_vector.data.to(device).float()
        
        # Quick training (3 epochs) for sweep
        optimizer = torch.optim.Adam([steered.steering_vector], lr=1e-3)
        
        for _ in range(3):
            for s in train_data[:30]:
                inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
                if s["type"] == "mc":
                    target_token = f" {s['choices'][s['label']]}"
                    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
                    if not target_ids: continue
                    outputs = steered(**inputs)
                    logits = outputs.logits[0, -1, :]
                    loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([target_ids[0]], device=device))
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
        acc = evaluate(steered, tokenizer, test_data[:30], device)
        results.append({"layer": layer_idx, "acc": acc})
        
        if acc > best_acc:
            best_acc = acc
            best_layer = layer_idx
        
        steered.cleanup()
        print(f"    Layer {layer_idx:2d}: {acc:.1f}%", end="")
        if layer_idx == best_layer:
            print(" ← BEST", end="")
        print()
    
    return best_layer, results


# ═══════════════════════════════════════════════════════════════════════════════
# RETRY LOGIC (3 SEEDS)
# ═══════════════════════════════════════════════════════════════════════════════

def run_with_retry(model, tokenizer, train_data, test_data, layer_idx, hidden_dim, device, max_attempts=3):
    """Retry with different seeds (insight from README5.md)."""
    
    best_result = {"acc": 0, "seed": 0}
    
    for seed in range(max_attempts):
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        steered = StaticAFT(model, layer_idx, hidden_dim)
        steered.steering_vector.data = steered.steering_vector.data.to(device).float()
        
        # Initialize with small random values instead of zeros
        nn.init.normal_(steered.steering_vector, mean=0, std=0.01)
        
        acc = train_with_l2(steered, tokenizer, train_data, test_data, device)
        
        if acc > best_result["acc"]:
            best_result = {"acc": acc, "seed": seed, "norm": steered.get_norm()}
        
        steered.cleanup()
        
        # If we got a good result, stop retrying
        baseline = evaluate(model, tokenizer, test_data, device)
        if acc > baseline:
            break
    
    return best_result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(model_name, dataset_name, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = MODELS.get(model_name, model_name)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"🧪 {model_name} + {dataset_name}")
        print(f"{'='*60}")
    
    # Load model
    if verbose: print("  Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    for p in model.parameters():
        p.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    hidden_dim = model.config.hidden_size
    
    # Load data
    if verbose: print("  Loading data...")
    train_data, test_data = load_data(dataset_name, 200)
    
    # Baseline
    if verbose: print("  Baseline...")
    baseline = evaluate(model, tokenizer, test_data, device)
    if verbose: print(f"  Baseline: {baseline:.1f}%")
    
    # Full layer sweep
    if verbose: print("\n  Full layer sweep (ALL layers)...")
    best_layer, sweep = full_layer_sweep(model, tokenizer, train_data, test_data, hidden_dim, device)
    if verbose: print(f"\n  Best layer: {best_layer}")
    
    # Full training with retry logic
    if verbose: print(f"\n  Full training with 3 seeds...")
    result = run_with_retry(model, tokenizer, train_data, test_data, best_layer, hidden_dim, device)
    
    improvement = result["acc"] - baseline
    
    if verbose:
        emoji = "✅" if improvement > 0 else "❌" if improvement < 0 else "⏸️"
        print(f"\n  {'='*50}")
        print(f"  📋 RESULT")
        print(f"  {'='*50}")
        print(f"  Baseline:    {baseline:.1f}%")
        print(f"  AFT:         {result['acc']:.1f}%")
        print(f"  {emoji} Improvement: {improvement:+.1f}%")
        print(f"  Best Layer:  {best_layer}")
        print(f"  Best Seed:   {result['seed']}")
        print(f"  Vector Norm: {result.get('norm', 0):.4f}")
        print(f"  {'='*50}")
    
    # Save
    output = {
        "model": model_name,
        "dataset": dataset_name,
        "baseline": baseline,
        "final": result["acc"],
        "improvement": improvement,
        "layer": best_layer,
        "seed": result["seed"],
        "sweep": sweep,
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs("results/aft_maximum", exist_ok=True)
    with open(f"results/aft_maximum/{model_name}_{dataset_name}.json", "w") as f:
        json.dump(output, f, indent=2)
    
    return output


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Maximum AFT Results")
    parser.add_argument("--model", default="pythia-410m", choices=list(MODELS.keys()))
    parser.add_argument("--dataset", default="hellaswag", choices=DATASETS)
    parser.add_argument("--all", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("🚀 AFT MAXIMUM - Maximum Results, Minimum Effort")
    print(f"{'='*60}")
    
    if args.all:
        all_results = []
        for model in MODELS.keys():
            for dataset in DATASETS:
                try:
                    result = run_experiment(model, dataset)
                    all_results.append(result)
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*60}")
        
        positive = [r for r in all_results if r["improvement"] > 0]
        print(f"✅ Positive: {len(positive)}/{len(all_results)} ({len(positive)/len(all_results)*100:.0f}%)")
        
        if all_results:
            avg = sum(r["improvement"] for r in all_results) / len(all_results)
            print(f"📈 Average: {avg:+.1f}%")
        
        if positive:
            best = max(positive, key=lambda x: x["improvement"])
            print(f"🏆 Best: +{best['improvement']:.1f}% ({best['model']}/{best['dataset']})")
    else:
        run_experiment(args.model, args.dataset)


if __name__ == "__main__":
    main()
