#!/usr/bin/env python3
"""
drab_vs_aft.py - Direct comparison: Static AFT vs Dynamic DRAB

This script runs both approaches side-by-side to identify what's causing
DRAB to underperform compared to AFT.

Hypothesis: DRAB's gating (α=0.01) is too weak. The dictionary/router adds
complexity that needs more training. Let's test:
1. Static AFT (baseline - what worked)
2. DRAB with stronger initialization
3. DRAB without gating
4. Simple dynamic (MLP generates vector directly)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import sys


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 1: STATIC AFT (The baseline that worked)
# ═══════════════════════════════════════════════════════════════════════════════

class StaticAFT(nn.Module):
    """Original AFT: Single learnable vector."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        # Just a single vector - this is what worked!
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hook_handle = None
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError(f"Unknown architecture")
    
    def _register_hook(self):
        layers = self._get_layers()
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Simple addition - no gating, no complexity
            vector = self.steering_vector.to(hidden.dtype).to(hidden.device)
            modified = hidden + vector
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 2: DRAB WITH STRONG INITIALIZATION (α=1.0, no gating)
# ═══════════════════════════════════════════════════════════════════════════════

class DRABStrong(nn.Module):
    """DRAB with stronger initialization - no gating, direct addition."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        
        # Router: context -> primitive weights
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives),
            nn.Softmax(dim=-1)
        )
        # Dictionary of primitives
        self.basis_vectors = nn.Parameter(torch.randn(num_primitives, hidden_dim) * 0.1)  # Stronger init
        
        self.hook_handle = None
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError(f"Unknown architecture")
    
    def _register_hook(self):
        layers = self._get_layers()
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Pool context
            pooled = hidden.mean(dim=1).float()  # [B, D]
            
            # Route to primitives
            weights = self.router(pooled)  # [B, K]
            
            # Weighted sum of basis vectors - NO GATING
            steering = torch.matmul(weights, self.basis_vectors)  # [B, D]
            steering = steering.unsqueeze(1).to(hidden.dtype)  # [B, 1, D]
            
            modified = hidden + steering
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# APPROACH 3: SIMPLE DYNAMIC (MLP generates vector directly)
# ═══════════════════════════════════════════════════════════════════════════════

class SimpleDynamic(nn.Module):
    """Simplest dynamic: Small MLP generates steering vector from context."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        
        # Tiny MLP: context -> steering vector
        # Much smaller than dictionary approach
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
        # Initialize output layer to near-zero
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        
        self.hook_handle = None
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError(f"Unknown architecture")
    
    def _register_hook(self):
        layers = self._get_layers()
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # Pool context and generate steering
            pooled = hidden.mean(dim=1).float()  # [B, D]
            steering = self.mlp(pooled)  # [B, D]
            steering = steering.unsqueeze(1).to(hidden.dtype)  # [B, 1, D]
            
            modified = hidden + steering
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_hellaswag(max_samples=100):
    """Load HellaSwag in MC format."""
    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({
            "prompt": prompt,
            "label": int(ex["label"]),
            "choices": ["A", "B", "C", "D"]
        })
    split = len(samples) // 2
    return samples[:split], samples[split:]


def evaluate(model, tokenizer, samples, device):
    """Evaluate accuracy."""
    correct = 0
    for s in samples:
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        
        probs = []
        for opt in s["choices"]:
            tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
            if not tid:
                tid = tokenizer.encode(opt, add_special_tokens=False)
            probs.append(logits[tid[0]].item() if tid else -float('inf'))
        
        if np.argmax(probs) == s["label"]:
            correct += 1
    
    return correct / len(samples) * 100


def train_and_eval(steered_model, tokenizer, train_data, test_data, device, epochs=5, lr=1e-3, name="Model"):
    """Train steering and evaluate."""
    
    # Get trainable params
    trainable = [p for p in steered_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"{'='*50}")
    
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(train_data)
        
        for i, s in enumerate(train_data):
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            
            target_token = f" {s['choices'][s['label']]}"
            target_ids = tokenizer.encode(target_token, add_special_tokens=False)
            if not target_ids:
                continue
            
            outputs = steered_model(**inputs)
            logits = outputs.logits[0, -1, :]
            loss = F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([target_ids[0]], device=device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_data)
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")
    
    # Evaluate
    acc = evaluate(steered_model, tokenizer, test_data, device)
    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "EleutherAI/pythia-410m"
    
    print(f"\n{'='*60}")
    print("🔬 DRAB vs AFT Direct Comparison")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    for p in model.parameters():
        p.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get layer info
    if hasattr(model, 'gpt_neox'):
        num_layers = len(model.gpt_neox.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    else:
        num_layers = len(model.transformer.h)
    
    layer_idx = num_layers // 2
    hidden_dim = model.config.hidden_size
    
    print(f"Layers: {num_layers}, Target: {layer_idx}, Hidden: {hidden_dim}")
    
    # Load data
    print("\n📚 Loading HellaSwag...")
    train_data, test_data = load_hellaswag(200)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Baseline
    print("\n📊 Baseline evaluation...")
    baseline_acc = evaluate(model, tokenizer, test_data, device)
    print(f"Baseline: {baseline_acc:.1f}%")
    
    results = {"baseline": baseline_acc}
    
    # Test each approach
    approaches = [
        ("Static AFT", lambda: StaticAFT(model, layer_idx, hidden_dim)),
        ("DRAB Strong", lambda: DRABStrong(model, layer_idx, hidden_dim, num_primitives=8)),
        ("Simple Dynamic", lambda: SimpleDynamic(model, layer_idx, hidden_dim)),
    ]
    
    for name, create_fn in approaches:
        steered = create_fn()
        
        # Move trainable params to device and float32
        for p in steered.parameters():
            if p.requires_grad:
                p.data = p.data.to(device).float()
        
        acc = train_and_eval(steered, tokenizer, train_data, test_data, device, epochs=5, lr=1e-3, name=name)
        improvement = acc - baseline_acc
        
        emoji = "✅" if improvement > 0 else "❌" if improvement < 0 else "⏸️"
        print(f"\n{emoji} {name}: {acc:.1f}% ({improvement:+.1f}%)")
        
        results[name] = {"accuracy": acc, "improvement": improvement}
        steered.cleanup()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 SUMMARY")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_acc:.1f}%")
    for name in ["Static AFT", "DRAB Strong", "Simple Dynamic"]:
        r = results[name]
        emoji = "✅" if r["improvement"] > 0 else "❌" if r["improvement"] < 0 else "⏸️"
        print(f"{emoji} {name}: {r['accuracy']:.1f}% ({r['improvement']:+.1f}%)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
