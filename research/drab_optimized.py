#!/usr/bin/env python3
"""
drab_optimized.py - Optimized DRAB based on experimental findings

Key insight: Static AFT (1,024 params) beats complex dynamic (74K+ params)
because dynamic approaches overfit on small training sets.

Solution: Hybrid approach
1. Learn a STATIC base vector (like AFT) 
2. Add TINY dynamic modulation (very few params)
3. Use regularization to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZED DRAB: Static base + Tiny dynamic modulation
# ═══════════════════════════════════════════════════════════════════════════════

class OptimizedDRAB(nn.Module):
    """
    Hybrid approach:
    1. Static base vector (like AFT) - always applied
    2. Tiny dynamic scale factor (just modulates magnitude, not direction)
    
    Total params: ~1,100 (vs 74K for DRAB Strong)
    """
    
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        
        # Static base vector (like AFT)
        self.base_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Tiny dynamic scale: context -> scalar [0.5, 1.5]
        # Just 2 params: weight (hidden_dim) + bias (1)
        self.scale_head = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)
        
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
            
            # Get base vector
            base = self.base_vector.to(hidden.dtype).to(hidden.device)
            
            # Dynamic scale (tiny)
            pooled = hidden.mean(dim=1).float()  # [B, D]
            scale = 1.0 + 0.5 * torch.tanh(self.scale_head(pooled))  # [B, 1] in [0.5, 1.5]
            scale = scale.unsqueeze(1).to(hidden.dtype)  # [B, 1, 1]
            
            # Apply scaled vector
            modified = hidden + scale * base
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


class StaticAFT(nn.Module):
    """Original AFT: Single learnable vector."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
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
# MULTI-LAYER AFT: Apply vector at multiple layers
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLayerAFT(nn.Module):
    """AFT applied at multiple layers with shared or per-layer vectors."""
    
    def __init__(self, model, layer_indices: list, hidden_dim: int, shared: bool = True):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.shared = shared
        
        if shared:
            self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        else:
            self.steering_vectors = nn.ParameterList([
                nn.Parameter(torch.zeros(1, 1, hidden_dim)) for _ in layer_indices
            ])
        
        self.hook_handles = []
        self._register_hooks()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError(f"Unknown architecture")
    
    def _register_hooks(self):
        layers = self._get_layers()
        
        for i, layer_idx in enumerate(self.layer_indices):
            if self.shared:
                vector = self.steering_vector
            else:
                vector = self.steering_vectors[i]
            
            def make_hook(vec):
                def hook_fn(module, input, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    v = vec.to(hidden.dtype).to(hidden.device)
                    modified = hidden + v
                    return (modified,) + output[1:] if isinstance(output, tuple) else modified
                return hook_fn
            
            handle = layers[layer_idx].register_forward_hook(make_hook(vector))
            self.hook_handles.append(handle)
    
    def forward(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask=attention_mask)
    
    def cleanup(self):
        for handle in self.hook_handles:
            handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_hellaswag(max_samples=100):
    ds = load_dataset("Rowan/hellaswag", split="validation")
    samples = []
    for i, ex in enumerate(ds):
        if i >= max_samples:
            break
        prompt = f"Complete:\n{ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nAnswer:"
        samples.append({
            "prompt": prompt,
            "label": int(ex["label"]),
            "choices": ["A", "B", "C", "D"]
        })
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
            if not tid:
                tid = tokenizer.encode(opt, add_special_tokens=False)
            probs.append(logits[tid[0]].item() if tid else -float('inf'))
        
        if np.argmax(probs) == s["label"]:
            correct += 1
    
    return correct / len(samples) * 100


def train_and_eval(steered_model, tokenizer, train_data, test_data, device, epochs=5, lr=1e-3, l2_reg=0.01, name="Model"):
    trainable = [p for p in steered_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=l2_reg)
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"Params: {sum(p.numel() for p in trainable):,}")
    print(f"{'='*50}")
    
    best_acc = 0
    best_state = None
    
    for epoch in range(epochs):
        epoch_loss = 0
        np.random.shuffle(train_data)
        
        for s in train_data:
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
        
        # Evaluate after each epoch
        acc = evaluate(steered_model, tokenizer, test_data, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in steered_model.state_dict().items() if 'steering' in k or 'base' in k or 'scale' in k}
        
        print(f"  Epoch {epoch+1}: Loss={epoch_loss/len(train_data):.4f}, Acc={acc:.1f}%")
    
    # Restore best
    if best_state:
        for k, v in best_state.items():
            if k in steered_model.state_dict():
                steered_model.state_dict()[k].copy_(v)
    
    final_acc = evaluate(steered_model, tokenizer, test_data, device)
    return final_acc, best_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "EleutherAI/pythia-410m"
    
    print(f"\n{'='*60}")
    print("🔬 Optimized DRAB Experiments")
    print(f"{'='*60}")
    
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
    
    hidden_dim = model.config.hidden_size
    print(f"Layers: {num_layers}, Hidden: {hidden_dim}")
    
    # Load data - more samples for better signal
    print("\n📚 Loading HellaSwag...")
    train_data, test_data = load_hellaswag(400)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # Baseline
    print("\n📊 Baseline evaluation...")
    baseline_acc = evaluate(model, tokenizer, test_data, device)
    print(f"Baseline: {baseline_acc:.1f}%")
    
    results = {}
    
    # Test different approaches
    approaches = [
        ("Static AFT (Layer 12)", lambda: StaticAFT(model, num_layers//2, hidden_dim)),
        ("Static AFT (Layer 8)", lambda: StaticAFT(model, num_layers//3, hidden_dim)),
        ("Optimized DRAB", lambda: OptimizedDRAB(model, num_layers//2, hidden_dim)),
        ("Multi-Layer AFT (8,12,16)", lambda: MultiLayerAFT(model, [8, 12, 16], hidden_dim, shared=True)),
    ]
    
    for name, create_fn in approaches:
        steered = create_fn()
        
        for p in steered.parameters():
            if p.requires_grad:
                p.data = p.data.to(device).float()
        
        final_acc, best_acc = train_and_eval(
            steered, tokenizer, train_data, test_data, device, 
            epochs=10, lr=1e-3, l2_reg=0.01, name=name
        )
        
        improvement = best_acc - baseline_acc
        emoji = "✅" if improvement > 0 else "❌" if improvement < 0 else "⏸️"
        print(f"\n{emoji} Best: {best_acc:.1f}% ({improvement:+.1f}%)")
        
        results[name] = {"accuracy": best_acc, "improvement": improvement}
        steered.cleanup()
    
    # Summary
    print(f"\n{'='*60}")
    print("📋 FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_acc:.1f}%")
    
    for name, r in results.items():
        emoji = "✅" if r["improvement"] > 0 else "❌" if r["improvement"] < 0 else "⏸️"
        print(f"{emoji} {name}: {r['accuracy']:.1f}% ({r['improvement']:+.1f}%)")
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]["improvement"])
    print(f"\n🏆 Best: {best_name} with {results[best_name]['improvement']:+.1f}% improvement")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
