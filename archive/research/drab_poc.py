#!/usr/bin/env python3
"""
drab_poc.py - Dynamic Reasoning Activation Boosters (Dictionary Version)

A minimal, runnable PoC for DRAB v2.0 with interpretable dictionary steering.
See README7.md for full specification.

Usage:
    python drab_poc.py --model "EleutherAI/pythia-410m" --dataset gsm8k
    python drab_poc.py --model "Qwen/Qwen1.5-0.5B" --dataset arc_challenge --primitives 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


class DictionaryDRAB(nn.Module):
    """
    Dictionary-based Dynamic Reasoning Activation Booster.
    
    Uses a small router MLP to select from K learned "reasoning primitive" vectors.
    Much more stable and interpretable than raw MLP generation.
    
    Architecture:
        1. Router: Maps pooled context to K weights via softmax
        2. Dictionary: K learnable basis vectors (the "reasoning primitives")
        3. Gate: Smooth injection control that starts near-zero
    
    Args:
        hidden_dim: Model's hidden dimension
        num_primitives: Number of basis vectors (K), default 8
    """
    
    def __init__(self, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        
        # 1. Router: Maps context to primitive weights
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives),
            nn.Softmax(dim=-1)
        )
        
        # 2. Dictionary: K learnable basis vectors (the "reasoning primitives")
        self.basis_vectors = nn.Parameter(
            torch.randn(num_primitives, hidden_dim) * 0.01
        )
        
        # 3. Gate: Smooth injection control (starts near-zero)
        self.gate = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        
        # Initialize gate bias to negative (default: don't intervene)
        nn.init.constant_(self.gate.bias, -2.0)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Generate context-adaptive steering vector.
        
        Args:
            hidden_states: [B, S, D] tensor from target layer
            
        Returns:
            steering_vector: [B, 1, D] tensor to add to hidden states
        """
        # Pool context (mean over sequence)
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Route to primitives
        weights = self.router(pooled)  # [B, K]
        
        # Weighted sum of basis vectors
        steering = torch.matmul(weights, self.basis_vectors)  # [B, D]
        
        # Apply gated injection
        gate_value = torch.tanh(self.gate(pooled))  # [B, 1]
        gated_steering = self.alpha * gate_value * steering  # [B, D]
        
        return gated_steering.unsqueeze(1)  # [B, 1, D] for broadcasting
    
    def get_primitive_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """For visualization: get the routing weights."""
        pooled = hidden_states.mean(dim=1)
        return self.router(pooled)
    
    def parameter_count(self) -> int:
        """Return total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DRABSteeredModel(nn.Module):
    """Wraps a frozen model with DRAB injection at a target layer."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.drab = DictionaryDRAB(hidden_dim, num_primitives)
        self.hook_handle = None
        self._register_hook()
    
    def _get_layers(self):
        """Get layer list for different architectures."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Llama, Qwen
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # Phi, GPT-2
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers  # Pythia
        else:
            raise ValueError(f"Unknown architecture: {type(self.model)}")
    
    def _register_hook(self):
        layers = self._get_layers()
        
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Generate and apply steering
            steering = self.drab(hidden)
            steering = steering.to(hidden.dtype)
            modified = hidden + steering
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_frozen_logits(self, input_ids, attention_mask=None):
        """Get logits from frozen model (for KL regularization)."""
        self.hook_handle.remove()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        self._register_hook()
        return outputs.logits
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


def load_training_data(dataset_name: str, num_samples: int):
    """Load and format training data for different datasets."""
    print(f"   Loading {dataset_name} dataset...")
    
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="train")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            question = item["question"]
            answer = item["answer"].split("####")[-1].strip()
            samples.append({
                "prompt": f"Question: {question}\nAnswer:",
                "target": answer
            })
        return samples
    
    elif dataset_name == "hellaswag":
        ds = load_dataset("hellaswag", split="train")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            ctx = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            samples.append({
                "prompt": f"{ctx}",
                "target": endings[label][:20]  # First 20 chars of correct ending
            })
        return samples
    
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="train")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            question = item["question"]
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            answer_idx = labels.index(answer_key)
            samples.append({
                "prompt": f"Question: {question}\nAnswer:",
                "target": choices[answer_idx]
            })
        return samples
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def evaluate_model(model, tokenizer, dataset_name: str, num_samples: int, device: str):
    """Evaluate model accuracy on test split."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
    elif dataset_name == "hellaswag":
        ds = load_dataset("hellaswag", split="validation")
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    correct = 0
    total = 0
    
    for item in ds.select(range(min(num_samples, len(ds)))):
        if dataset_name == "gsm8k":
            prompt = f"Question: {item['question']}\nAnswer:"
            target = item["answer"].split("####")[-1].strip()
        elif dataset_name == "hellaswag":
            prompt = item["ctx"]
            endings = item["endings"]
            label = int(item["label"])
            target = endings[label][:20]
        else:  # arc_challenge
            prompt = f"Question: {item['question']}\nAnswer:"
            choices = item["choices"]["text"]
            labels = item["choices"]["label"]
            answer_key = item["answerKey"]
            answer_idx = labels.index(answer_key)
            target = choices[answer_idx]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            pred_id = outputs.logits[0, -1, :].argmax().item()
        
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if target_ids and pred_id == target_ids[0]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0


def visualize_primitives(steered, tokenizer, samples, device):
    """Show how primitives activate for different prompts."""
    primitive_names = [f"P{i}" for i in range(steered.drab.num_primitives)]
    
    for i, sample in enumerate(samples):
        inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Get hidden states at target layer
            layers = steered._get_layers()
            hidden_states = None
            
            def capture_hook(module, input, output):
                nonlocal hidden_states
                hidden_states = output[0] if isinstance(output, tuple) else output
            
            hook = layers[steered.layer_idx].register_forward_hook(capture_hook)
            steered.model(**inputs)
            hook.remove()
            
            weights = steered.drab.get_primitive_weights(hidden_states)
        
        # Display
        prompt_preview = sample["prompt"][:50].replace("\n", " ")
        print(f"\n   Sample {i+1}: \"{prompt_preview}...\"")
        
        weights_np = weights[0].cpu().numpy()
        for j, (name, w) in enumerate(zip(primitive_names, weights_np)):
            bar_len = int(w * 20)
            print(f"     {name}: {'█' * bar_len}{'░' * (20-bar_len)} {w:.2f}")


def train_drab(
    model_name: str = "EleutherAI/pythia-410m",
    dataset_name: str = "gsm8k",
    num_samples: int = 100,
    epochs: int = 5,
    lr: float = 1e-3,
    num_primitives: int = 8,
    lambda_kl: float = 0.1,
    device: str = "cuda"
):
    """
    Train DRAB adapter on a reasoning dataset.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: Dataset to train on (gsm8k, hellaswag, arc_challenge)
        num_samples: Number of training samples
        epochs: Training epochs
        lr: Learning rate
        num_primitives: Number of basis vectors (K)
        lambda_kl: KL regularization strength (0 to disable)
        device: cuda or cpu
        
    Returns:
        dict with baseline, steered accuracy, and improvement
    """
    print(f"\n{'='*60}")
    print(f"🚀 DRAB v2.0 Training")
    print(f"{'='*60}")
    print(f"   Model:      {model_name}")
    print(f"   Dataset:    {dataset_name}")
    print(f"   Primitives: {num_primitives}")
    print(f"   Samples:    {num_samples}")
    print(f"   Epochs:     {epochs}")
    print(f"   KL Lambda:  {lambda_kl}")
    print(f"{'='*60}\n")
    
    # Determine device
    if device == "cuda" and not torch.cuda.is_available():
        print("   ⚠️  CUDA not available, falling back to CPU")
        device = "cpu"
    
    # Load model (frozen)
    print(f"   Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine layer count and select middle layer
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
    elif hasattr(model, 'gpt_neox'):
        num_layers = len(model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown architecture: {type(model)}")
    
    layer_idx = num_layers // 2
    print(f"   Model has {num_layers} layers, targeting layer {layer_idx}")
    
    # Create DRAB wrapper
    steered = DRABSteeredModel(
        model, 
        layer_idx, 
        model.config.hidden_size,
        num_primitives
    )
    
    print(f"   DRAB parameters: {steered.drab.parameter_count():,}")
    
    # Move DRAB to device
    steered.drab = steered.drab.to(device)
    if device == "cuda":
        steered.drab = steered.drab.float()  # Train in fp32
    
    # Only train DRAB parameters
    optimizer = torch.optim.AdamW(steered.drab.parameters(), lr=lr)
    
    # Load dataset
    dataset = load_training_data(dataset_name, num_samples)
    print(f"   Loaded {len(dataset)} training samples\n")
    
    # Training loop
    print("📈 Training:")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for sample in dataset:
            prompt = sample["prompt"]
            target = sample["target"]
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            target_ids = tokenizer.encode(target, add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]
            
            # Forward pass
            outputs = steered(**inputs)
            logits = outputs.logits[0, -1, :]
            
            # Cross-entropy loss
            loss_ce = F.cross_entropy(
                logits.unsqueeze(0).float(),
                torch.tensor([target_id], device=device)
            )
            
            # KL regularization (optional but recommended)
            if lambda_kl > 0:
                frozen_logits = steered.get_frozen_logits(**inputs)[0, -1, :]
                loss_kl = F.kl_div(
                    F.log_softmax(logits.float(), dim=-1),
                    F.softmax(frozen_logits.float(), dim=-1),
                    reduction='batchmean'
                )
                loss = loss_ce + lambda_kl * loss_kl
            else:
                loss = loss_ce
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        gate_value = steered.drab.alpha.item()
        print(f"   Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Gate α = {gate_value:.4f}")
    
    # Evaluate
    print("\n📊 Evaluating...")
    baseline_acc = evaluate_model(model, tokenizer, dataset_name, min(50, num_samples), device)
    steered_acc = evaluate_model(steered, tokenizer, dataset_name, min(50, num_samples), device)
    
    improvement = steered_acc - baseline_acc
    emoji = "✅" if improvement > 0 else "❌" if improvement < 0 else "⏸️"
    
    print(f"\n{'='*60}")
    print(f"📋 RESULTS")
    print(f"{'='*60}")
    print(f"   Baseline Accuracy: {baseline_acc:.1%}")
    print(f"   DRAB Accuracy:     {steered_acc:.1%}")
    print(f"   {emoji} Improvement:     {improvement:+.1%}")
    print(f"{'='*60}")
    
    # Visualize primitive activations
    if len(dataset) >= 3:
        print("\n🔍 Primitive Activation Analysis (first 3 samples):")
        visualize_primitives(steered, tokenizer, dataset[:3], device)
    
    steered.cleanup()
    
    return {
        "model": model_name,
        "dataset": dataset_name,
        "baseline": baseline_acc,
        "steered": steered_acc,
        "improvement": improvement,
        "num_primitives": num_primitives,
        "epochs": epochs,
        "samples": num_samples
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRAB v2.0 - Dynamic Reasoning Activation Boosters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drab_poc.py --model "EleutherAI/pythia-410m" --dataset gsm8k
  python drab_poc.py --model "Qwen/Qwen1.5-0.5B" --dataset arc_challenge --primitives 16
  python drab_poc.py --model "microsoft/phi-1_5" --dataset hellaswag --epochs 10

See README7.md for full documentation.
        """
    )
    parser.add_argument("--model", default="EleutherAI/pythia-410m", 
                        help="HuggingFace model identifier")
    parser.add_argument("--dataset", default="gsm8k", 
                        choices=["gsm8k", "hellaswag", "arc_challenge"],
                        help="Dataset to train on")
    parser.add_argument("--samples", type=int, default=100, 
                        help="Number of training samples")
    parser.add_argument("--epochs", type=int, default=5, 
                        help="Training epochs")
    parser.add_argument("--primitives", type=int, default=8, 
                        help="Number of basis vectors (K)")
    parser.add_argument("--kl", type=float, default=0.1, 
                        help="KL regularization strength (0 to disable)")
    parser.add_argument("--lr", type=float, default=1e-3, 
                        help="Learning rate")
    parser.add_argument("--device", default="cuda", 
                        choices=["cuda", "cpu"],
                        help="Device to use")
    
    args = parser.parse_args()
    
    result = train_drab(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.samples,
        epochs=args.epochs,
        num_primitives=args.primitives,
        lambda_kl=args.kl,
        lr=args.lr,
        device=args.device
    )
    
    print(f"\n✅ Complete! Result: {result}")
