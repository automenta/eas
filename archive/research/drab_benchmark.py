#!/usr/bin/env python3
"""
drab_benchmark.py - Universal DRAB Benchmark System

Applies all AFT success factors plus DRAB improvements:
- Multiple models (7+ small models 0.4B-1.8B)
- Multiple datasets (HellaSwag, ARC-Challenge, GSM8K)
- Proper MC evaluation (compare token probabilities, not just first token)
- Layer sweep (25-75% depth, step=2 for speed)
- Fast iterations (50 train, 50 test samples per quick run)
- Live animated progress
- Auto-save results to JSON
- Console and Dashboard modes

Usage:
    python drab_benchmark.py --console --quick                    # Quick smoke test
    python drab_benchmark.py --quick                              # Quick dashboard
    python drab_benchmark.py --model pythia-410m --dataset gsm8k  # Specific run
    python drab_benchmark.py --all --quick                        # Full sweep (all models x datasets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "qwen-0.5b": "Qwen/Qwen1.5-0.5B",
    "qwen-0.5b-chat": "Qwen/Qwen1.5-0.5B-Chat",
    "qwen-1.8b": "Qwen/Qwen1.5-1.8B",
    "phi-1.5": "microsoft/phi-1_5",
    "stablelm-1.6b": "stabilityai/stablelm-base-alpha-3b",
}

DATASETS = ["hellaswag", "arc_challenge", "gsm8k"]

# Quick mode settings (for fast iteration)
QUICK_SAMPLES = 50
QUICK_EPOCHS = 3
QUICK_LAYER_STEP = 4

# Full mode settings
FULL_SAMPLES = 200
FULL_EPOCHS = 5
FULL_LAYER_STEP = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DRAB CORE
# ═══════════════════════════════════════════════════════════════════════════════

class DictionaryDRAB(nn.Module):
    """Dictionary-based Dynamic Reasoning Activation Booster."""
    
    def __init__(self, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives
        
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_primitives),
            nn.Softmax(dim=-1)
        )
        self.basis_vectors = nn.Parameter(torch.randn(num_primitives, hidden_dim) * 0.01)
        self.gate = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.01))
        nn.init.constant_(self.gate.bias, -2.0)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        pooled = hidden_states.mean(dim=1).float()
        weights = self.router(pooled)
        steering = torch.matmul(weights, self.basis_vectors)
        gate_value = torch.tanh(self.gate(pooled))
        gated_steering = self.alpha * gate_value * steering
        return gated_steering.unsqueeze(1).to(original_dtype)
    
    def get_primitive_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1).float()
        return self.router(pooled)


class DRABModel(nn.Module):
    """Wraps a frozen model with DRAB injection."""
    
    def __init__(self, model, tokenizer, layer_idx: int, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.layer_idx = layer_idx
        self.drab = DictionaryDRAB(hidden_dim, num_primitives)
        self.hook_handle = None
        self._hidden_cache = None
        self._register_hook()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError(f"Unknown architecture: {type(self.model)}")
    
    def _register_hook(self):
        layers = self._get_layers()
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._hidden_cache = hidden.detach()
            steering = self.drab(hidden)
            modified = hidden + steering
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (AFT-style with proper MC format)
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_name: str, tokenizer, max_samples: int = 100):
    """Load data in AFT format with proper multiple-choice structure."""
    samples = []
    
    if dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
            prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
            prompt += "\n\nThe best answer is:"
            samples.append({
                "prompt": prompt,
                "label": int(ex["label"]),
                "choices": ["A", "B", "C", "D"],
                "type": "multiple_choice"
            })
    
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\nChoices:\n"
            choices = ex['choices']
            label_map = {label: idx for idx, label in enumerate(choices['label'])}
            
            for label, text in zip(choices['label'], choices['text']):
                prompt += f"{label}. {text}\n"
            prompt += "\nAnswer:"
            
            samples.append({
                "prompt": prompt,
                "label": label_map[ex['answerKey']],
                "choices": choices['label'],
                "type": "multiple_choice"
            })
    
    elif dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\nAnswer:"
            answer_number = ex['answer'].split("####")[-1].strip()
            samples.append({
                "prompt": prompt,
                "answer_text": ex['answer'],
                "answer_number": answer_number,
                "type": "generation"
            })
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # 50/50 split
    split_idx = len(samples) // 2
    return samples[:split_idx], samples[split_idx:]


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION (AFT-style: compare token probabilities for MC)
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, tokenizer, samples: List[Dict], device: str) -> float:
    """Evaluate model accuracy using proper token probability comparison."""
    correct = 0
    total = 0
    
    for s in samples:
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        if s["type"] == "multiple_choice":
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            # Get probability for each choice token
            probs = []
            for opt in s["choices"]:
                # Try " A" first (with space), fallback to "A"
                tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
                if not tid:
                    tid = tokenizer.encode(opt, add_special_tokens=False)
                
                if tid:
                    probs.append(logits[tid[0]].item())
                else:
                    probs.append(-float('inf'))
            
            if np.argmax(probs) == s["label"]:
                correct += 1
            total += 1
        
        elif s["type"] == "generation":
            # For GSM8K: generate and check if answer number appears
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False
                )
            generated = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            if s["answer_number"] in generated:
                correct += 1
            total += 1
    
    return correct / total * 100 if total > 0 else 0


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentState:
    """State for tracking experiment progress."""
    model_name: str = ""
    dataset: str = ""
    phase: str = "initializing"
    layer_idx: int = 0
    num_layers: int = 0
    epoch: int = 0
    max_epochs: int = 5
    sample: int = 0
    max_samples: int = 50
    current_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    baseline_acc: float = 0.0
    steered_acc: float = 0.0
    improvement: float = 0.0
    primitive_weights: List[float] = field(default_factory=lambda: [0.125] * 8)
    start_time: datetime = field(default_factory=datetime.now)
    results: List[Dict] = field(default_factory=list)
    sweep_results: List[Dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_experiment(
    model_name: str,
    dataset_name: str,
    state: ExperimentState,
    on_update: Callable[[ExperimentState], None],
    num_samples: int = 100,
    epochs: int = 5,
    num_primitives: int = 8,
    layer_step: int = 2,
    auto_layer: bool = True
) -> Dict:
    """Run a single DRAB experiment with proper AFT-style methodology."""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Update state
    state.model_name = model_name
    state.dataset = dataset_name
    state.max_samples = num_samples
    state.max_epochs = epochs
    state.phase = "loading"
    on_update(state)
    
    # Load model
    model_path = MODELS.get(model_name, model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get layer info
    def get_layers(m):
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers
        elif hasattr(m, 'transformer') and hasattr(m.transformer, 'h'):
            return m.transformer.h
        elif hasattr(m, 'gpt_neox'):
            return m.gpt_neox.layers
        raise ValueError(f"Unknown architecture: {type(m)}")
    
    layers = get_layers(model)
    num_layers = len(layers)
    state.num_layers = num_layers
    
    # Load data
    state.phase = "loading_data"
    on_update(state)
    train_data, test_data = load_data(dataset_name, tokenizer, num_samples * 2)
    
    # Baseline evaluation
    state.phase = "baseline"
    on_update(state)
    baseline_acc = evaluate(model, tokenizer, test_data, device)
    state.baseline_acc = baseline_acc
    on_update(state)
    
    # Layer sweep (if enabled)
    best_layer = num_layers // 2
    best_layer_loss = float('inf')
    
    if auto_layer:
        state.phase = "layer_sweep"
        state.sweep_results = []
        
        start_layer = int(num_layers * 0.25)
        end_layer = int(num_layers * 0.75)
        
        for layer_idx in range(start_layer, end_layer + 1, layer_step):
            state.layer_idx = layer_idx
            on_update(state)
            
            # Quick training on subset (1 epoch)
            drab = DRABModel(model, tokenizer, layer_idx, model.config.hidden_size, num_primitives)
            drab.drab = drab.drab.to(device)
            optimizer = torch.optim.AdamW(drab.drab.parameters(), lr=1e-3)
            
            subset = train_data[:min(30, len(train_data))]
            total_loss = 0
            
            for s in subset:
                inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
                
                if s["type"] == "multiple_choice":
                    target_token = f" {s['choices'][s['label']]}"
                    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
                    if not target_ids:
                        continue
                    target_id = target_ids[0]
                    
                    outputs = drab(**inputs)
                    logits = outputs.logits[0, -1, :]
                    loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([target_id], device=device))
                else:
                    # Generation: train on first few answer tokens
                    answer_tokens = tokenizer.encode(s["answer_number"], add_special_tokens=False)[:3]
                    if not answer_tokens:
                        continue
                    
                    outputs = drab(**inputs)
                    logits = outputs.logits[0, -1, :]
                    loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([answer_tokens[0]], device=device))
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(subset) if subset else float('inf')
            state.sweep_results.append({"layer": layer_idx, "loss": avg_loss})
            
            if avg_loss < best_layer_loss:
                best_layer_loss = avg_loss
                best_layer = layer_idx
            
            drab.cleanup()
        
        state.layer_idx = best_layer
        on_update(state)
    
    # Full training on best layer
    state.phase = "training"
    state.layer_idx = best_layer
    state.loss_history = []
    on_update(state)
    
    drab = DRABModel(model, tokenizer, best_layer, model.config.hidden_size, num_primitives)
    drab.drab = drab.drab.to(device)
    optimizer = torch.optim.AdamW(drab.drab.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        state.epoch = epoch + 1
        np.random.shuffle(train_data)
        
        for i, s in enumerate(train_data):
            state.sample = i + 1
            
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            
            if s["type"] == "multiple_choice":
                target_token = f" {s['choices'][s['label']]}"
                target_ids = tokenizer.encode(target_token, add_special_tokens=False)
                if not target_ids:
                    continue
                target_id = target_ids[0]
                
                outputs = drab(**inputs)
                logits = outputs.logits[0, -1, :]
                loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([target_id], device=device))
            else:
                answer_tokens = tokenizer.encode(s["answer_number"], add_special_tokens=False)[:3]
                if not answer_tokens:
                    continue
                
                outputs = drab(**inputs)
                logits = outputs.logits[0, -1, :]
                loss = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([answer_tokens[0]], device=device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(drab.drab.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            state.current_loss = loss.item()
            state.loss_history.append(loss.item())
            
            if drab._hidden_cache is not None:
                with torch.no_grad():
                    weights = drab.drab.get_primitive_weights(drab._hidden_cache)
                    state.primitive_weights = weights[0].cpu().tolist()
            
            on_update(state)
    
    # Final evaluation
    state.phase = "evaluating"
    on_update(state)
    
    steered_acc = evaluate(drab, tokenizer, test_data, device)
    state.steered_acc = steered_acc
    state.improvement = steered_acc - baseline_acc
    
    # Record result
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "layer": best_layer,
        "num_layers": num_layers,
        "baseline_acc": baseline_acc,
        "steered_acc": steered_acc,
        "improvement": state.improvement,
        "epochs": epochs,
        "samples": num_samples,
        "primitives": num_primitives,
        "timestamp": datetime.now().isoformat()
    }
    state.results.append(result)
    state.phase = "done"
    
    drab.cleanup()
    on_update(state)
    
    # Save results
    save_result(result)
    
    return result


def save_result(result: Dict):
    """Save result to JSON file."""
    os.makedirs("results/drab", exist_ok=True)
    model_slug = result["model"].replace("/", "_")
    filename = f"results/drab/{model_slug}_{result['dataset']}.json"
    
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE MODE
# ═══════════════════════════════════════════════════════════════════════════════

def console_update(state: ExperimentState):
    """Console output callback."""
    if state.phase == "loading":
        print(f"📦 Loading {state.model_name}...")
    elif state.phase == "loading_data":
        print(f"📚 Loading {state.dataset}...")
    elif state.phase == "baseline":
        print(f"📊 Baseline: {state.baseline_acc:.1f}%")
    elif state.phase == "layer_sweep":
        if state.sweep_results:
            last = state.sweep_results[-1]
            print(f"   Layer {last['layer']}: Loss {last['loss']:.4f}")
    elif state.phase == "training":
        if state.sample == 1 or state.sample % 10 == 0:
            print(f"   Epoch {state.epoch}/{state.max_epochs} | Sample {state.sample}/{state.max_samples} | Loss: {state.current_loss:.4f}")
    elif state.phase == "evaluating":
        print("📊 Evaluating...")
    elif state.phase == "done":
        emoji = "✅" if state.improvement > 0 else "❌" if state.improvement < 0 else "⏸️"
        print(f"\n{'='*60}")
        print(f"📋 RESULTS: {state.model_name} / {state.dataset}")
        print(f"{'='*60}")
        print(f"   Layer:       {state.layer_idx}/{state.num_layers}")
        print(f"   Baseline:    {state.baseline_acc:.1f}%")
        print(f"   DRAB:        {state.steered_acc:.1f}%")
        print(f"   {emoji} Improvement: {state.improvement:+.1f}%")
        print(f"{'='*60}\n")


def run_console_mode(args):
    """Run in console mode."""
    print(f"\n🚀 DRAB Benchmark (Console Mode)")
    print(f"{'='*60}")
    
    if args.all:
        models = list(MODELS.keys())[:3] if args.quick else list(MODELS.keys())
        datasets = DATASETS
    else:
        models = [args.model]
        datasets = [args.dataset]
    
    print(f"   Models:   {', '.join(models)}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Samples:  {args.samples}")
    print(f"   Epochs:   {args.epochs}")
    print(f"{'='*60}\n")
    
    state = ExperimentState()
    all_results = []
    
    for model in models:
        for dataset in datasets:
            print(f"\n{'='*60}")
            print(f"🧪 {model} + {dataset}")
            print(f"{'='*60}\n")
            
            try:
                result = run_single_experiment(
                    model_name=model,
                    dataset_name=dataset,
                    state=state,
                    on_update=console_update,
                    num_samples=args.samples,
                    epochs=args.epochs,
                    num_primitives=args.primitives,
                    layer_step=args.layer_step,
                    auto_layer=not args.no_sweep
                )
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("📊 SUMMARY")
        print(f"{'='*60}")
        
        positive = sum(1 for r in all_results if r["improvement"] > 0)
        negative = sum(1 for r in all_results if r["improvement"] < 0)
        neutral = sum(1 for r in all_results if r["improvement"] == 0)
        avg = sum(r["improvement"] for r in all_results) / len(all_results)
        
        print(f"   ✅ Positive: {positive}/{len(all_results)} ({positive/len(all_results)*100:.0f}%)")
        print(f"   ❌ Negative: {negative}/{len(all_results)}")
        print(f"   ⏸️ Neutral:  {neutral}/{len(all_results)}")
        print(f"   📈 Average:  {avg:+.1f}%")
        
        if positive > 0:
            best = max(all_results, key=lambda x: x["improvement"])
            print(f"   🌟 Best: +{best['improvement']:.1f}% ({best['model']}/{best['dataset']})")
        
        print(f"{'='*60}\n")
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_dashboard_mode(args):
    """Run with Rich TUI dashboard."""
    try:
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.layout import Layout
        from rich.text import Text
        from rich import box
    except ImportError:
        print("⚠️ Rich not installed. Falling back to console mode.")
        return run_console_mode(args)
    
    console = Console()
    
    def create_dashboard(state: ExperimentState) -> Layout:
        layout = Layout()
        
        # Header
        header = Text()
        header.append("🧠 ", style="bold")
        header.append("DRAB BENCHMARK", style="bold cyan")
        elapsed = datetime.now() - state.start_time
        header.append(f"  [{str(elapsed).split('.')[0]}]", style="dim")
        
        # Progress
        total_steps = state.max_epochs * state.max_samples
        current_step = (state.epoch - 1) * state.max_samples + state.sample if state.epoch > 0 else 0
        pct = current_step / total_steps * 100 if total_steps > 0 else 0
        bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        
        progress = Text()
        progress.append(f"\n  Model:   {state.model_name}\n", style="cyan")
        progress.append(f"  Dataset: {state.dataset}\n", style="green")
        progress.append(f"  Phase:   {state.phase}\n", style="yellow")
        progress.append(f"  Layer:   {state.layer_idx}/{state.num_layers}\n")
        progress.append(f"  Epoch:   {state.epoch}/{state.max_epochs}  Sample: {state.sample}/{state.max_samples}\n\n")
        progress.append(f"  [{bar}] {pct:.0f}%\n", style="cyan")
        
        if state.phase == "done":
            progress.append(f"\n  Baseline: {state.baseline_acc:.1f}%\n", style="dim")
            progress.append(f"  DRAB:     {state.steered_acc:.1f}%\n", style="bold cyan")
            emoji = "✅" if state.improvement > 0 else "❌" if state.improvement < 0 else "⏸️"
            style = "bold green" if state.improvement > 0 else "bold red" if state.improvement < 0 else "bold"
            progress.append(f"  {emoji} Improvement: {state.improvement:+.1f}%\n", style=style)
        
        # Results table
        table = Table(box=box.SIMPLE, show_header=True)
        table.add_column("Model", style="cyan")
        table.add_column("Dataset", style="green")
        table.add_column("Baseline", justify="right")
        table.add_column("DRAB", justify="right")
        table.add_column("Δ", justify="right")
        
        for r in state.results:
            delta = r["improvement"]
            style = "green" if delta > 0 else "red" if delta < 0 else "dim"
            table.add_row(
                r["model"],
                r["dataset"],
                f"{r['baseline_acc']:.1f}%",
                f"{r['steered_acc']:.1f}%",
                Text(f"{delta:+.1f}%", style=style)
            )
        
        # Primitives
        prim_text = Text()
        names = ["Math", "Logic", "Facts", "Safe", "Fmt", "Chain", "Abs", "Mem"]
        for i, (name, w) in enumerate(zip(names, state.primitive_weights)):
            bar_len = int(w * 15)
            prim_text.append(f"  {name:5} ", style="dim")
            prim_text.append("█" * bar_len + "░" * (15 - bar_len), style=["red", "green", "blue", "yellow", "magenta", "cyan", "white", "bright_black"][i])
            prim_text.append(f" {w:.2f}\n")
        
        # Layout
        layout.split_column(
            Layout(Panel(header, box=box.DOUBLE), size=3),
            Layout(name="main")
        )
        
        layout["main"].split_row(
            Layout(Panel(progress, title="Progress", border_style="cyan")),
            Layout(name="right", size=40)
        )
        
        layout["right"].split_column(
            Layout(Panel(prim_text, title="Primitives", border_style="magenta"), size=12),
            Layout(Panel(table, title="Results", border_style="blue"))
        )
        
        return layout
    
    # Determine models/datasets
    if args.all:
        models = list(MODELS.keys())[:3] if args.quick else list(MODELS.keys())
        datasets = DATASETS
    else:
        models = [args.model]
        datasets = [args.dataset]
    
    state = ExperimentState()
    live_display = None
    
    def dashboard_update(state: ExperimentState):
        if live_display:
            live_display.update(create_dashboard(state))
    
    console.print(f"\n🚀 DRAB Benchmark Dashboard")
    console.print(f"Models: {', '.join(models)}")
    console.print(f"Datasets: {', '.join(datasets)}\n")
    
    with Live(create_dashboard(state), console=console, refresh_per_second=4, screen=True) as live:
        live_display = live
        
        for model in models:
            for dataset in datasets:
                try:
                    run_single_experiment(
                        model_name=model,
                        dataset_name=dataset,
                        state=state,
                        on_update=dashboard_update,
                        num_samples=args.samples,
                        epochs=args.epochs,
                        num_primitives=args.primitives,
                        layer_step=args.layer_step,
                        auto_layer=not args.no_sweep
                    )
                    time.sleep(1)  # Pause to show result
                except Exception as e:
                    console.print(f"[red]Error: {e}[/]")
                    continue
    
    # Final summary
    if state.results:
        console.print(f"\n{'='*60}")
        console.print("[bold cyan]📊 FINAL SUMMARY[/]")
        console.print(f"{'='*60}")
        
        positive = sum(1 for r in state.results if r["improvement"] > 0)
        avg = sum(r["improvement"] for r in state.results) / len(state.results)
        
        console.print(f"   ✅ Positive: {positive}/{len(state.results)} ({positive/len(state.results)*100:.0f}%)")
        console.print(f"   📈 Average:  {avg:+.1f}%")
        
        if positive > 0:
            best = max(state.results, key=lambda x: x["improvement"])
            console.print(f"   🌟 Best: +{best['improvement']:.1f}% ({best['model']}/{best['dataset']})")
        
        console.print()
    
    return state.results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRAB Universal Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drab_benchmark.py --console --quick        # Quick console test
  python drab_benchmark.py --quick                  # Quick dashboard
  python drab_benchmark.py --all --quick            # All models x datasets (quick)
  python drab_benchmark.py --model qwen-0.5b        # Specific model
        """
    )
    parser.add_argument("--model", default="pythia-410m", choices=list(MODELS.keys()) + list(MODELS.values()))
    parser.add_argument("--dataset", default="hellaswag", choices=DATASETS)
    parser.add_argument("--all", action="store_true", help="Run all models x datasets")
    parser.add_argument("--samples", type=int, default=None, help="Samples per split")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--primitives", type=int, default=8, help="Number of primitives")
    parser.add_argument("--layer-step", type=int, default=None, help="Layer sweep step")
    parser.add_argument("--no-sweep", action="store_true", help="Disable layer sweep")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer samples)")
    parser.add_argument("--console", action="store_true", help="Console mode (no TUI)")
    
    args = parser.parse_args()
    
    # Apply quick/full defaults
    if args.quick:
        args.samples = args.samples or QUICK_SAMPLES
        args.epochs = args.epochs or QUICK_EPOCHS
        args.layer_step = args.layer_step or QUICK_LAYER_STEP
    else:
        args.samples = args.samples or FULL_SAMPLES
        args.epochs = args.epochs or FULL_EPOCHS
        args.layer_step = args.layer_step or FULL_LAYER_STEP
    
    if args.console:
        run_console_mode(args)
    else:
        run_dashboard_mode(args)


if __name__ == "__main__":
    main()
