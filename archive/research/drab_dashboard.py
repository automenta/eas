#!/usr/bin/env python3
"""
drab_dashboard.py - Dynamic Reasoning Activation Boosters Dashboard

An engaging TUI for running DRAB experiments with animated visualizations.
Supports both console mode (smoke test) and full dashboard mode.

Usage:
    python drab_dashboard.py --console           # Console smoke test
    python drab_dashboard.py                     # Full dashboard
    python drab_dashboard.py --quick             # Quick demo (fewer samples)
"""

import time
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# DRAB CORE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class DictionaryDRAB(nn.Module):
    """
    Dictionary-based Dynamic Reasoning Activation Booster.
    
    Uses a small router MLP to select from K learned "reasoning primitive" vectors.
    Much more stable and interpretable than raw MLP generation.
    """
    
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
        # Convert to float32 for MLP computation (model may be in float16)
        original_dtype = hidden_states.dtype
        pooled = hidden_states.mean(dim=1).float()
        weights = self.router(pooled)
        steering = torch.matmul(weights, self.basis_vectors)
        gate_value = torch.tanh(self.gate(pooled))
        gated_steering = self.alpha * gate_value * steering
        # Convert back to original dtype for injection
        return gated_steering.unsqueeze(1).to(original_dtype)
    
    def get_primitive_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pooled = hidden_states.mean(dim=1).float()
        return self.router(pooled)
    
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DRABSteeredModel(nn.Module):
    """Wraps a frozen model with DRAB injection at a target layer."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int, num_primitives: int = 8):
        super().__init__()
        self.model = model
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
            steering = self.drab(hidden).to(hidden.dtype)
            modified = hidden + steering
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def get_frozen_logits(self, input_ids, attention_mask=None):
        self.hook_handle.remove()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
        self._register_hook()
        return outputs.logits
    
    def cleanup(self):
        if self.hook_handle:
            self.hook_handle.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(dataset_name: str, num_samples: int, split: str = "train"):
    """Load training or test data."""
    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split=split if split == "train" else "test")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            answer = item["answer"].split("####")[-1].strip()
            samples.append({"prompt": f"Question: {item['question']}\nAnswer:", "target": answer})
        return samples
    
    elif dataset_name == "hellaswag":
        ds = load_dataset("hellaswag", split=split if split == "train" else "validation")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            label = int(item["label"])
            samples.append({"prompt": item["ctx"], "target": item["endings"][label][:20]})
        return samples
    
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split=split if split == "train" else "test")
        samples = []
        for item in ds.select(range(min(num_samples, len(ds)))):
            labels = item["choices"]["label"]
            idx = labels.index(item["answerKey"])
            samples.append({"prompt": f"Question: {item['question']}\nAnswer:", "target": item["choices"]["text"][idx]})
        return samples
    
    raise ValueError(f"Unknown dataset: {dataset_name}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentState:
    """Shared state for both console and dashboard modes."""
    model_name: str = ""
    dataset: str = ""
    phase: str = "initializing"  # initializing, loading, training, evaluating, done
    epoch: int = 0
    max_epochs: int = 5
    sample: int = 0
    max_samples: int = 50
    current_prompt: str = ""
    current_target: str = ""
    current_prediction: str = ""
    current_loss: float = 0.0
    is_correct: bool = False
    primitive_weights: List[float] = field(default_factory=lambda: [0.125] * 8)
    loss_history: List[float] = field(default_factory=list)
    baseline_acc: float = 0.0
    steered_acc: float = 0.0
    improvement: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    results: List[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER (Shared between console and dashboard)
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    model_name: str,
    dataset_name: str,
    state: ExperimentState,
    on_update: Callable[[ExperimentState], None],
    num_samples: int = 50,
    epochs: int = 5,
    num_primitives: int = 8,
    lambda_kl: float = 0.1
) -> dict:
    """
    Run a single DRAB experiment.
    
    Args:
        model_name: HuggingFace model identifier
        dataset_name: Dataset to train on
        state: Shared experiment state
        on_update: Callback for UI updates (console or dashboard)
        num_samples: Training samples
        epochs: Training epochs
        num_primitives: Number of basis vectors
        lambda_kl: KL regularization strength
        
    Returns:
        Result dictionary with baseline, steered accuracy, and improvement
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Update state
    state.model_name = model_name.split("/")[-1]
    state.dataset = dataset_name
    state.max_samples = num_samples
    state.max_epochs = epochs
    state.phase = "loading"
    on_update(state)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    if device == "cpu":
        model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine layer count
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
    layer_idx = num_layers // 2
    
    steered = DRABSteeredModel(model, layer_idx, model.config.hidden_size, num_primitives)
    steered.drab = steered.drab.to(device)
    if device == "cuda":
        steered.drab = steered.drab.float()
    
    optimizer = torch.optim.AdamW(steered.drab.parameters(), lr=1e-3)
    
    # Load data
    train_data = load_data(dataset_name, num_samples, split="train")
    
    # Training
    state.phase = "training"
    state.loss_history = []
    on_update(state)
    
    for epoch in range(epochs):
        state.epoch = epoch + 1
        
        for i, sample in enumerate(train_data):
            state.sample = i + 1
            state.current_prompt = sample["prompt"]
            state.current_target = sample["target"]
            
            inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            target_ids = tokenizer.encode(sample["target"], add_special_tokens=False)
            if not target_ids:
                continue
            target_id = target_ids[0]
            
            outputs = steered(**inputs)
            logits = outputs.logits[0, -1, :]
            
            loss_ce = F.cross_entropy(logits.unsqueeze(0).float(), torch.tensor([target_id], device=device))
            
            # KL regularization
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
            
            # Update state
            state.current_loss = loss.item()
            state.loss_history.append(loss.item())
            
            pred_id = logits.argmax().item()
            state.current_prediction = tokenizer.decode([pred_id])
            state.is_correct = (pred_id == target_id)
            
            # Get primitive weights
            if steered._hidden_cache is not None:
                with torch.no_grad():
                    weights = steered.drab.get_primitive_weights(steered._hidden_cache)
                    state.primitive_weights = weights[0].cpu().tolist()
            
            on_update(state)
    
    # Evaluation
    state.phase = "evaluating"
    on_update(state)
    
    test_data = load_data(dataset_name, min(30, num_samples), split="test")
    
    def evaluate(model_to_eval):
        correct = total = 0
        for sample in test_data:
            inputs = tokenizer(sample["prompt"], return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_to_eval(**inputs)
                pred_id = outputs.logits[0, -1, :].argmax().item()
            
            target_ids = tokenizer.encode(sample["target"], add_special_tokens=False)
            if target_ids and pred_id == target_ids[0]:
                correct += 1
            total += 1
        return correct / total * 100 if total > 0 else 0
    
    state.baseline_acc = evaluate(model)
    state.steered_acc = evaluate(steered)
    state.improvement = state.steered_acc - state.baseline_acc
    
    # Record result
    result = {
        "model": model_name,
        "dataset": dataset_name,
        "baseline": state.baseline_acc,
        "steered": state.steered_acc,
        "improvement": state.improvement,
        "epochs": epochs,
        "samples": num_samples,
        "primitives": num_primitives
    }
    state.results.append(result)
    state.phase = "done"
    
    steered.cleanup()
    on_update(state)
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLE MODE (Smoke Test)
# ═══════════════════════════════════════════════════════════════════════════════

def console_update(state: ExperimentState):
    """Simple console output for smoke testing."""
    if state.phase == "loading":
        print(f"📥 Loading {state.model_name}...")
    elif state.phase == "training":
        if state.sample == 1 or state.sample % 10 == 0:
            print(f"   Epoch {state.epoch}/{state.max_epochs} | Sample {state.sample}/{state.max_samples} | Loss: {state.current_loss:.4f}")
    elif state.phase == "evaluating":
        print("📊 Evaluating...")
    elif state.phase == "done":
        emoji = "✅" if state.improvement > 0 else "❌" if state.improvement < 0 else "⏸️"
        print(f"\n{'='*50}")
        print(f"📋 RESULTS")
        print(f"{'='*50}")
        print(f"   Baseline: {state.baseline_acc:.1f}%")
        print(f"   DRAB:     {state.steered_acc:.1f}%")
        print(f"   {emoji} Improvement: {state.improvement:+.1f}%")
        print(f"{'='*50}\n")


def run_console_mode(args):
    """Run experiment in console mode (smoke test)."""
    print(f"\n🚀 DRAB Experiment (Console Mode)")
    print(f"{'='*50}")
    print(f"   Model:      {args.model}")
    print(f"   Dataset:    {args.dataset}")
    print(f"   Samples:    {args.samples}")
    print(f"   Epochs:     {args.epochs}")
    print(f"   Primitives: {args.primitives}")
    print(f"{'='*50}\n")
    
    state = ExperimentState()
    
    result = run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        state=state,
        on_update=console_update,
        num_samples=args.samples,
        epochs=args.epochs,
        num_primitives=args.primitives,
        lambda_kl=args.kl
    )
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD MODE (Rich TUI)
# ═══════════════════════════════════════════════════════════════════════════════

def run_dashboard_mode(args):
    """Run experiment with animated Rich dashboard."""
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.layout import Layout
    from rich.text import Text
    from rich.align import Align
    from rich import box
    
    console = Console()
    
    def create_header(state: ExperimentState) -> Panel:
        elapsed = datetime.now() - state.start_time
        elapsed_str = str(elapsed).split('.')[0]
        
        title = Text()
        title.append("🧠 ", style="bold")
        title.append("DRAB", style="bold cyan")
        title.append(" REASONING BOOSTER DASHBOARD", style="bold white")
        
        right = Text(f"[{elapsed_str}]", style="dim cyan")
        
        return Panel(
            Align.center(title),
            subtitle=Align.right(right),
            style="bold blue",
            box=box.DOUBLE
        )
    
    def create_experiment_panel(state: ExperimentState) -> Panel:
        total_steps = state.max_epochs * state.max_samples
        current_step = (state.epoch - 1) * state.max_samples + state.sample if state.epoch > 0 else 0
        progress_pct = (current_step / total_steps * 100) if total_steps > 0 else 0
        
        bar_width = 40
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        content = Text()
        content.append(f"  Model:   ", style="dim")
        content.append(f"{state.model_name}\n", style="bold cyan")
        content.append(f"  Dataset: ", style="dim")
        content.append(f"{state.dataset}", style="bold green")
        content.append(f"          Phase: ", style="dim")
        content.append(f"{state.phase}\n", style="yellow")
        content.append(f"  Epoch:   ", style="dim")
        content.append(f"{state.epoch}/{state.max_epochs}", style="bold")
        content.append(f"          Sample: ", style="dim")
        content.append(f"{state.sample}/{state.max_samples}\n\n", style="bold")
        content.append(f"  {bar} ", style="cyan")
        content.append(f"{progress_pct:.0f}%", style="bold cyan")
        
        return Panel(content, title="[bold white]📊 CURRENT EXPERIMENT[/]", border_style="cyan", box=box.ROUNDED)
    
    def create_sample_panel(state: ExperimentState) -> Panel:
        prompt_preview = state.current_prompt[:60] + "..." if len(state.current_prompt) > 60 else state.current_prompt
        prompt_preview = prompt_preview.replace("\n", " ")
        
        content = Text()
        content.append("  📝 PROMPT:\n", style="bold yellow")
        content.append(f"  \"{prompt_preview}\"\n\n", style="italic")
        content.append("  🎯 TARGET: ", style="bold green")
        content.append(f"\"{state.current_target}\"\n", style="green")
        content.append("  🤖 MODEL:  ", style="bold cyan")
        content.append(f"\"{state.current_prediction}\" ", style="cyan")
        
        if state.current_prediction:
            if state.is_correct:
                content.append("✅", style="bold green")
            else:
                content.append("❌", style="bold red")
        
        content.append(f"           Loss: ", style="dim")
        content.append(f"{state.current_loss:.4f}", style="bold magenta")
        
        return Panel(content, title="[bold white]🔬 LIVE SAMPLE[/]", border_style="yellow", box=box.ROUNDED)
    
    def create_primitives_panel(state: ExperimentState) -> Panel:
        weights = state.primitive_weights
        primitive_names = ["Math", "Logic", "Facts", "Safety", "Format", "Chain", "Abstract", "Memory"]
        colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "white", "bright_black"]
        
        content = Text()
        for name, w, color in zip(primitive_names, weights, colors):
            bar_len = int(w * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            content.append(f"  {name:8} ", style="dim")
            content.append(bar, style=color)
            content.append(f" {w:.2f}\n", style="bold")
        
        return Panel(content, title="[bold white]🧬 PRIMITIVE ACTIVATIONS[/]", border_style="magenta", box=box.ROUNDED)
    
    def create_loss_panel(state: ExperimentState) -> Panel:
        losses = state.loss_history[-30:] if state.loss_history else []
        
        if not losses:
            content = Text("  Waiting for training data...", style="dim italic")
        else:
            max_loss = max(losses) if losses else 1
            min_loss = min(losses) if losses else 0
            range_loss = max_loss - min_loss if max_loss != min_loss else 1
            
            chart_height = 5
            chart_width = min(len(losses), 30)
            
            content = Text()
            for row in range(chart_height):
                threshold = max_loss - (row / chart_height) * range_loss
                line = "  "
                for loss in losses[-chart_width:]:
                    if loss >= threshold:
                        line += "█"
                    else:
                        line += " "
                content.append(line + "\n", style="cyan")
            
            content.append(f"\n  Current: {losses[-1]:.4f}  Min: {min_loss:.4f}  Max: {max_loss:.4f}", style="dim")
        
        return Panel(content, title="[bold white]📉 TRAINING DYNAMICS[/]", border_style="green", box=box.ROUNDED)
    
    def create_results_panel(state: ExperimentState) -> Panel:
        if state.phase == "done":
            content = Text()
            emoji = "✅" if state.improvement > 0 else "❌" if state.improvement < 0 else "⏸️"
            
            content.append(f"\n  {emoji} ", style="bold")
            content.append(f"EXPERIMENT COMPLETE\n\n", style="bold white")
            content.append(f"  Baseline Accuracy:  ", style="dim")
            content.append(f"{state.baseline_acc:.1f}%\n", style="bold")
            content.append(f"  DRAB Accuracy:      ", style="dim")
            content.append(f"{state.steered_acc:.1f}%\n", style="bold cyan")
            content.append(f"  Improvement:        ", style="dim")
            
            style = "bold green" if state.improvement > 0 else "bold red" if state.improvement < 0 else "bold"
            content.append(f"{state.improvement:+.1f}%\n", style=style)
        else:
            content = Text()
            content.append(f"\n  🔄 ", style="bold")
            content.append(f"Running experiment...\n\n", style="dim italic")
            content.append(f"  Phase: {state.phase}\n", style="dim")
        
        return Panel(content, title="[bold white]📋 RESULTS[/]", border_style="blue", box=box.ROUNDED)
    
    def create_stats_panel(state: ExperimentState) -> Panel:
        results = state.results
        
        if results:
            positive = sum(1 for r in results if r["improvement"] > 0)
            neutral = sum(1 for r in results if r["improvement"] == 0)
            negative = sum(1 for r in results if r["improvement"] < 0)
            total = len(results)
            avg_improvement = sum(r["improvement"] for r in results) / total if total > 0 else 0
            best = max(results, key=lambda x: x["improvement"]) if results else None
        else:
            positive = neutral = negative = total = 0
            avg_improvement = 0
            best = None
        
        content = Text()
        content.append(f"  Experiments: ", style="dim")
        content.append(f"{total}\n\n", style="bold")
        
        content.append(f"  ✅ Positive: ", style="green")
        content.append(f"{positive} ", style="bold green")
        content.append(f"  ⏸️ Neutral: ", style="dim")
        content.append(f"{neutral} ", style="bold")
        content.append(f"  ❌ Negative: ", style="red")
        content.append(f"{negative}\n\n", style="bold red")
        
        content.append(f"  📈 Avg Improvement: ", style="dim")
        style = "bold green" if avg_improvement > 0 else "bold red" if avg_improvement < 0 else "bold"
        content.append(f"{avg_improvement:+.1f}%\n", style=style)
        
        if best:
            content.append(f"\n  🌟 Best: ", style="dim")
            content.append(f"+{best['improvement']:.1f}%", style="bold green")
        
        return Panel(content, title="[bold white]📊 STATISTICS[/]", border_style="yellow", box=box.ROUNDED)
    
    def render_dashboard(state: ExperimentState) -> Layout:
        layout = Layout()
        
        layout.split_column(
            Layout(create_header(state), name="header", size=3),
            Layout(name="body"),
            Layout(create_stats_panel(state), name="footer", size=10)
        )
        
        layout["body"].split_row(
            Layout(name="left"),
            Layout(name="right", size=35)
        )
        
        layout["left"].split_column(
            Layout(create_experiment_panel(state), size=9),
            Layout(create_sample_panel(state), size=8),
            Layout(create_loss_panel(state), size=10),
            Layout(create_results_panel(state))
        )
        
        layout["right"].split_column(
            Layout(create_primitives_panel(state))
        )
        
        return layout
    
    # Setup state and live display
    state = ExperimentState()
    
    print(f"\n🚀 DRAB Reasoning Booster Dashboard\n")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.samples}, Epochs: {args.epochs}\n")
    
    live_display = None
    
    def dashboard_update(state: ExperimentState):
        nonlocal live_display
        if live_display:
            live_display.update(render_dashboard(state))
    
    with Live(render_dashboard(state), console=console, refresh_per_second=4, screen=True) as live:
        live_display = live
        try:
            result = run_experiment(
                model_name=args.model,
                dataset_name=args.dataset,
                state=state,
                on_update=dashboard_update,
                num_samples=args.samples,
                epochs=args.epochs,
                num_primitives=args.primitives,
                lambda_kl=args.kl
            )
            
            # Show final result for a moment
            time.sleep(3)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/]")
            return None
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/]")
            raise
    
    # Print final summary
    console.print("\n" + "=" * 50)
    console.print("[bold cyan]📋 FINAL SUMMARY[/]")
    console.print("=" * 50)
    
    emoji = "✅" if state.improvement > 0 else "❌" if state.improvement < 0 else "⏸️"
    console.print(f"\n{emoji} DRAB Improvement: {state.improvement:+.1f}%")
    console.print(f"   Baseline: {state.baseline_acc:.1f}% → DRAB: {state.steered_acc:.1f}%")
    console.print("\n")
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRAB - Dynamic Reasoning Activation Boosters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drab_dashboard.py --console                    # Console smoke test
  python drab_dashboard.py                              # Full dashboard
  python drab_dashboard.py --quick                      # Quick demo
  python drab_dashboard.py --model "Qwen/Qwen1.5-0.5B"  # Different model
        """
    )
    parser.add_argument("--model", default="EleutherAI/pythia-410m", help="Model to test")
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "hellaswag", "arc_challenge"])
    parser.add_argument("--samples", type=int, default=50, help="Training samples")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--primitives", type=int, default=8, help="Number of basis vectors")
    parser.add_argument("--kl", type=float, default=0.1, help="KL regularization strength")
    parser.add_argument("--quick", action="store_true", help="Quick demo (fewer samples)")
    parser.add_argument("--console", action="store_true", help="Console mode (no Rich dashboard)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.samples = 20
        args.epochs = 3
    
    if args.console:
        run_console_mode(args)
    else:
        # Check if Rich is available
        try:
            from rich.console import Console
            run_dashboard_mode(args)
        except ImportError:
            print("⚠️  Rich library not installed. Falling back to console mode.")
            print("   Install with: pip install rich\n")
            run_console_mode(args)


if __name__ == "__main__":
    main()
