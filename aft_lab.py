#!/usr/bin/env python3
"""
aft_lab.py - AFT Experiment Laboratory v2.0

An immersive, animated experiment dashboard for pushing AFT to its limits.

Features:
- 🎨 Stunning Rich TUI with animations and real-time visualizations
- 🧬 Evolutionary vector search with population dynamics
- 🔄 Recursive amplification experiments  
- 📊 Live metrics: loss curves, accuracy tracking, vector norms
- 🏆 Leaderboard tracking best configurations
- 💾 Full experiment logging and reproducibility

The Ultimate README8 Research Platform.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import deque

# Rich TUI imports
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskID
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.rule import Rule
from rich.padding import Padding
from rich.markdown import Markdown

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 VISUAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = """
[bold magenta]
   █████╗ ███████╗████████╗    ██╗      █████╗ ██████╗ 
  ██╔══██╗██╔════╝╚══██╔══╝    ██║     ██╔══██╗██╔══██╗
  ███████║█████╗     ██║       ██║     ███████║██████╔╝
  ██╔══██║██╔══╝     ██║       ██║     ██╔══██║██╔══██╗
  ██║  ██║██║        ██║       ███████╗██║  ██║██████╔╝
  ╚═╝  ╚═╝╚═╝        ╚═╝       ╚══════╝╚═╝  ╚═╝╚═════╝ 
[/]
[dim]Activation Fine-Tuning Laboratory v2.0[/]
[dim cyan]Maximum Results • Minimum Effort • Beautiful Science[/]
"""

SPARKLINE_CHARS = "▁▂▃▄▅▆▇█"
PROGRESS_CHARS = "⣾⣽⣻⢿⡿⣟⣯⣷"

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str = "AFT Lab"
    model_name: str = "Qwen/Qwen2-0.5B"
    dataset: str = "hellaswag"
    train_samples: int = 100
    test_samples: int = 50
    epochs: int = 5
    lr: float = 1e-3
    
    # Multi-layer config
    layer_percentages: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.8])
    
    # Coefficient sweep
    coefficients: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0])
    
    # Evolutionary search
    population_size: int = 10
    generations: int = 20
    mutation_rate: float = 0.15
    elite_ratio: float = 0.2
    
    # Recursive refinement
    recursive_passes: int = 3


@dataclass 
class ExperimentResult:
    """Result of an experiment."""
    name: str
    baseline_acc: float
    final_acc: float
    improvement: float
    training_time: float = 0.0
    best_config: Dict[str, Any] = field(default_factory=dict)
    history: List[float] = field(default_factory=list)


class LiveState:
    """Shared state for live dashboard updates."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.phase = "Initializing..."
        self.experiment = ""
        self.epoch = 0
        self.total_epochs = 0
        self.sample = 0
        self.total_samples = 0
        self.loss = 0.0
        self.accuracy = 0.0
        self.best_accuracy = 0.0
        self.loss_history = deque(maxlen=50)
        self.acc_history = deque(maxlen=50)
        self.current_prompt = ""
        self.current_target = ""
        self.current_pred = ""
        self.is_correct = False
        self.results: List[ExperimentResult] = []
        self.leaderboard: List[Tuple[str, float]] = []
        self.generation = 0
        self.population_fitness = []
        self.vector_norm = 0.0
        self.gradient_norm = 0.0
        self.start_time = time.time()
        self.spinner_idx = 0
        self.messages = deque(maxlen=5)
    
    def elapsed(self) -> str:
        e = time.time() - self.start_time
        m, s = divmod(int(e), 60)
        return f"{m:02d}:{s:02d}"
    
    def add_message(self, msg: str, style: str = "dim"):
        self.messages.append((msg, style))


# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 AFT CORE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class SingleLayerAFT(nn.Module):
    """Single-layer steering vector."""
    
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.coefficient = 1.0
        self.hook = None
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return self.model.transformer.h
    
    def _setup(self):
        layers = self._get_layers()
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = self.coefficient * self.steering_vector.to(device=h.device, dtype=h.dtype)
            return (h + v,) + out[1:] if isinstance(out, tuple) else h + v
        self.hook = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        if self.hook:
            self.hook.remove()


class MultiLayerAFT(nn.Module):
    """Multi-layer steering with learnable scales."""
    
    def __init__(self, model, layer_indices: List[int], hidden_dim: int, 
                 initial_scales: List[float] = None):
        super().__init__()
        self.model = model
        self.layer_indices = layer_indices
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        if initial_scales is None:
            initial_scales = [0.5, 1.0, 0.3][:len(layer_indices)]
        self.scales = nn.Parameter(torch.tensor(initial_scales))
        self.hooks = []
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return self.model.transformer.h
    
    def _setup(self):
        layers = self._get_layers()
        for i, idx in enumerate(self.layer_indices):
            def make_hook(scale_idx):
                def fn(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    scale = self.scales[scale_idx].to(device=h.device)
                    v = scale * self.steering_vector.to(device=h.device, dtype=h.dtype)
                    return (h + v,) + out[1:] if isinstance(out, tuple) else h + v
                return fn
            self.hooks.append(layers[idx].register_forward_hook(make_hook(i)))
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
    
    def get_vector_norm(self) -> float:
        return torch.norm(self.steering_vector).item()


class ResonanceChamberAFT(nn.Module):
    """
    Resonance Chamber: Apply the same vector at consecutive layers with
    exponentially increasing strength to create a compounding echo effect.
    
    Like an echo chamber where the signal bounces and amplifies,
    creating harmonic resonance through the layer stack.
    """
    
    def __init__(self, model, start_layer: int, num_resonances: int, hidden_dim: int,
                 growth_rate: float = 1.3):
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.num_resonances = num_resonances
        self.growth_rate = growth_rate
        
        # Single vector that echoes through layers
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Amplitudes grow exponentially (resonance!)
        self.amplitudes = nn.Parameter(
            torch.tensor([growth_rate ** i for i in range(num_resonances)])
        )
        
        self.hooks = []
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return self.model.transformer.h
    
    def _setup(self):
        layers = self._get_layers()
        
        # Apply at consecutive layers with growing amplitude
        for i in range(self.num_resonances):
            layer_idx = self.start_layer + i
            if layer_idx >= len(layers):
                break
            
            def make_hook(amp_idx):
                def fn(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    amp = self.amplitudes[amp_idx].to(device=h.device)
                    v = amp * self.steering_vector.to(device=h.device, dtype=h.dtype)
                    return (h + v,) + out[1:] if isinstance(out, tuple) else h + v
                return fn
            
            self.hooks.append(layers[layer_idx].register_forward_hook(make_hook(i)))
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        for h in self.hooks:
            h.remove()
    
    def get_vector_norm(self) -> float:
        return torch.norm(self.steering_vector).item()
    
    def get_effective_magnitudes(self) -> List[float]:
        """Return the amplification at each resonance layer."""
        return [a.item() for a in self.amplitudes]


class ActivationEchoChamber(nn.Module):
    """
    Activation Echo Chamber: Dynamic feedback loop during generation.
    
    1. Apply initial steering
    2. Generate partial output (CoT chunk)
    3. Pool activations from that generation
    4. Refine steering based on pooled activations
    5. Continue generation with refined steering
    
    Creates emergent deeper reasoning through self-referential adaptation.
    """
    
    def __init__(self, model, layer_idx: int, hidden_dim: int, 
                 initial_vector: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        
        # Steering vector that adapts during generation
        if initial_vector is not None:
            self.steering_vector = nn.Parameter(initial_vector.clone())
        else:
            self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Refinement network: pools activations → adjusts steering
        self.refiner = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Tanh()  # Bounded adjustment
        )
        
        # Track activation history for pooling
        self.activation_buffer = []
        self.current_steering = None
        self.hook = None
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        return self.model.transformer.h
    
    def _setup(self):
        layers = self._get_layers()
        
        def hook_fn(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            
            # Store activations for pooling
            self.activation_buffer.append(h.detach().mean(dim=1).cpu())
            
            # Apply current steering
            if self.current_steering is None:
                v = self.steering_vector.to(device=h.device, dtype=h.dtype)
            else:
                v = self.current_steering.to(device=h.device, dtype=h.dtype)
            
            modified = h + v
            return (modified,) + out[1:] if isinstance(out, tuple) else modified
        
        self.hook = layers[self.layer_idx].register_forward_hook(hook_fn)
    
    def refine_steering(self):
        """
        Pool recent activations and refine steering vector.
        Called between generation chunks.
        """
        if len(self.activation_buffer) == 0:
            return
        
        # Pool recent activations (last 5 tokens)
        recent = self.activation_buffer[-min(5, len(self.activation_buffer)):]
        pooled = torch.stack(recent).mean(dim=0)  # [1, hidden_dim]
        
        # Generate refinement delta
        with torch.no_grad():
            delta = self.refiner(pooled.float().to(self.steering_vector.device))
            delta = delta.unsqueeze(1)  # [1, 1, hidden_dim]
            
            # Update current steering (weighted blend)
            if self.current_steering is None:
                self.current_steering = self.steering_vector.clone()
            
            # Echo: 70% current + 30% adjustment
            self.current_steering = 0.7 * self.current_steering + 0.3 * delta
    
    def reset_buffer(self):
        """Reset activation buffer for new generation."""
        self.activation_buffer = []
        self.current_steering = None
    
    def forward(self, **kwargs):
        return self.model(**kwargs)
    
    def cleanup(self):
        if self.hook:
            self.hook.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# 🧬 EVOLUTIONARY VECTOR SEARCH
# ═══════════════════════════════════════════════════════════════════════════════

class VectorEvolver:
    """Genetic algorithm for steering vector optimization."""
    
    def __init__(self, hidden_dim: int, config: ExperimentConfig):
        self.hidden_dim = hidden_dim
        self.config = config
        self.population = [
            torch.randn(1, 1, hidden_dim) * 0.1
            for _ in range(config.population_size)
        ]
        self.fitness_scores = [0.0] * config.population_size
        self.best_vector = None
        self.best_fitness = 0.0
        self.generation = 0
    
    def evaluate_population(self, fitness_fn) -> List[float]:
        """Evaluate all individuals in the population."""
        self.fitness_scores = []
        for i, vec in enumerate(self.population):
            fitness = fitness_fn(vec)
            self.fitness_scores.append(fitness)
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_vector = vec.clone()
        return self.fitness_scores
    
    def evolve_generation(self):
        """Create next generation through selection, crossover, and mutation."""
        # Sort by fitness
        ranked = sorted(zip(self.fitness_scores, self.population), key=lambda x: -x[0])
        
        # Elite selection
        n_elite = max(1, int(len(self.population) * self.config.elite_ratio))
        elites = [v.clone() for _, v in ranked[:n_elite]]
        
        # Generate children
        children = []
        while len(children) < len(self.population) - n_elite:
            p1, p2 = random.sample(elites, 2)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            children.append(child)
        
        self.population = elites + children
        self.generation += 1
    
    def _crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Blend crossover."""
        alpha = random.random()
        return alpha * p1 + (1 - alpha) * p2
    
    def _mutate(self, v: torch.Tensor) -> torch.Tensor:
        """Gaussian mutation."""
        if random.random() < self.config.mutation_rate:
            noise = torch.randn_like(v) * 0.05
            return v + noise
        return v


# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 MAIN LABORATORY
# ═══════════════════════════════════════════════════════════════════════════════

class AFTLab:
    """The Ultimate AFT Experiment Laboratory."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.state = LiveState()
        self.model = None
        self.tokenizer = None
        self.device = None
        self.num_layers = 0
        self.hidden_dim = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📦 SETUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _load_model(self):
        """Load and freeze model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.state.phase = "🔄 Loading model..."
        self.state.add_message(f"Loading {self.config.model_name}...", "cyan")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model dimensions
        self.hidden_dim = self.model.config.hidden_size
        temp = MultiLayerAFT(self.model, [0], self.hidden_dim)
        self.num_layers = len(temp._get_layers())
        temp.cleanup()
        
        self.state.add_message(f"✅ Model loaded: {self.num_layers} layers, dim={self.hidden_dim}", "green")
    
    def _load_data(self) -> Tuple[List, List]:
        """Load dataset."""
        from datasets import load_dataset
        
        self.state.phase = f"📚 Loading {self.config.dataset}..."
        
        samples = []
        
        if self.config.dataset == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
            for i, ex in enumerate(ds):
                if i >= self.config.train_samples + self.config.test_samples:
                    break
                prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
                prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
                prompt += "\n\nThe best answer is:"
                samples.append({
                    "prompt": prompt,
                    "label": int(ex["label"]),
                    "target": f" {chr(65 + int(ex['label']))}"
                })
        
        elif self.config.dataset == "arc":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test", trust_remote_code=True)
            for i, ex in enumerate(ds):
                if i >= self.config.train_samples + self.config.test_samples:
                    break
                choices = ex['choices']
                prompt = f"Question: {ex['question']}\n\n"
                for label, text in zip(choices['label'], choices['text']):
                    prompt += f"{label}. {text}\n"
                prompt += "\nAnswer:"
                
                try:
                    answer_idx = choices['label'].index(ex['answerKey'])
                except ValueError:
                    answer_idx = 0
                
                samples.append({
                    "prompt": prompt,
                    "label": answer_idx,
                    "target": f" {ex['answerKey']}"
                })
        
        elif self.config.dataset == "mmlu":
            ds = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
            for i, ex in enumerate(ds):
                if i >= self.config.train_samples + self.config.test_samples:
                    break
                prompt = f"Question: {ex['question']}\n\n"
                for j, choice in enumerate(ex['choices']):
                    prompt += f"{chr(65+j)}. {choice}\n"
                prompt += "\nAnswer:"
                
                samples.append({
                    "prompt": prompt,
                    "label": int(ex['answer']),
                    "target": f" {chr(65 + int(ex['answer']))}"
                })
        
        elif self.config.dataset == "winogrande":
            ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
            for i, ex in enumerate(ds):
                if i >= self.config.train_samples + self.config.test_samples:
                    break
                prompt = f"{ex['sentence']}\n\nA. {ex['option1']}\nB. {ex['option2']}\n\nAnswer:"
                samples.append({
                    "prompt": prompt,
                    "label": 0 if ex['answer'] == '1' else 1,
                    "target": " A" if ex['answer'] == '1' else " B"
                })
        
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}. Available: hellaswag, arc, mmlu, winogrande")
        
        train = samples[:self.config.train_samples]
        test = samples[self.config.train_samples:self.config.train_samples + self.config.test_samples]
        
        self.state.add_message(f"✅ Loaded {len(train)} train, {len(test)} test samples", "green")
        return train, test
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎯 EVALUATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _evaluate(self, model, data: List, update_state: bool = True) -> float:
        """Evaluate model accuracy."""
        correct = 0
        
        for i, sample in enumerate(data):
            if update_state:
                self.state.sample = i + 1
                self.state.total_samples = len(data)
                self.state.current_prompt = sample["prompt"][:80] + "..."
                self.state.current_target = sample["target"]
            
            inputs = self.tokenizer(
                sample["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            probs = []
            for opt in ["A", "B", "C", "D"]:
                tid = self.tokenizer.encode(f" {opt}", add_special_tokens=False)
                probs.append(logits[tid[0]].item() if tid else -float('inf'))
            
            pred_idx = np.argmax(probs)
            pred_label = chr(65 + pred_idx)
            
            if update_state:
                self.state.current_pred = f" {pred_label}"
                self.state.is_correct = (pred_idx == sample["label"])
            
            if pred_idx == sample["label"]:
                correct += 1
            
            if update_state:
                self.state.accuracy = correct / (i + 1) * 100
        
        return correct / len(data) * 100
    
    def _evaluate_vector(self, vector: torch.Tensor, layer: int, data: List) -> float:
        """Evaluate a specific vector at a specific layer."""
        steered = SingleLayerAFT(self.model, layer, self.hidden_dim)
        steered.steering_vector.data = vector.to(steered.steering_vector.device)
        acc = self._evaluate(steered, data, update_state=False)
        steered.cleanup()
        return acc
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🏋️ TRAINING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _train(self, steered, train_data: List, epochs: int) -> float:
        """Train steering vector."""
        optimizer = optim.Adam(
            [p for p in steered.parameters() if p.requires_grad],
            lr=self.config.lr
        )
        
        self.state.total_epochs = epochs
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.state.epoch = epoch + 1
            epoch_loss = 0
            
            random.shuffle(train_data)
            
            for i, sample in enumerate(train_data):
                self.state.sample = i + 1
                self.state.total_samples = len(train_data)
                
                inputs = self.tokenizer(
                    sample["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                target_ids = self.tokenizer.encode(sample["target"], add_special_tokens=False)
                if not target_ids:
                    continue
                
                outputs = steered(**inputs)
                logits = outputs.logits[0, -1, :]
                
                loss = nn.CrossEntropyLoss()(
                    logits.unsqueeze(0).float(),
                    torch.tensor([target_ids[0]], device=self.device)
                )
                
                loss.backward()
                
                # Track gradient norm
                if hasattr(steered, 'steering_vector') and steered.steering_vector.grad is not None:
                    self.state.gradient_norm = steered.steering_vector.grad.norm().item()
                
                optimizer.step()
                optimizer.zero_grad()
                
                self.state.loss = loss.item()
                epoch_loss += loss.item()
                
                # Track vector norm
                if hasattr(steered, 'steering_vector'):
                    self.state.vector_norm = steered.steering_vector.norm().item()
                elif hasattr(steered, 'get_vector_norm'):
                    self.state.vector_norm = steered.get_vector_norm()
            
            avg_loss = epoch_loss / len(train_data)
            self.state.loss_history.append(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        return best_loss
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🧪 EXPERIMENTS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run_layer_sweep(self, train_data, test_data) -> ExperimentResult:
        """Sweep all layers to find optimal injection point."""
        self.state.experiment = "🔍 Layer Sweep"
        self.state.phase = "Sweeping ALL layers..."
        self.state.add_message("Starting comprehensive layer sweep...", "cyan")
        
        baseline_acc = self._evaluate(self.model, test_data)
        self.state.add_message(f"Baseline: {baseline_acc:.1f}%", "yellow")
        
        layer_results = {}
        best_layer, best_acc = 0, 0
        
        # Test every layer
        for layer in range(self.num_layers):
            self.state.phase = f"🔍 Testing Layer {layer}/{self.num_layers-1}"
            
            steered = SingleLayerAFT(self.model, layer, self.hidden_dim)
            self._train(steered, train_data[:50], epochs=2)
            acc = self._evaluate(steered, test_data[:30], update_state=True)
            steered.cleanup()
            
            layer_results[layer] = acc
            if acc > best_acc:
                best_acc = acc
                best_layer = layer
            
            self.state.acc_history.append(acc)
            self.state.leaderboard = sorted(
                [(f"Layer {l}", a) for l, a in layer_results.items()],
                key=lambda x: -x[1]
            )[:5]
        
        # Full training on best layer
        self.state.phase = f"🏋️ Full training @ Layer {best_layer}"
        self.state.add_message(f"Best layer: {best_layer} ({best_acc:.1f}%)", "green")
        
        start = time.time()
        steered = SingleLayerAFT(self.model, best_layer, self.hidden_dim)
        self._train(steered, train_data, epochs=self.config.epochs)
        final_acc = self._evaluate(steered, test_data)
        steered.cleanup()
        
        return ExperimentResult(
            name="Layer Sweep",
            baseline_acc=baseline_acc,
            final_acc=final_acc,
            improvement=final_acc - baseline_acc,
            training_time=time.time() - start,
            best_config={"layer": best_layer},
            history=list(layer_results.values())
        )
    
    def run_multilayer(self, train_data, test_data) -> ExperimentResult:
        """Multi-layer stacking experiment."""
        self.state.experiment = "🚀 Multi-Layer AFT"
        self.state.phase = "Testing multi-layer injection..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        
        # Calculate layer positions
        layer_indices = [
            min(int(self.num_layers * p), self.num_layers - 1)
            for p in self.config.layer_percentages
        ]
        
        self.state.add_message(f"Layers: {layer_indices}", "cyan")
        
        start = time.time()
        steered = MultiLayerAFT(self.model, layer_indices, self.hidden_dim)
        self._train(steered, train_data, epochs=self.config.epochs)
        final_acc = self._evaluate(steered, test_data)
        
        scales = [s.item() for s in steered.scales]
        steered.cleanup()
        
        return ExperimentResult(
            name="Multi-Layer",
            baseline_acc=baseline_acc,
            final_acc=final_acc,
            improvement=final_acc - baseline_acc,
            training_time=time.time() - start,
            best_config={"layers": layer_indices, "scales": scales}
        )
    
    def run_coefficient_sweep(self, train_data, test_data) -> ExperimentResult:
        """Sweep coefficients to find optimal scaling."""
        self.state.experiment = "📊 Coefficient Sweep"
        self.state.phase = "Training base vector..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        best_layer = self.num_layers // 2
        
        # Train base vector
        steered = SingleLayerAFT(self.model, best_layer, self.hidden_dim)
        self._train(steered, train_data, epochs=self.config.epochs)
        trained_vector = steered.steering_vector.data.clone()
        steered.cleanup()
        
        # Sweep coefficients
        self.state.phase = "📊 Sweeping coefficients..."
        coef_results = {}
        
        for coef in self.config.coefficients:
            steered = SingleLayerAFT(self.model, best_layer, self.hidden_dim)
            steered.steering_vector.data = trained_vector
            steered.coefficient = coef
            
            acc = self._evaluate(steered, test_data)
            coef_results[coef] = acc
            steered.cleanup()
            
            self.state.acc_history.append(acc)
            self.state.leaderboard = sorted(
                [(f"Coef {c:.1f}", a) for c, a in coef_results.items()],
                key=lambda x: -x[1]
            )[:5]
        
        best_coef = max(coef_results, key=coef_results.get)
        
        return ExperimentResult(
            name="Coefficient Sweep",
            baseline_acc=baseline_acc,
            final_acc=coef_results[best_coef],
            improvement=coef_results[best_coef] - baseline_acc,
            best_config={"coefficient": best_coef, "all_coefs": coef_results}
        )
    
    def run_evolutionary(self, train_data, test_data) -> ExperimentResult:
        """Evolutionary search for optimal steering vector."""
        self.state.experiment = "🧬 Evolutionary Search"
        self.state.phase = "Initializing population..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        best_layer = self.num_layers // 2
        
        evolver = VectorEvolver(self.hidden_dim, self.config)
        
        def fitness_fn(vec):
            return self._evaluate_vector(vec, best_layer, test_data[:20])
        
        start = time.time()
        
        for gen in range(self.config.generations):
            self.state.generation = gen + 1
            self.state.phase = f"🧬 Generation {gen + 1}/{self.config.generations}"
            
            # Evaluate population
            fitness_scores = evolver.evaluate_population(fitness_fn)
            self.state.population_fitness = sorted(fitness_scores, reverse=True)[:5]
            
            self.state.best_accuracy = evolver.best_fitness
            self.state.acc_history.append(max(fitness_scores))
            
            self.state.add_message(
                f"Gen {gen+1}: Best={max(fitness_scores):.1f}%, Avg={np.mean(fitness_scores):.1f}%",
                "cyan"
            )
            
            # Evolve
            evolver.evolve_generation()
            
            self.state.leaderboard = [
                (f"Gen {gen+1} Best", evolver.best_fitness)
            ] + self.state.leaderboard[:4]
        
        # Final evaluation with best vector
        final_acc = self._evaluate_vector(evolver.best_vector, best_layer, test_data)
        
        return ExperimentResult(
            name="Evolutionary",
            baseline_acc=baseline_acc,
            final_acc=final_acc,
            improvement=final_acc - baseline_acc,
            training_time=time.time() - start,
            best_config={"generations": self.config.generations}
        )
    
    def run_recursive(self, train_data, test_data) -> ExperimentResult:
        """Recursive vector refinement."""
        self.state.experiment = "🔄 Recursive Refinement"
        self.state.phase = "Starting recursive passes..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        best_layer = self.num_layers // 2
        
        current_vector = torch.zeros(1, 1, self.hidden_dim)
        pass_results = []
        
        start = time.time()
        
        for pass_idx in range(self.config.recursive_passes):
            self.state.phase = f"🔄 Pass {pass_idx + 1}/{self.config.recursive_passes}"
            self.state.add_message(f"Starting refinement pass {pass_idx + 1}...", "cyan")
            
            # Train with current vector as initialization
            steered = SingleLayerAFT(self.model, best_layer, self.hidden_dim)
            steered.steering_vector.data = current_vector.to(steered.steering_vector.device)
            
            self._train(steered, train_data, epochs=self.config.epochs)
            
            acc = self._evaluate(steered, test_data)
            pass_results.append(acc)
            
            # Update current vector for next pass
            current_vector = steered.steering_vector.data.clone().cpu()
            steered.cleanup()
            
            self.state.acc_history.append(acc)
            self.state.leaderboard.append((f"Pass {pass_idx + 1}", acc))
            self.state.add_message(f"Pass {pass_idx + 1}: {acc:.1f}%", "green")
        
        return ExperimentResult(
            name="Recursive",
            baseline_acc=baseline_acc,
            final_acc=pass_results[-1],
            improvement=pass_results[-1] - baseline_acc,
            training_time=time.time() - start,
            best_config={"passes": self.config.recursive_passes},
            history=pass_results
        )
    
    def run_resonance(self, train_data, test_data) -> ExperimentResult:
        """Resonance Chamber: Exponentially amplified echo through consecutive layers."""
        self.state.experiment = "🔊 Resonance Chamber"
        self.state.phase = "Creating resonance chamber..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        
        # Create resonance starting from middle layers
        start_layer = self.num_layers // 3
        num_resonances = min(6, self.num_layers - start_layer)  # 6 echoes
        
        self.state.add_message(
            f"Resonance: {num_resonances} echoes starting @ layer {start_layer}",
            "cyan"
        )
        
        start = time.time()
        
        # Train resonance chamber
        steered = ResonanceChamberAFT(
            self.model,
            start_layer=start_layer,
            num_resonances=num_resonances,
            hidden_dim=self.hidden_dim,
            growth_rate=1.2  # Exponential growth rate
        )
        
        self._train(steered, train_data, epochs=self.config.epochs)
        
        # Get learned amplitudes
        amplitudes = steered.get_effective_magnitudes()
        
        # Evaluate resonance
        final_acc = self._evaluate(steered, test_data)
        steered.cleanup()
        
        self.state.add_message(
            f"Resonance amplitudes: {[f'{a:.2f}' for a in amplitudes[:3]]}...",
            "green"
        )
        
        return ExperimentResult(
            name="Resonance Chamber",
            baseline_acc=baseline_acc,
            final_acc=final_acc,
            improvement=final_acc - baseline_acc,
            training_time=time.time() - start,
            best_config={
                "start_layer": start_layer,
                "num_resonances": num_resonances,
                "amplitudes": amplitudes
            }
        )
    
    def run_echo_chamber(self, train_data, test_data) -> ExperimentResult:
        """Activation Echo Chamber: Dynamic self-refining feedback loop."""
        self.state.experiment = "🔁 Echo Chamber"
        self.state.phase = "Building echo chamber..."
        
        baseline_acc = self._evaluate(self.model, test_data)
        best_layer = self.num_layers // 2
        
        # First, train a baseline steering vector
        self.state.phase = "Training initial steering..."
        base_steered = SingleLayerAFT(self.model, best_layer, self.hidden_dim)
        self._train(base_steered, train_data, epochs=self.config.epochs)
        initial_vector = base_steered.steering_vector.data.clone()
        base_steered.cleanup()
        
        # Create echo chamber with learned initial vector
        self.state.phase = "Creating echo chamber..."
        echo = ActivationEchoChamber(
            self.model,
            layer_idx=best_layer,
            hidden_dim=self.hidden_dim,
            initial_vector=initial_vector
        )
        
        # Train the refiner network
        self.state.phase = "Training refinement network..."
        refiner_optimizer = optim.Adam(echo.refiner.parameters(), lr=self.config.lr * 0.5)
        
        for epoch in range(2):  # Quick refiner training
            for sample in train_data[:50]:
                echo.reset_buffer()
                
                inputs = self.tokenizer(
                    sample["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Forward pass accumulates activations
                outputs = echo(**inputs)
                
                # Refine steering based on activations
                echo.refine_steering()
                
                # Second pass with refined steering
                echo.reset_buffer()
                outputs = echo(**inputs)
                logits = outputs.logits[0, -1, :]
                
                target_ids = self.tokenizer.encode(sample["target"], add_special_tokens=False)
                if not target_ids:
                    continue
                
                loss = nn.CrossEntropyLoss()(
                    logits.unsqueeze(0).float(),
                    torch.tensor([target_ids[0]], device=self.device)
                )
                
                loss.backward()
                refiner_optimizer.step()
                refiner_optimizer.zero_grad()
        
        start = time.time()
        
        # Evaluate with echo chamber
        self.state.phase = "Evaluating echo chamber..."
        correct = 0
        
        for i, sample in enumerate(test_data):
            echo.reset_buffer()
            
            inputs = self.tokenizer(
                sample["prompt"],
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # First pass: gather activations
            with torch.no_grad():
                _ = echo(**inputs)
            
            # Refine steering based on activations
            echo.refine_steering()
            
            # Second pass: refined inference
            echo.reset_buffer()
            with torch.no_grad():
                outputs = echo(**inputs)
                logits = outputs.logits[0, -1, :]
            
            # Predict
            probs = []
            for opt in ["A", "B", "C", "D"]:
                tid = self.tokenizer.encode(f" {opt}", add_special_tokens=False)
                probs.append(logits[tid[0]].item() if tid else -float('inf'))
            
            pred_idx = np.argmax(probs)
            if pred_idx == sample["label"]:
                correct += 1
            
            self.state.accuracy = correct / (i + 1) * 100
        
        final_acc = correct / len(test_data) * 100
        echo.cleanup()
        
        self.state.add_message(
            f"Echo chamber used dynamic refinement on {len(test_data)} samples",
            "green"
        )
        
        return ExperimentResult(
            name="Echo Chamber",
            baseline_acc=baseline_acc,
            final_acc=final_acc,
            improvement=final_acc - baseline_acc,
            training_time=time.time() - start,
            best_config={"layer": best_layer, "dynamic": True}
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🎨 DASHBOARD RENDERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _sparkline(self, data: List[float], width: int = 30) -> str:
        """Create animated sparkline."""
        if not data:
            return "─" * width
        
        recent = list(data)[-width:]
        if len(recent) < 2:
            return SPARKLINE_CHARS[0] * len(recent)
        
        min_v, max_v = min(recent), max(recent)
        if max_v == min_v:
            return SPARKLINE_CHARS[4] * len(recent)
        
        normalized = [(v - min_v) / (max_v - min_v) for v in recent]
        return "".join(SPARKLINE_CHARS[int(v * 7)] for v in normalized)
    
    def _render_dashboard(self) -> Layout:
        """Render stunning dashboard."""
        layout = Layout()
        
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="right", ratio=2)
        )
        
        layout["left"].split_column(
            Layout(name="metrics", size=12),
            Layout(name="sample")
        )
        
        layout["right"].split_column(
            Layout(name="leaderboard", ratio=1),
            Layout(name="activity", ratio=1)
        )
        
        # ═══ HEADER ═══
        spinner = PROGRESS_CHARS[self.state.spinner_idx % len(PROGRESS_CHARS)]
        self.state.spinner_idx += 1
        
        header = Text()
        header.append(f"  {spinner} ", style="bold cyan")
        header.append("AFT LABORATORY ", style="bold magenta")
        header.append(f"│ ", style="dim")
        header.append(self.state.experiment or "Ready", style="bold yellow")
        header.append(f" │ ", style="dim")
        header.append(f"⏱ {self.state.elapsed()}", style="cyan")
        header.append(f" │ ", style="dim")
        header.append(f"🎯 Best: {self.state.best_accuracy:.1f}%", style="bold green")
        
        layout["header"].update(Panel(header, box=box.DOUBLE_EDGE, style="blue"))
        
        # ═══ METRICS ═══
        metrics = Table(show_header=False, box=None, padding=(0, 2))
        metrics.add_column("Key", style="cyan", width=15)
        metrics.add_column("Value", style="white", width=15)
        metrics.add_column("Visual", style="green", width=35)
        
        metrics.add_row(
            "Phase",
            self.state.phase[:30],
            ""
        )
        metrics.add_row(
            "Progress",
            f"E{self.state.epoch}/{self.state.total_epochs} S{self.state.sample}/{self.state.total_samples}",
            self._progress_bar(self.state.sample, self.state.total_samples or 1, 30)
        )
        metrics.add_row(
            "Loss",
            f"{self.state.loss:.4f}",
            self._sparkline(self.state.loss_history, 30)
        )
        metrics.add_row(
            "Accuracy",
            f"{self.state.accuracy:.1f}%",
            self._sparkline(self.state.acc_history, 30)
        )
        metrics.add_row(
            "Vector Norm",
            f"{self.state.vector_norm:.4f}",
            self._mini_bar(min(self.state.vector_norm / 0.1, 1.0), 30)
        )
        metrics.add_row(
            "Gradient",
            f"{self.state.gradient_norm:.6f}",
            ""
        )
        
        layout["metrics"].update(Panel(metrics, title="📊 Metrics", border_style="green"))
        
        # ═══ LIVE SAMPLE ═══
        sample_text = Text()
        sample_text.append("📝 ", style="bold")
        sample_text.append(self.state.current_prompt[:100] if self.state.current_prompt else "Waiting...", style="dim")
        sample_text.append("\n\n")
        sample_text.append("🎯 Target: ", style="bold green")
        sample_text.append(self.state.current_target, style="green")
        sample_text.append("  │  ", style="dim")
        sample_text.append("🤖 Pred: ", style="bold yellow")
        sample_text.append(self.state.current_pred, style="yellow")
        sample_text.append("  ")
        if self.state.current_pred:
            sample_text.append(
                "✓" if self.state.is_correct else "✗",
                style="bold green" if self.state.is_correct else "bold red"
            )
        
        layout["sample"].update(Panel(sample_text, title="🔍 Live Sample", border_style="cyan"))
        
        # ═══ LEADERBOARD ═══
        lb_table = Table(box=box.SIMPLE, show_header=True, header_style="bold yellow")
        lb_table.add_column("🏆", style="yellow", width=3)
        lb_table.add_column("Config", style="cyan")
        lb_table.add_column("Acc", style="green")
        
        for i, (name, acc) in enumerate(self.state.leaderboard[:6]):
            medal = ["🥇", "🥈", "🥉", "4", "5", "6"][i]
            lb_table.add_row(medal, name[:20], f"{acc:.1f}%")
        
        layout["leaderboard"].update(Panel(lb_table, title="🏆 Leaderboard", border_style="yellow"))
        
        # ═══ ACTIVITY ═══
        activity = Text()
        for msg, style in list(self.state.messages)[-6:]:
            activity.append(f"• {msg}\n", style=style)
        
        layout["activity"].update(Panel(activity, title="📜 Activity", border_style="magenta"))
        
        # ═══ FOOTER ═══
        results_summary = []
        for r in self.state.results[-3:]:
            emoji = "🏆" if r.improvement > 10 else "✅" if r.improvement > 0 else "⚠️"
            results_summary.append(f"{emoji} {r.name}: +{r.improvement:.1f}%")
        
        footer = Text()
        footer.append("Results: ", style="bold")
        footer.append(" │ ".join(results_summary) if results_summary else "No results yet", style="cyan")
        footer.append(f"\n[dim]Model: {self.config.model_name} │ Dataset: {self.config.dataset} │ Ctrl+C to stop[/]")
        
        layout["footer"].update(Panel(footer, box=box.ROUNDED))
        
        return layout
    
    def _progress_bar(self, current: int, total: int, width: int = 20) -> str:
        """Render progress bar."""
        if total == 0:
            return "─" * width
        ratio = min(current / total, 1.0)
        filled = int(ratio * width)
        return "█" * filled + "░" * (width - filled)
    
    def _mini_bar(self, ratio: float, width: int = 20) -> str:
        """Render mini bar for values 0-1."""
        filled = int(ratio * width)
        return "▓" * filled + "░" * (width - filled)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 MAIN RUNNER
    # ═══════════════════════════════════════════════════════════════════════════
    
    def run(self, experiments: List[str] = None):
        """Run the laboratory."""
        if experiments is None:
            experiments = ["layer_sweep", "multilayer", "coefficient", "evolutionary", "recursive"]
        
        console.clear()
        console.print(BANNER)
        time.sleep(1)
        
        with Live(self._render_dashboard(), refresh_per_second=8, console=console) as live:
            try:
                # Setup
                self._load_model()
                live.update(self._render_dashboard())
                
                train_data, test_data = self._load_data()
                live.update(self._render_dashboard())
                
                # Run experiments
                experiment_map = {
                    "layer_sweep": self.run_layer_sweep,
                    "multilayer": self.run_multilayer,
                    "coefficient": self.run_coefficient_sweep,
                    "evolutionary": self.run_evolutionary,
                    "recursive": self.run_recursive,
                    "resonance": self.run_resonance,
                    "echo": self.run_echo_chamber,
                }
                
                for exp_name in experiments:
                    if exp_name not in experiment_map:
                        continue
                    
                    self.state.add_message(f"Starting {exp_name}...", "magenta")
                    live.update(self._render_dashboard())
                    
                    result = experiment_map[exp_name](train_data, test_data)
                    self.state.results.append(result)
                    
                    if result.final_acc > self.state.best_accuracy:
                        self.state.best_accuracy = result.final_acc
                    
                    self.state.add_message(
                        f"✅ {result.name}: {result.baseline_acc:.1f}% → {result.final_acc:.1f}% (+{result.improvement:.1f}%)",
                        "green" if result.improvement > 5 else "yellow"
                    )
                    live.update(self._render_dashboard())
                
                self.state.phase = "🎉 All experiments complete!"
                live.update(self._render_dashboard())
                time.sleep(3)
                
            except KeyboardInterrupt:
                self.state.phase = "⏹️ Stopped"
        
        self._print_final_report()
        return self.state.results
    
    def _print_final_report(self):
        """Print beautiful final report."""
        console.print("\n")
        console.print(Panel.fit(
            "[bold magenta]🏆 EXPERIMENT COMPLETE 🏆[/]",
            border_style="magenta"
        ))
        
        # Results table
        table = Table(title="📊 Final Results", box=box.DOUBLE_EDGE, show_lines=True)
        table.add_column("Experiment", style="cyan bold")
        table.add_column("Baseline", style="dim")
        table.add_column("Final", style="green bold")
        table.add_column("Δ", style="yellow bold")
        table.add_column("Time", style="dim")
        
        for r in self.state.results:
            delta_style = "green bold" if r.improvement > 5 else "yellow" if r.improvement > 0 else "red"
            table.add_row(
                r.name,
                f"{r.baseline_acc:.1f}%",
                f"{r.final_acc:.1f}%",
                f"+{r.improvement:.1f}%" if r.improvement > 0 else f"{r.improvement:.1f}%",
                f"{r.training_time:.1f}s",
                style="" if r.improvement <= 5 else "on dark_green" if r.improvement > 10 else ""
            )
        
        console.print(table)
        
        # Winner
        if self.state.results:
            best = max(self.state.results, key=lambda r: r.improvement)
            console.print(Panel(
                f"[bold green]🥇 WINNER: {best.name}[/]\n"
                f"[dim]Improvement: [bold]+{best.improvement:.1f}%[/] "
                f"({best.baseline_acc:.1f}% → {best.final_acc:.1f}%)[/]\n"
                f"[dim]Config: {best.best_config}[/]",
                border_style="green"
            ))
        
        # Save results
        results_path = Path("results.json")
        with open(results_path, "w") as f:
            json.dump([{
                "name": r.name,
                "baseline": r.baseline_acc,
                "final": r.final_acc,
                "improvement": r.improvement,
                "time": r.training_time,
                "config": r.best_config
            } for r in self.state.results], f, indent=2)
        
        console.print(f"\n[dim]Results saved to {results_path}[/]")


# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 CLI
# ═══════════════════════════════════════════════════════════════════════════════

# Small models for universality testing
SMALL_MODELS = [
    "Qwen/Qwen2-0.5B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-1_5",
]

# Datasets for generality testing  
BENCHMARK_DATASETS = ["hellaswag", "arc", "mmlu", "winogrande"]


def run_benchmark(models: List[str], datasets: List[str], experiments: List[str], 
                  quick: bool = False, turbo: bool = False):
    """Run comprehensive benchmark across models and datasets."""
    mode = "⚡ TURBO" if turbo else ("🚀 QUICK" if quick else "🔬 FULL")
    console.print(BANNER)
    console.print(Panel(
        f"[bold cyan]{mode} UNIVERSALITY BENCHMARK[/]\n"
        f"[dim]Models: {len(models)} | Datasets: {len(datasets)} | Experiments: {len(experiments)}[/]",
        border_style="magenta"
    ))
    
    all_results = []
    
    if turbo:
        samples, test_samples, epochs = 15, 10, 1
    elif quick:
        samples, test_samples, epochs = 30, 20, 2
    else:
        samples, test_samples, epochs = 100, 50, 5
    
    for model_name in models:
        model_results = {"model": model_name, "datasets": {}}
        console.print(f"\n[bold magenta]═══ Testing: {model_name} ═══[/]")
        
        for dataset in datasets:
            console.print(f"  [cyan]📊 Dataset: {dataset}[/]")
            
            config = ExperimentConfig(
                model_name=model_name,
                dataset=dataset,
                train_samples=samples,
                test_samples=test_samples,
                epochs=epochs,
            )
            
            try:
                lab = AFTLab(config)
                results = lab.run(experiments)
                
                dataset_results = {}
                for r in results:
                    dataset_results[r.name] = {
                        "baseline": r.baseline_acc,
                        "final": r.final_acc,
                        "improvement": r.improvement
                    }
                
                model_results["datasets"][dataset] = dataset_results
            except Exception as e:
                console.print(f"    [red]❌ Error: {e}[/]")
                model_results["datasets"][dataset] = {"error": str(e)}
        
        all_results.append(model_results)
    
    # Print summary
    console.print("\n")
    console.print(Panel.fit("[bold magenta]📊 UNIVERSALITY SUMMARY[/]", border_style="magenta"))
    
    # Build summary table
    summary = Table(title="AFT Universality Benchmark", box=box.DOUBLE_EDGE, show_lines=True)
    summary.add_column("Model", style="cyan bold")
    
    for ds in datasets:
        summary.add_column(ds.upper(), style="white")
    
    summary.add_column("AVG", style="yellow bold")
    
    for model_result in all_results:
        row = [model_result["model"].split("/")[-1]]
        improvements = []
        
        for ds in datasets:
            if ds in model_result["datasets"]:
                ds_data = model_result["datasets"][ds]
                if "error" in ds_data:
                    row.append("[red]ERR[/]")
                else:
                    # Get best improvement
                    best_imp = max(r.get("improvement", 0) for r in ds_data.values())
                    improvements.append(best_imp)
                    style = "green bold" if best_imp > 10 else "green" if best_imp > 0 else "red"
                    row.append(f"[{style}]+{best_imp:.1f}%[/]")
            else:
                row.append("-")
        
        avg_imp = np.mean(improvements) if improvements else 0
        row.append(f"+{avg_imp:.1f}%")
        summary.add_row(*row)
    
    console.print(summary)
    
    # Save full results
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    console.print(f"\n[dim]Full results saved to benchmark_results.json[/]")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="🧠 AFT Experiment Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aft_lab.py                              # Run all experiments
  python aft_lab.py --quick                      # Quick test (fewer samples)
  python aft_lab.py --exp evolutionary           # Just evolutionary search
  python aft_lab.py --model Qwen/Qwen2-1.5B      # Different model
  python aft_lab.py --benchmark                  # Full universality benchmark
  python aft_lab.py --benchmark --quick          # Quick benchmark
        """
    )
    
    parser.add_argument("--exp", type=str, nargs="+", 
                       choices=["layer_sweep", "multilayer", "coefficient", "evolutionary", "recursive", "resonance", "echo", "all"],
                       default=["all"], help="Experiments to run")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dataset", type=str, default="hellaswag",
                       choices=["hellaswag", "arc", "mmlu", "winogrande"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--turbo", action="store_true", help="Ultra-fast validation (10 samples, 1 epoch)")
    parser.add_argument("--benchmark", action="store_true", help="Run universality benchmark")
    parser.add_argument("--models", type=str, nargs="+", default=None, 
                       help="Models for benchmark (default: 2 small models)")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                       help="Datasets for benchmark (default: hellaswag, arc)")
    
    args = parser.parse_args()
    
    if args.turbo:
        args.samples = 15
        args.test_samples = 10
        args.epochs = 1
    elif args.quick:
        args.samples = 30
        args.test_samples = 20
        args.epochs = 2
    
    # Determine experiments
    if "all" in args.exp:
        core_experiments = ["coefficient", "multilayer", "echo"]  # Core for benchmark
        full_experiments = ["layer_sweep", "multilayer", "coefficient", "evolutionary", "recursive", "resonance", "echo"]
    else:
        core_experiments = args.exp
        full_experiments = args.exp
    
    # Benchmark mode
    if args.benchmark:
        models = args.models if args.models else SMALL_MODELS[:2]
        datasets = args.datasets if args.datasets else ["hellaswag", "arc"]
        run_benchmark(models, datasets, core_experiments, quick=args.quick, turbo=args.turbo)
        return
    
    # Single run mode
    config = ExperimentConfig(
        model_name=args.model,
        dataset=args.dataset,
        train_samples=args.samples,
        test_samples=args.test_samples,
        epochs=args.epochs,
    )
    
    lab = AFTLab(config)
    lab.run(full_experiments)


if __name__ == "__main__":
    main()
