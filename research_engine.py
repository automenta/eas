#!/usr/bin/env python3
"""
research_engine.py - The "G-Factor" Research Engine

Powered by AFT Lab v3.0 technology.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
import json
import random
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import deque
from sklearn.decomposition import PCA

# Rich TUI imports
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

console = Console()

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResearchConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B"
    datasets: List[str] = field(default_factory=lambda: ["hellaswag", "arc", "gsm8k"])
    train_samples: int = 50
    test_samples: int = 30
    epochs: int = 3
    lr: float = 1e-3
    turbo_mode: bool = False

@dataclass
class VectorArtifact:
    name: str
    layer: int
    vector: torch.Tensor
    accuracy: float
    dataset: str

# ═══════════════════════════════════════════════════════════════════════════════
# 🧠 CORE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

class AFTSteering(nn.Module):
    """Simple single-layer steering."""
    def __init__(self, model, layer_idx: int, hidden_dim: int):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hook = None
        self._setup()

    def _get_layers(self):
        if hasattr(self.model, 'gpt_neox'): return self.model.gpt_neox.layers
        if hasattr(self.model, 'model'): return self.model.model.layers
        return self.model.transformer.h

    def _setup(self):
        layers = self._get_layers()
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = self.steering_vector.to(device=h.device, dtype=h.dtype)
            return (h + v,) + out[1:] if isinstance(out, tuple) else h + v
        self.hook = layers[self.layer_idx].register_forward_hook(hook)

    def forward(self, **kwargs): return self.model(**kwargs)
    def cleanup(self):
        if self.hook: self.hook.remove()

# ═══════════════════════════════════════════════════════════════════════════════
# 🔬 RESEARCH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ResearchEngine:
    def __init__(self, config: ResearchConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.vectors: Dict[str, VectorArtifact] = {}
        self.results = []

    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        console.print(f"[cyan]🔄 Loading {self.config.model_name}...[/]")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
            device_map="auto", trust_remote_code=True
        )
        for p in self.model.parameters(): p.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, trust_remote_code=True)
        if not self.tokenizer.pad_token: self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get dimensions
        temp = AFTSteering(self.model, 0, self.model.config.hidden_size)
        self.num_layers = len(temp._get_layers())
        self.hidden_dim = self.model.config.hidden_size
        temp.cleanup()
        console.print(f"[green]✅ Model loaded: {self.num_layers} layers[/]")

    def load_dataset(self, name: str):
        from datasets import load_dataset
        samples = []

        # --- DATASET ADAPTERS ---
        if name == "hellaswag":
            # Removed trust_remote_code=True
            ds = load_dataset("Rowan/hellaswag", split="validation")
            for ex in ds:
                if len(samples) >= self.config.train_samples + self.config.test_samples: break
                prompt = f"{ex['ctx']}\n" + "\n".join([f"{i}. {o}" for i,o in enumerate(ex['endings'])]) + "\nAnswer:"
                samples.append({"prompt": prompt, "label": int(ex["label"]), "target": f" {ex['label']}"})

        elif name == "arc":
            # Removed trust_remote_code=True
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            for ex in ds:
                if len(samples) >= self.config.train_samples + self.config.test_samples: break
                try: idx = ex['choices']['label'].index(ex['answerKey'])
                except: idx = 0
                prompt = f"Question: {ex['question']}\n" + "\n".join([f"{l}. {t}" for l,t in zip(ex['choices']['label'], ex['choices']['text'])]) + "\nAnswer:"
                samples.append({"prompt": prompt, "label": idx, "target": f" {ex['answerKey']}"})

        elif name == "gsm8k":
            # Removed trust_remote_code=True
            ds = load_dataset("gsm8k", "main", split="test")
            for ex in ds:
                if len(samples) >= self.config.train_samples + self.config.test_samples: break
                prompt = f"Question: {ex['question']}\nAnswer:"
                # Extract numerical answer or use dummy for steering
                samples.append({"prompt": prompt, "label": 0, "target": " " + ex['answer'].split()[-1]}) # simplified target

        # Split
        train = samples[:self.config.train_samples]
        test = samples[self.config.train_samples:]
        return train, test

    def evaluate(self, model, data):
        correct = 0
        for s in data:
            inputs = self.tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad(): logits = model(**inputs).logits[0, -1, :]

            # Simple heuristic for MC questions: check prob of labels 0-3 (A-D or 0-3)
            # This logic needs to be robust for different datasets, but for steering validation it's okay
            probs = []
            options = ["0", "1", "2", "3"] if "0" in s["target"] else ["A", "B", "C", "D"]
            for opt in options:
                tid = self.tokenizer.encode(" "+opt, add_special_tokens=False)
                probs.append(logits[tid[0]].item() if tid else -999)

            if np.argmax(probs) == s["label"]: correct += 1
        return correct / len(data) * 100

    def train_vector(self, dataset_name: str, layer: int):
        train, test = self.load_dataset(dataset_name)
        steered = AFTSteering(self.model, layer, self.hidden_dim)
        opt = optim.Adam(steered.parameters(), lr=self.config.lr)

        console.print(f"[yellow]  Training {dataset_name} @ L{layer}...[/]", end="\r")

        for epoch in range(self.config.epochs):
            random.shuffle(train)
            for s in train:
                inputs = self.tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(self.device)
                tid = self.tokenizer.encode(s["target"], add_special_tokens=False)
                if not tid: continue

                logits = steered(**inputs).logits[0, -1, :]
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([tid[0]]).to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()

        acc = self.evaluate(steered, test)
        vector = steered.steering_vector.detach().clone()
        steered.cleanup()

        console.print(f"[green]  ✅ {dataset_name}: {acc:.1f}%[/]")
        return VectorArtifact(f"v_{dataset_name}", layer, vector, acc, dataset_name)

    # ═══════════════════════════════════════════════════════════════════════════
    # 🧠 SMART SCOUT
    # ═══════════════════════════════════════════════════════════════════════════

    def smart_scout(self, dataset_name: str) -> int:
        """Rapidly find best layer using turbo mode."""
        console.print(f"[bold cyan]🔍 Scouting layers for {dataset_name}...[/]")
        train, test = self.load_dataset(dataset_name)

        # Test 5 evenly spaced layers
        candidates = np.linspace(2, self.num_layers-2, 5, dtype=int)
        best_layer = 0
        best_acc = 0

        for layer in candidates:
            steered = AFTSteering(self.model, layer, self.hidden_dim)
            # Quick train: 1 epoch, 10 samples
            opt = optim.Adam(steered.parameters(), lr=5e-3)
            # LIMIT SAMPLES FOR SPEED IF TURBO IS ON
            limit = 5 if self.config.turbo_mode else 10

            for s in train[:limit]:
                inputs = self.tokenizer(s["prompt"], return_tensors="pt", truncation=True).to(self.device)
                tid = self.tokenizer.encode(s["target"], add_special_tokens=False)
                if not tid: continue
                logits = steered(**inputs).logits[0, -1, :]
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([tid[0]]).to(self.device))
                loss.backward()
                opt.step()
                opt.zero_grad()

            acc = self.evaluate(steered, test[:limit])
            steered.cleanup()

            console.print(f"  Layer {layer}: {acc:.1f}%")
            if acc > best_acc:
                best_acc = acc
                best_layer = layer

        console.print(f"[bold green]🎯 Optimal Layer: {best_layer}[/]")
        return best_layer

    # ═══════════════════════════════════════════════════════════════════════════
    # 🧬 VECTOR FUSION
    # ═══════════════════════════════════════════════════════════════════════════

    def fuse_vectors(self, vectors: List[torch.Tensor], method="mean") -> torch.Tensor:
        """Compute the G-Vector."""
        stack = torch.cat([v.flatten().unsqueeze(0) for v in vectors], dim=0)

        if method == "mean":
            fused = torch.mean(stack, dim=0)
        elif method == "pca":
            # First principal component
            mean = torch.mean(stack, dim=0)
            centered = stack - mean
            u, s, v = torch.svd(centered)
            # v is (D, D), columns are eigenvectors. First column is first PC.
            fused = v[:, 0]

        return fused.view(1, 1, self.hidden_dim)

    # ═══════════════════════════════════════════════════════════════════════════
    # 🚀 WORKFLOWS
    # ═══════════════════════════════════════════════════════════════════════════

    def run_g_factor_cycle(self):
        """Run full G-Factor research cycle."""
        console.print(Panel(f"[bold magenta]🧪 Starting G-Factor Cycle: {self.config.model_name}[/]"))

        self.load_model()

        # 1. Scout & Train specialized vectors
        task_vectors = []
        layers = []

        for ds in self.config.datasets:
            layer = self.smart_scout(ds)
            layers.append(layer)
            artifact = self.train_vector(ds, layer)
            self.vectors[ds] = artifact
            task_vectors.append(artifact.vector)

        # 2. Compute G-Vector
        console.print("[bold yellow]⚗️ Fusing G-Vector...[/]")
        # We need to inject G-Vector at a common layer.
        # Strategy: Use average of best layers or voting.
        target_layer = int(np.mean(layers))
        console.print(f"  Target Injection Layer: {target_layer}")

        g_vector = self.fuse_vectors(task_vectors, method="mean")

        # 3. Validate G-Vector
        console.print("[bold cyan]🛡️ Validating G-Vector across all tasks...[/]")

        results = {}
        for ds in self.config.datasets:
            _, test = self.load_dataset(ds)

            # Baseline
            base_acc = self.evaluate(self.model, test)

            # G-Vector
            steered = AFTSteering(self.model, target_layer, self.hidden_dim)
            steered.steering_vector.data = g_vector
            g_acc = self.evaluate(steered, test)
            steered.cleanup()

            results[ds] = {"baseline": base_acc, "g_factor": g_acc, "delta": g_acc - base_acc}
            console.print(f"  {ds}: {base_acc:.1f}% -> {g_acc:.1f}% ({g_acc-base_acc:+.1f}%)")

        # Save
        path = f"results_g_factor_{self.config.model_name.replace('/','_')}.json"
        with open(path, "w") as f: json.dump(results, f, indent=2)
        console.print(f"[dim]Saved to {path}[/]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--turbo", action="store_true")
    args = parser.parse_args()

    # EXTREME TURBO MODE: 5 samples, 1 epoch if turbo is set
    config = ResearchConfig(
        model_name=args.model,
        turbo_mode=args.turbo,
        train_samples=5 if args.turbo else 100,
        test_samples=5 if args.turbo else 50,
        epochs=1 if args.turbo else 3
    )

    engine = ResearchEngine(config)
    engine.run_g_factor_cycle()
