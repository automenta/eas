#!/usr/bin/env python3
"""
phi2_intervention.py

Test activation-based interventions on Phi-2 which CAN reason (38% baseline).
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import json
from datetime import datetime
from typing import Optional


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActivationCollector:
    """Collect activations from a specific layer."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.activation = None
        self._hook = None
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Phi-2, Llama
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer.h  # GPT-2
        raise ValueError("Unknown architecture")
    
    def _setup(self):
        layers = self._get_layers()
        self._hook = layers[self.layer_idx].register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.activation = hidden.detach().mean(dim=1).squeeze().cpu().numpy()
    
    def cleanup(self):
        if self._hook:
            self._hook.remove()


class ActivationInjector:
    """Inject activation vectors."""
    
    def __init__(self, model, layer_idx: int, strength: float = 0.3):
        self.model = model
        self.layer_idx = layer_idx
        self.strength = strength
        self.direction = None
        self.mode = "normal"
        self._hook = None
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            return self.model.transformer.h
        raise ValueError("Unknown architecture")
    
    def _setup(self):
        layers = self._get_layers()
        self._hook = layers[self.layer_idx].register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        if self.mode == "inject" and self.direction is not None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            direction = self.direction.unsqueeze(0).unsqueeze(0)
            direction = direction.expand(hidden.shape[0], hidden.shape[1], -1)
            modified = hidden + self.strength * direction
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return output
    
    def set_direction(self, direction: torch.Tensor):
        self.direction = direction
        self.mode = "inject"
    
    def disable(self):
        self.mode = "normal"
    
    def cleanup(self):
        if self._hook:
            self._hook.remove()


def get_model_prediction(model, tokenizer, prompt, device):
    """Get model's MCQ prediction."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    probs = []
    for opt in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(f" {opt}", add_special_tokens=False)
        if token_ids:
            prob = F.softmax(logits, dim=-1)[token_ids[0]].item()
        else:
            prob = 0.0
        probs.append(prob)
    
    return probs.index(max(probs)), max(probs)


def run_phi2_experiment():
    """Test activation interventions on Phi-2."""
    start_time = datetime.now()
    device = get_device()
    
    print("=" * 70)
    print("PHI-2 ACTIVATION INTERVENTION EXPERIMENT")
    print("=" * 70)
    
    # Load model
    print("\nLoading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("Loading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 400:
            break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({
            "prompt": prompt,
            "context": ex["ctx"],
            "correct_idx": int(ex["label"])
        })
    
    train_samples = samples[:200]
    test_samples = samples[200:]
    
    # Phase 1: Collect activations and learn "correct" direction
    print("\n[Phase 1] Learning correct/incorrect activation patterns...")
    
    n_layers = len(model.model.layers)
    layer_idx = n_layers // 2
    print(f"Using layer {layer_idx}/{n_layers}")
    
    collector = ActivationCollector(model, layer_idx)
    
    correct_acts = []
    incorrect_acts = []
    
    for sample in tqdm(train_samples, desc="Collecting activations"):
        inputs = tokenizer(sample["context"], return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = model(**inputs)
        
        act = collector.activation
        pred, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
        
        if pred == sample["correct_idx"]:
            correct_acts.append(act)
        else:
            incorrect_acts.append(act)
    
    collector.cleanup()
    
    print(f"Correct: {len(correct_acts)}, Incorrect: {len(incorrect_acts)}")
    
    if len(correct_acts) < 10:
        print("Not enough correct samples")
        return
    
    # Compute "correctness direction"
    correct_mean = np.mean(correct_acts, axis=0)
    incorrect_mean = np.mean(incorrect_acts, axis=0)
    correctness_direction = correct_mean - incorrect_mean
    correctness_direction = correctness_direction / np.linalg.norm(correctness_direction)
    correctness_direction = torch.tensor(correctness_direction, device=device, dtype=torch.float16)
    
    print(f"Direction magnitude: {np.linalg.norm(correct_mean - incorrect_mean):.3f}")
    
    # Phase 2: Test intervention
    print("\n[Phase 2] Testing activation steering...")
    
    baseline_correct = 0
    steered_correct = 0
    
    injector = ActivationInjector(model, layer_idx, strength=0.5)
    
    for sample in tqdm(test_samples, desc="Testing"):
        # Baseline
        injector.disable()
        pred_base, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
        if pred_base == sample["correct_idx"]:
            baseline_correct += 1
        
        # With steering toward "correct"
        injector.set_direction(correctness_direction)
        pred_steer, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
        if pred_steer == sample["correct_idx"]:
            steered_correct += 1
    
    injector.cleanup()
    
    n_test = len(test_samples)
    baseline_acc = baseline_correct / n_test
    steered_acc = steered_correct / n_test
    improvement = steered_acc - baseline_acc
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline accuracy: {baseline_acc:.1%}")
    print(f"Steered accuracy: {steered_acc:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    print(f"Duration: {duration:.1f} minutes")
    
    if improvement > 0.03:
        result = "SUCCESS"
    elif improvement > 0:
        result = "PARTIAL"
    else:
        result = "NO IMPROVEMENT"
    
    print(f"\n>>> {result}")
    
    results = {
        "date": datetime.now().isoformat(),
        "model": "microsoft/phi-2",
        "layer": layer_idx,
        "n_train": len(train_samples),
        "n_test": n_test,
        "n_correct_examples": len(correct_acts),
        "n_incorrect_examples": len(incorrect_acts),
        "baseline_accuracy": baseline_acc,
        "steered_accuracy": steered_acc,
        "improvement": improvement,
        "result": result
    }
    
    with open("results/experiments/phi2_intervention.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    run_phi2_experiment()
