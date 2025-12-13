#!/usr/bin/env python3
"""
skill_transfer.py

NOVEL CONTRIBUTION: Unsupervised Skill Transfer via Activation Projection

Pre-mortem showed 72% correlation between GPT-2 and Pythia activations.
This experiment tests whether we can:
1. Extract "skill activations" from a capable model on easy examples
2. Project them to a less capable model's space
3. INJECT them during inference on hard examples
4. Measure if the less capable model improves

This is DIFFERENT from knowledge distillation because:
- No training/fine-tuning
- Works at inference time
- Transfers via activation space, not outputs
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import Ridge
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import json
from datetime import datetime


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ActivationInjector:
    """Inject projected activations from source model into target model."""
    
    def __init__(self, model, layer_idx: int, injection_strength: float = 0.5):
        self.model = model
        self.layer_idx = layer_idx
        self.injection_strength = injection_strength
        self.injection_vector = None
        self.mode = "normal"
        self._hook = None
        self._setup()
    
    def _get_layers(self):
        if hasattr(self.model, 'transformer'):
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'):
            return self.model.gpt_neox.layers
        raise ValueError("Unknown architecture")
    
    def _setup(self):
        layers = self._get_layers()
        self._hook = layers[self.layer_idx].register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        if self.mode == "inject" and self.injection_vector is not None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            injection = self.injection_vector.unsqueeze(0).unsqueeze(0)
            injection = injection.expand(hidden.shape[0], hidden.shape[1], -1)
            modified = hidden + self.injection_strength * injection
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return output
    
    def set_injection(self, vector: torch.Tensor):
        self.injection_vector = vector
        self.mode = "inject"
    
    def disable(self):
        self.mode = "normal"
    
    def cleanup(self):
        if self._hook:
            self._hook.remove()


def get_model_answer_probs(model, tokenizer, prompt, device):
    """Get probability distribution over answer tokens."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
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
    
    return probs


def run_skill_transfer():
    """
    Main experiment: Can we transfer "reasoning skill" via activation projection?
    """
    start_time = datetime.now()
    device = get_device()
    
    print("=" * 70)
    print("NOVEL EXPERIMENT: UNSUPERVISED SKILL TRANSFER")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    source_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device).eval()
    source_tok = AutoTokenizer.from_pretrained("gpt2-medium")
    source_tok.pad_token = source_tok.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    target_tok = AutoTokenizer.from_pretrained("gpt2")
    target_tok.pad_token = target_tok.eos_token
    
    # Load data
    print("\nLoading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 500:
            break
        prompt = f"Context: {ex['ctx']}\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\nAnswer:"
        samples.append({
            "prompt": prompt,
            "context": ex["ctx"],
            "correct_idx": int(ex["label"])
        })
    
    print(f"Loaded {len(samples)} samples")
    
    # Phase 1: Find EASY examples (source correct)
    print("\n[Phase 1] Finding easy examples...")
    easy_samples = []
    easy_acts_source = []
    easy_acts_target = []
    
    for sample in tqdm(samples[:200], desc="Evaluating"):
        probs = get_model_answer_probs(source_model, source_tok, sample["prompt"], device)
        pred = probs.index(max(probs))
        if pred == sample["correct_idx"]:
            easy_samples.append(sample)
            
            inputs = source_tok(sample["context"], return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = source_model(**inputs, output_hidden_states=True)
            act_s = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            easy_acts_source.append(act_s)
            
            inputs = target_tok(sample["context"], return_tensors="pt", truncation=True, max_length=128).to(device)
            with torch.no_grad():
                out = target_model(**inputs, output_hidden_states=True)
            act_t = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
            easy_acts_target.append(act_t)
    
    print(f"Found {len(easy_samples)} easy samples")
    
    # Phase 2: Learn projection
    print("\n[Phase 2] Learning activation projection...")
    X = np.vstack(easy_acts_source)
    Y = np.vstack(easy_acts_target)
    
    projector = Ridge(alpha=1.0)
    projector.fit(X, Y)
    
    mean_success = X.mean(axis=0)
    success_target = projector.predict(mean_success.reshape(1, -1))[0]
    success_target = torch.tensor(success_target, device=device, dtype=torch.float32)
    
    print(f"Projection: {X.shape[1]}d -> {Y.shape[1]}d")
    
    # Phase 3: Test on HARD examples
    print("\n[Phase 3] Testing on hard examples...")
    hard_samples = []
    for sample in tqdm(samples[200:400], desc="Finding hard"):
        probs = get_model_answer_probs(target_model, target_tok, sample["prompt"], device)
        pred = probs.index(max(probs))
        if pred != sample["correct_idx"]:
            hard_samples.append(sample)
    
    print(f"Found {len(hard_samples)} hard samples")
    
    n_layers = len(target_model.transformer.h)
    layer = n_layers // 2
    
    baseline_correct = 0
    injected_correct = 0
    
    injector = ActivationInjector(target_model, layer, injection_strength=0.3)
    
    for sample in tqdm(hard_samples[:50], desc="Testing"):
        injector.disable()
        probs = get_model_answer_probs(target_model, target_tok, sample["prompt"], device)
        if probs.index(max(probs)) == sample["correct_idx"]:
            baseline_correct += 1
        
        injector.set_injection(success_target)
        probs = get_model_answer_probs(target_model, target_tok, sample["prompt"], device)
        if probs.index(max(probs)) == sample["correct_idx"]:
            injected_correct += 1
    
    injector.cleanup()
    
    n_test = min(len(hard_samples), 50)
    baseline_acc = baseline_correct / n_test
    injected_acc = injected_correct / n_test
    improvement = injected_acc - baseline_acc
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Baseline: {baseline_acc:.1%}")
    print(f"With injection: {injected_acc:.1%}")
    print(f"Improvement: {improvement:+.1%}")
    
    result = "SUCCESS" if improvement > 0.05 else ("PARTIAL" if improvement > 0 else "FAILED")
    print(f"\n>>> {result}")
    
    results = {
        "date": datetime.now().isoformat(),
        "baseline_accuracy": baseline_acc,
        "injected_accuracy": injected_acc,
        "improvement": improvement,
        "result": result
    }
    
    with open("results/experiments/skill_transfer.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    run_skill_transfer()
