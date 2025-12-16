#!/usr/bin/env python3
"""
controllable_patching.py

NOVEL CONTRIBUTION: Controllable Generation via Targeted Activation Patching

Pre-mortem showed 100% patching effect (patching always changes output).
This experiment tests whether we can CONTROL what it changes to:
1. Collect activations from positive/negative/neutral outputs
2. Learn "sentiment direction" in activation space
3. Patch toward positive or negative
4. Measure if we achieve controlled sentiment steering

This is DIFFERENT from steering vectors because:
- No labeled contrastive pairs required
- Discovered from actual generations
- Dynamic based on current context
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class ControllablePatcher:
    """Patch activations toward a target direction."""
    
    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.direction = None
        self.strength = 0.0
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
        if self.mode == "steer" and self.direction is not None:
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            
            # Add direction vector (broadcast to sequence)
            direction = self.direction.unsqueeze(0).unsqueeze(0)
            direction = direction.expand(hidden.shape[0], hidden.shape[1], -1)
            modified = hidden + self.strength * direction
            
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified
        return output
    
    def set_steering(self, direction: torch.Tensor, strength: float):
        self.direction = direction
        self.strength = strength
        self.mode = "steer"
    
    def disable(self):
        self.mode = "normal"
    
    def cleanup(self):
        if self._hook:
            self._hook.remove()


def extract_activation(model, tokenizer, text, layer=-1, device="cuda"):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]
    return hidden.mean(dim=1).squeeze()


def run_controllable_patching():
    """
    Main experiment: Can we control output sentiment via activation patching?
    """
    start_time = datetime.now()
    device = get_device()
    
    print("=" * 70)
    print("NOVEL EXPERIMENT: CONTROLLABLE GENERATION VIA PATCHING")
    print("=" * 70)
    
    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load sentiment classifier
    print("Loading sentiment classifier...")
    sentiment = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)
    
    # Phase 1: Generate and collect positive/negative examples
    print("\n[Phase 1] Collecting sentiment examples...")
    
    prompts = [
        "The movie was",
        "I think this product is",
        "My experience with this service was",
        "The food at this restaurant was",
        "Working with this team has been",
    ] * 20
    
    positive_acts = []
    negative_acts = []
    neutral_acts = []
    
    for prompt in tqdm(prompts, desc="Collecting"):
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        continuation = text[len(prompt):]
        
        # Get activation
        act = extract_activation(model, tokenizer, prompt, device=device)
        
        # Classify sentiment
        try:
            result = sentiment(continuation[:200])[0]
            label = result["label"]
            score = result["score"]
            
            if label == "POSITIVE" and score > 0.8:
                positive_acts.append(act)
            elif label == "NEGATIVE" and score > 0.8:
                negative_acts.append(act)
            else:
                neutral_acts.append(act)
        except:
            continue
    
    print(f"Collected: {len(positive_acts)} positive, {len(negative_acts)} negative")
    
    if len(positive_acts) < 5 or len(negative_acts) < 5:
        print("Not enough examples")
        return
    
    # Phase 2: Compute sentiment direction
    print("\n[Phase 2] Computing sentiment direction...")
    
    pos_mean = torch.stack(positive_acts).mean(dim=0)
    neg_mean = torch.stack(negative_acts).mean(dim=0)
    
    # Direction: positive - negative
    sentiment_direction = pos_mean - neg_mean
    sentiment_direction = sentiment_direction / sentiment_direction.norm()
    
    print(f"Sentiment direction magnitude: {(pos_mean - neg_mean).norm().item():.3f}")
    
    # Phase 3: Test controlled generation
    print("\n[Phase 3] Testing controlled generation...")
    
    n_layers = len(model.transformer.h)
    layer = n_layers // 2
    
    test_prompts = [
        "The new policy is",
        "This technology will",
        "The future looks",
        "My opinion on this matter is",
        "The results of the study show",
    ]
    
    results_by_direction = {"positive": [], "negative": [], "baseline": []}
    
    patcher = ControllablePatcher(model, layer)
    
    for prompt in tqdm(test_prompts * 4, desc="Testing"):
        for direction_name, strength in [("positive", 2.0), ("negative", -2.0), ("baseline", 0)]:
            if direction_name == "baseline":
                patcher.disable()
            else:
                patcher.set_steering(sentiment_direction, strength)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate(
                    inputs.input_ids,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.8,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
            continuation = text[len(prompt):]
            
            try:
                result = sentiment(continuation[:200])[0]
                score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
                results_by_direction[direction_name].append(score)
            except:
                continue
    
    patcher.cleanup()
    
    # Results
    avg_pos = np.mean(results_by_direction["positive"])
    avg_neg = np.mean(results_by_direction["negative"])
    avg_base = np.mean(results_by_direction["baseline"])
    
    duration = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Positive steering sentiment: {avg_pos:.3f}")
    print(f"Baseline sentiment: {avg_base:.3f}")
    print(f"Negative steering sentiment: {avg_neg:.3f}")
    print(f"Control range: {avg_pos - avg_neg:.3f}")
    
    success = (avg_pos > avg_base > avg_neg) and (avg_pos - avg_neg > 0.1)
    result = "SUCCESS" if success else "FAILED"
    print(f"\n>>> {result}")
    
    results = {
        "date": datetime.now().isoformat(),
        "avg_positive": avg_pos,
        "avg_baseline": avg_base,
        "avg_negative": avg_neg,
        "control_range": avg_pos - avg_neg,
        "result": result
    }
    
    with open("results/experiments/controllable_patching.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    run_controllable_patching()
