#!/usr/bin/env python3
"""
aft_universality.py - Universal Activation Fine-Tuning Script

This script generalizes the AFT technique to multiple models and datasets.
It supports:
- Models: TinyLlama-1.1B, Qwen-1.5-1.8B, Gemma-2B, Phi-2
- Datasets: HellaSwag, ARC-Challenge, GSM8K
- Features: Automated Layer Sweep, Strict Train/Test Split, Live Progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
import argparse
import sys
import time
import os
from typing import List, Dict, Any, Optional

# --- Utilities ---

def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def print_progress(current, total, prefix="", suffix="", bar_len=30):
    filled_len = int(round(bar_len * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '█' * filled_len + '░' * (bar_len - filled_len)
    sys.stdout.write(f'\r{prefix} [{bar}] {percents}% {suffix}')
    sys.stdout.flush()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Wrapper ---

class ModelWrapper:
    """
    Abstracts away architecture differences (e.g., layer access).
    """
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name
        print(f"📦 Loading {model_name}...")
        start_load = time.time()
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"✅ Model loaded & frozen in {time.time()-start_load:.1f}s")

    def get_layers(self):
        """Returns the list of transformer layers."""
        if hasattr(self.model, 'model'): # Llama, Qwen, Mistral
            return self.model.model.layers
        elif hasattr(self.model, 'transformer'): # Phi-2
            return self.model.transformer.h
        elif hasattr(self.model, 'gpt_neox'): # Pythia / GPT-NeoX
            return self.model.gpt_neox.layers
        else:
            # Fallback: Try to find any module list named 'layers' or 'h'
            if hasattr(self.model, 'layers'): return self.model.layers
            if hasattr(self.model, 'h'): return self.model.h
            raise ValueError(f"Unknown model architecture for {self.model_name}")

    def get_hidden_dim(self):
        return self.model.config.hidden_size

# --- Steered Model ---

class SteeredModel(nn.Module):
    """Wraps a model to inject a learnable vector at a specific layer."""
    def __init__(self, wrapper: ModelWrapper, layer_idx: int):
        super().__init__()
        self.wrapper = wrapper
        self.layer_idx = layer_idx
        self.hidden_dim = wrapper.get_hidden_dim()
        
        # The learnable steering vector
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, self.hidden_dim, dtype=torch.float32))
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        layers = self.wrapper.get_layers()
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Inject vector (broadcast over batch/seq)
            vector = self.steering_vector.to(hidden.dtype).to(hidden.device)
            modified = hidden + vector
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.wrapper.model(input_ids, attention_mask=attention_mask, labels=labels)

    def cleanup(self):
        if self.hook_handle: self.hook_handle.remove()

    def generate(self, *args, **kwargs):
        # Forward generate call to the underlying model
        # Note: This won't use the steering vector unless we re-register the hook on the inner model
        # during generation. The current hook is registered on the inner model's layers, 
        # so it SHOULD work if the inner model's forward is called.
        return self.wrapper.model.generate(*args, **kwargs)

# --- Data Loading ---

# --- Data Loading ---

def load_data(dataset_name, tokenizer, max_samples=400):
    print(f"📚 Loading {dataset_name} (limit={max_samples})...")
    samples = []
    
    if dataset_name == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
            prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
            prompt += "\n\nThe best answer is:"
            target_token = f" {chr(65 + int(ex['label']))}"
            samples.append({
                "prompt": prompt,
                "label": int(ex["label"]),
                "target_text": target_token,
                "type": "multiple_choice"
            })
            
    elif dataset_name == "arc_challenge":
        ds = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\nChoices:\n"
            choices = ex['choices']
            label_map = {label: i for i, label in enumerate(choices['label'])}
            
            for label, text in zip(choices['label'], choices['text']):
                prompt += f"{label}. {text}\n"
            prompt += "\nAnswer:"
            
            target_label_idx = label_map[ex['answerKey']]
            target_token = f" {ex['answerKey']}"
            
            samples.append({
                "prompt": prompt,
                "label": target_label_idx,
                "target_text": target_token,
                "type": "multiple_choice",
                "choices": choices['label'] # Store choice labels (A, B, C, D, 1, 2, 3, 4)
            })

    elif dataset_name == "gsm8k":
        # GSM8K is generation, but for AFT we can treat it as next-token prediction on the reasoning path?
        # Or just standard SFT on the answer?
        # For now, let's stick to the plan: "Positive transfer to distinct reasoning tasks".
        # We'll try to learn to output the *first step* or just standard causal loss on the solution.
        ds = load_dataset("gsm8k", "main", split="test")
        for i, ex in enumerate(ds):
            if i >= max_samples: break
            prompt = f"Question: {ex['question']}\n\nAnswer:"
            # Extract the numerical answer
            answer_part = ex['answer'].split("####")[-1].strip()
            samples.append({
                "prompt": prompt,
                "target_text": ex['answer'], # Full answer for training
                "answer_number": answer_part, # For evaluation
                "type": "generation"
            })
            
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Split
    split_idx = len(samples) // 2
    train_set = samples[:split_idx]
    test_set = samples[split_idx:]
    print(f"✅ Loaded {len(train_set)} train, {len(test_set)} test samples")
    return train_set, test_set

# --- Evaluation ---

def evaluate(model, tokenizer, samples, device, label="Eval"):
    model.eval()
    correct = 0
    total = 0
    
    for i, s in enumerate(samples):
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        if s["type"] == "multiple_choice":
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
            
            probs = []
            choices = s.get("choices", ["A", "B", "C", "D"])
            for opt in choices:
                tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
                # Handle cases where tokenizer might split " A" differently
                if not tid: 
                     # Fallback for some tokenizers
                     tid = tokenizer.encode(opt, add_special_tokens=False)
                
                probs.append(logits[tid[0]].item() if tid else -float('inf'))
            
            if np.argmax(probs) == s["label"]:
                correct += 1
            total += 1
            
        elif s["type"] == "generation":
            # For GSM8K, we might just check perplexity or do a short generation
            # For this script, let's do a simple generation check if it contains the answer
            # But generation is slow. Let's stick to perplexity/loss for "eval" on generation tasks for now?
            # Or just skip GSM8K eval in this loop and rely on training loss?
            # Let's do a quick generation for GSM8K
            # GSM8K Evaluation: Generate and check for the number
            with torch.no_grad():
                # Generate up to 100 tokens to allow for reasoning + answer
                gen = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
            
            # Decode only the new tokens
            generated_text = tokenizer.decode(gen[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Check if the answer number is in the generated text
            # This is a loose check, but better than nothing for a quick script
            if s["answer_number"] in generated_text:
                correct += 1
            total += 1
            
        print_progress(i+1, len(samples), prefix=f"  {label}:", suffix=f"Acc: {correct/(total if total else 1):.1%}")
    
    sys.stdout.write("\n")
    return correct / total if total > 0 else 0.0

# --- Training ---

def train_vector(wrapper, train_set, test_set, layer_idx, epochs, lr, device, verbose=True):
    steered_model = SteeredModel(wrapper, layer_idx)
    optimizer = optim.Adam([steered_model.steering_vector], lr=lr)
    
    if verbose: print_banner(f"🏋️ TRAINING LAYER {layer_idx}", "-")
    
    loss_history = []
    
    for epoch in range(epochs):
        if verbose: print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0
        np.random.shuffle(train_set)
        
        steered_model.train()
        
        for i, s in enumerate(train_set):
            inputs = wrapper.tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=1024).to(device)
            
            # Target handling
            if s["type"] == "multiple_choice":
                target_text = s["target_text"]
                target_ids = wrapper.tokenizer.encode(target_text, add_special_tokens=False)
                if not target_ids: continue
                target_id = target_ids[0]
                
                outputs = steered_model(**inputs)
                logits = outputs.logits[0, -1, :]
                loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([target_id], device=device))
                
            elif s["type"] == "generation":
                # For generation, we train on the whole answer? Or just the first few tokens?
                # Let's train on the whole answer (Teacher Forcing)
                full_text = s["prompt"] + s["target_text"]
                inputs_full = wrapper.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
                labels = inputs_full.input_ids.clone()
                # Mask prompt
                prompt_len = inputs.input_ids.shape[1]
                labels[:, :prompt_len] = -100
                
                outputs = steered_model(inputs_full.input_ids, attention_mask=inputs_full.attention_mask, labels=labels)
                loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(steered_model.parameters(), 1.0)
            
            epoch_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            
            # Progress
            loss_str = f"{loss.item():.4f}"
            if verbose: print_progress(i+1, len(train_set), prefix="  Train:", suffix=f"Loss: {loss_str}")
            
        avg_loss = epoch_loss / len(train_set)
        loss_history.append(avg_loss)
        if verbose: print(f"\n  📉 Avg Loss: {avg_loss:.4f}")
        
    return steered_model, loss_history

# --- Main ---

# --- Main Execution Logic ---

def run_experiment(model_name, dataset_name, layer=None, auto_layer=False, epochs=5, lr=1e-3, device="cuda", verbose=True, limit=400):
    if verbose:
        print_banner(f"🧠 AFT UNIVERSALITY: {model_name} on {dataset_name} 🧠")
    
    # 1. Load Model
    try:
        wrapper = ModelWrapper(model_name, device)
    except Exception as e:
        if verbose: print(f"❌ Failed to load model: {e}")
        return {"error": str(e)}
    
    # 2. Load Data
    try:
        train_set, test_set = load_data(dataset_name, wrapper.tokenizer, max_samples=limit)
    except Exception as e:
        if verbose: print(f"❌ Failed to load data: {e}")
        return {"error": str(e)}
    
    # 3. Baseline
    if verbose: print_banner("📊 BASELINE EVALUATION", "-")
    baseline_acc = evaluate(wrapper.model, wrapper.tokenizer, test_set, device, label="Baseline")
    if verbose: print(f"  🎯 Baseline Accuracy: {baseline_acc:.1%}")
    
    # 4. Determine Layer
    target_layer = layer
    if auto_layer:
        if verbose: print_banner("🔍 AUTO-LAYER SWEEP", "-")
        best_loss = float('inf')
        best_layer = 0
        
        # Sweep middle layers (e.g., 25% to 75% depth)
        num_layers = len(wrapper.get_layers())
        start_layer = int(num_layers * 0.25)
        end_layer = int(num_layers * 0.75)
        
        if verbose: print(f"Sweeping layers {start_layer} to {end_layer}...")
        
        for l in range(start_layer, end_layer + 1, 2): # Step by 2 to save time
            # Train for 1 epoch on subset
            subset = train_set[:50]
            model, history = train_vector(wrapper, subset, [], l, epochs=1, lr=lr, device=device, verbose=False) # Mute sweep details
            final_loss = history[-1]
            if verbose: print(f"  Layer {l}: Loss {final_loss:.4f}")
            
            if final_loss < best_loss:
                best_loss = final_loss
                best_layer = l
            
            model.cleanup() # Remove hook before next layer
        
        if verbose: print(f"✅ Best Layer Found: {best_layer}")
        target_layer = best_layer
    
    if target_layer is None:
        # Default to middle
        target_layer = len(wrapper.get_layers()) // 2
        if verbose: print(f"⚠️ No layer specified, defaulting to middle layer: {target_layer}")

    # 5. Train Final Vector
    steered_model, loss_history = train_vector(wrapper, train_set, test_set, target_layer, epochs, lr, device, verbose=verbose)
    
    # 6. Final Eval
    if verbose: print_banner("🚀 FINAL EVALUATION", "-")
    final_acc = evaluate(steered_model, wrapper.tokenizer, test_set, device, label="Final")
    improvement = final_acc - baseline_acc
    
    if verbose:
        print_banner("🎉 RESULTS", "=")
        print(f"  🎯 Baseline Accuracy:    {baseline_acc:.1%}")
        print(f"  🧠 Steered Accuracy:     {final_acc:.1%}")
        print(f"  📈 Improvement:          {improvement:+.1%}")
    
    # Save results
    os.makedirs("results/universality", exist_ok=True)
    model_slug = model_name.replace("/", "_")
    filename = f"results/universality/{model_slug}_{dataset_name}.json"
    
    results = {
        "model": model_name,
        "dataset": dataset_name,
        "layer": target_layer,
        "baseline_acc": baseline_acc,
        "final_acc": final_acc,
        "improvement": improvement,
        "loss_history": loss_history
    }
    
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    if verbose: print(f"\n💾 Results saved to {filename}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="AFT Universality Experiment")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., microsoft/phi-2)")
    parser.add_argument("--dataset", type=str, default="hellaswag", choices=["hellaswag", "arc_challenge", "gsm8k"])
    parser.add_argument("--layer", type=int, default=None, help="Layer to inject vector")
    parser.add_argument("--auto-layer", action="store_true", help="Sweep layers to find best")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=400, help="Max samples to load")
    args = parser.parse_args()
    
    device = get_device()
    run_experiment(args.model, args.dataset, args.layer, args.auto_layer, args.epochs, args.lr, device, verbose=True, limit=args.limit)

if __name__ == "__main__":
    main()
