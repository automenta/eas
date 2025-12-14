#!/usr/bin/env python3
"""
poc_aft.py - "Activation Fine-Tuning"

INNOVATION: Instead of guessing the steering vector, we LEARN it.
We freeze the model and use Gradient Descent to find the optimal
activation injection vector that minimizes loss on the correct answer.

This creates a "Reasoning Booster" that can be permanently added to the model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import json
from datetime import datetime
import sys
import time

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

class SteeredModel(nn.Module):
    """Wraps a model to inject a learnable vector at a specific layer."""
    def __init__(self, model, layer_idx, hidden_dim):
        super().__init__()
        self.model = model
        self.layer_idx = layer_idx
        # The learnable steering vector
        self.steering_vector = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        if hasattr(self.model, 'model'): layers = self.model.model.layers
        else: layers = self.model.transformer.h
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Inject vector (broadcast over batch/seq)
            # hidden: [B, S, D]
            # vector: [1, 1, D]
            # Ensure vector is on the same device and dtype as hidden
            vector = self.steering_vector.to(hidden.device).to(hidden.dtype)
            modified = hidden + vector
            return (modified,) + output[1:] if isinstance(output, tuple) else modified
        
        self.hook_handle = layers[self.layer_idx].register_forward_hook(hook_fn)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def cleanup(self):
        if self.hook_handle: self.hook_handle.remove()

def get_accuracy(model, tokenizer, samples, device):
    correct = 0
    model.eval()
    for i, s in enumerate(samples):
        inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1, :]
        
        probs = []
        for opt in ["A", "B", "C", "D"]:
            tid = tokenizer.encode(f" {opt}", add_special_tokens=False)
            probs.append(logits[tid[0]].item() if tid else -float('inf'))
        
        if np.argmax(probs) == s["label"]:
            correct += 1
        
        if (i+1) % 10 == 0:
            print_progress(i+1, len(samples), prefix="  Eval:", suffix=f"Acc: {correct/(i+1):.1%}")
    sys.stdout.write("\n")
    return correct / len(samples)

def main():
    device = get_device()
    print_banner("🧠 PHI-2 ACTIVATION FINE-TUNING (AFT) 🧠")
    
    # 1. Load Data
    print("📚 Loading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation")
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= 400: break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        
        target_token = f" {chr(65 + int(ex['label']))}"
        
        samples.append({
            "prompt": prompt,
            "label": int(ex["label"]),
            "target_text": target_token
        })
    
    train_set = samples[:200]
    test_set = samples[200:400]
    print(f"✅ Loaded {len(train_set)} train, {len(test_set)} test samples")

    # 2. Load Model
    print("\n📦 Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Model loaded & frozen")

    # 3. Setup Steering
    hidden_dim = model.config.hidden_size
    layer_idx = 16 # Middle layer
    print(f"\n📍 Injecting learnable vector at Layer {layer_idx}")
    
    steered_model = SteeredModel(model, layer_idx, hidden_dim)
    
    # Optimizer - only optimize the vector
    optimizer = optim.Adam([steered_model.steering_vector], lr=1e-2)
    
    # 4. Baseline Evaluation
    print_banner("📊 BASELINE EVALUATION", "-")
    baseline_acc = get_accuracy(model, tokenizer, test_set, device)
    print(f"Baseline Accuracy: {baseline_acc:.1%}")

    # 5. Training Loop
    print_banner("🏋️ TRAINING REASONING VECTOR", "-")
    
    epochs = 5
    batch_size = 4 # Gradient accumulation steps (simulated)
    
    steered_model.train() # Enable grads (for the vector)
    
    loss_history = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        # Shuffle train set
        np.random.shuffle(train_set)
        
        for i, s in enumerate(train_set):
            # Prepare input
            inputs = tokenizer(s["prompt"], return_tensors="pt", truncation=True, max_length=512).to(device)
            target_ids = tokenizer.encode(s["target_text"], add_special_tokens=False)
            if not target_ids: continue
            target_id = target_ids[0]
            
            # Forward pass
            outputs = steered_model(**inputs)
            logits = outputs.logits[0, -1, :] # Logits for next token
            
            # Loss: maximize prob of target_id
            loss = nn.CrossEntropyLoss()(logits.unsqueeze(0), torch.tensor([target_id], device=device))
            
            # Backward
            loss.backward()
            
            epoch_loss += loss.item()
            
            # Update every batch_size steps
            if (i+1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            print_progress(i+1, len(train_set), prefix="  Train:", suffix=f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_set)
        loss_history.append(avg_loss)
        print(f"\n  Avg Loss: {avg_loss:.4f}")

    # 6. Final Evaluation
    print_banner("🚀 FINAL EVALUATION", "-")
    
    # Switch to eval mode (though vector is still active)
    steered_model.eval() 
    
    final_acc = get_accuracy(model, tokenizer, test_set, device)
    improvement = final_acc - baseline_acc
    
    print(f"Baseline Accuracy: {baseline_acc:.1%}")
    print(f"Steered Accuracy:  {final_acc:.1%}")
    print(f"Improvement:       {improvement:+.1%}")
    
    vector_norm = torch.norm(steered_model.steering_vector).item()
    print(f"Learned Vector Norm: {vector_norm:.4f}")
    
    if improvement > 0.02:
        print("\n🎉 SUCCESS: Learned vector significantly improves performance!")
    elif improvement > 0:
        print("\n📈 PARTIAL: Small improvement learned.")
    else:
        print("\n❌ RESULT: Failed to learn beneficial vector.")

    steered_model.cleanup()

if __name__ == "__main__":
    import os
    os.makedirs("results/experiments", exist_ok=True)
    main()
