#!/usr/bin/env python3
"""
phi2_sentiment_demo.py - The "Magic" Demo

Demonstrates undeniable control over the model's output by steering
sentiment from Negative to Positive in real-time.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import sys

def print_banner(text, char="="):
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

class SteeringHook:
    def __init__(self, model):
        self.model = model
        self.layer_idx = None
        self.direction = None
        self.strength = 0.0
        self.handle = None
        self.activations = []
        self.mode = "off"

    def set_layer(self, layer_idx):
        if self.handle: self.handle.remove()
        if hasattr(self.model, 'model'): layers = self.model.model.layers
        else: layers = self.model.transformer.h
        self.layer_idx = layer_idx
        self.handle = layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.mode == "collect":
            self.activations.append(hidden.detach()[:,-1,:].float().cpu().numpy())
        elif self.mode == "steer" and self.direction is not None:
            dtype = hidden.dtype
            device = hidden.device
            steering = self.direction.to(device).to(dtype).view(1, 1, -1)
            return (hidden + self.strength * steering,) + output[1:] if isinstance(output, tuple) else (hidden + self.strength * steering)
        return output

    def cleanup(self):
        if self.handle: self.handle.remove()

def generate(model, tokenizer, prompt, max_new_tokens=30):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("🚀 Starting Sentiment Demo...", flush=True)
    device = get_device()
    print_banner("✨ PHI-2 SENTIMENT STEERING DEMO ✨")
    
    print("📦 Loading Phi-2...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("✅ Ready.")

    # 1. Extract Sentiment Vector
    print("\n🧪 Extracting Sentiment Vector...")
    pos_prompts = [
        "I absolutely loved this movie! It was",
        "The food was delicious and the service was",
        "This is the best day of my life because",
        "I am so happy and grateful for",
        "The product works perfectly and I am"
    ]
    neg_prompts = [
        "I absolutely hated this movie! It was",
        "The food was terrible and the service was",
        "This is the worst day of my life because",
        "I am so angry and disappointed about",
        "The product is broken and I am"
    ]
    
    hook = SteeringHook(model)
    layer = 16
    hook.set_layer(layer)
    
    # Collect Positive
    hook.mode = "collect"
    hook.activations = []
    for p in pos_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
    pos_mean = np.mean(np.concatenate(hook.activations, axis=0), axis=0)
    
    # Collect Negative
    hook.activations = []
    for p in neg_prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        with torch.no_grad(): model(**inputs)
    neg_mean = np.mean(np.concatenate(hook.activations, axis=0), axis=0)
    
    # Vector: Positive - Negative
    diff = pos_mean - neg_mean
    direction = torch.tensor(diff / np.linalg.norm(diff), dtype=torch.float32)
    hook.direction = direction
    print("✅ Vector Extracted.")

    # 2. The Magic Show
    print_banner("🎩 THE MAGIC SHOW 🎩", "-")
    
    test_prompts = [
        "The service at this restaurant was",
        "I think this game is",
        "My experience with the support team was",
        "The weather today is",
        "I feel"
    ]
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        
        # Baseline
        hook.mode = "off"
        base_out = generate(model, tokenizer, prompt)
        print(f"  ⚪ Normal:  {base_out[len(prompt):].strip()}")
        
        # Steered Positive
        hook.mode = "steer"
        hook.strength = 2.0 # Strong positive
        pos_out = generate(model, tokenizer, prompt)
        print(f"  🟢 Steered: \033[92m{pos_out[len(prompt):].strip()}\033[0m")
        
        # Steered Negative (Inverse)
        hook.strength = -2.0 # Strong negative
        neg_out = generate(model, tokenizer, prompt)
        print(f"  🔴 Inverse: \033[91m{neg_out[len(prompt):].strip()}\033[0m")

    print_banner("🎉 DEMO COMPLETE", "=")
    print("We can programmatically control the model's emotional state.")

if __name__ == "__main__":
    main()
