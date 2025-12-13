#!/usr/bin/env python3
"""
capable_model_test.py

Test activation-based interventions on models that CAN ACTUALLY REASON.
- Phi-2 (2.7B): ~60% on reasoning tasks
- TinyLlama (1.1B): ~40-50% on reasoning tasks
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import json
from datetime import datetime


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


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


def test_model_baseline(model_name: str, n_samples: int = 200):
    """Test baseline accuracy on HellaSwag."""
    device = get_device()
    
    print(f"\n{'='*60}")
    print(f"TESTING: {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load data
    print("Loading HellaSwag...")
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    samples = []
    for i, ex in enumerate(dataset):
        if i >= n_samples:
            break
        prompt = f"Complete the sentence:\n{ex['ctx']}\n\n"
        prompt += "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['endings'])])
        prompt += "\n\nThe best answer is:"
        samples.append({
            "prompt": prompt,
            "correct_idx": int(ex["label"])
        })
    
    # Evaluate
    print(f"Evaluating on {len(samples)} samples...")
    correct = 0
    for sample in tqdm(samples, desc="Testing"):
        try:
            pred, _ = get_model_prediction(model, tokenizer, sample["prompt"], device)
            if pred == sample["correct_idx"]:
                correct += 1
        except Exception as e:
            continue
    
    accuracy = correct / len(samples)
    
    print(f"\n>>> {model_name}: {accuracy:.1%} accuracy ({correct}/{len(samples)})")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    
    return accuracy


def main():
    print("=" * 60)
    print("FINDING SMALLEST REASONING-CAPABLE MODELS")
    print("=" * 60)
    
    models_to_test = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "microsoft/phi-2",
    ]
    
    results = {}
    for model_name in models_to_test:
        try:
            acc = test_model_baseline(model_name, n_samples=100)
            results[model_name] = acc
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            results[model_name] = None
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model, acc in results.items():
        if acc:
            status = "✓ CAN REASON" if acc > 0.35 else "✗ Random"
            print(f"  {model}: {acc:.1%} {status}")
        else:
            print(f"  {model}: FAILED")
    
    # Save results
    with open("results/experiments/model_capabilities.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
