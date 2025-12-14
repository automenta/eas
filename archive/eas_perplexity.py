#!/usr/bin/env python3
"""
eas_perplexity.py - Test EAS on perplexity/language modeling

Instead of MCQ accuracy, measure whether EAS improves:
1. Perplexity (lower = better language modeling)
2. Next-token prediction accuracy
3. Generation coherence

EAS hypothesis: steering toward "successful" activation patterns
should improve ANY language model metric, not just reasoning.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from eas_core import EASConfig, EASIntervener, wrap_model_with_eas
from utils import get_device, MODEL_REGISTRY


def calculate_perplexity(model, tokenizer, texts, device, max_length=128):
    """Calculate perplexity on a list of texts."""
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(texts, desc="Calculating perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=max_length).to(device)
        
        if inputs.input_ids.shape[1] < 2:
            continue
            
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss.item()
            num_tokens = inputs.input_ids.shape[1] - 1
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = np.exp(avg_loss)
    return perplexity


def calculate_next_token_accuracy(model, tokenizer, texts, device, intervener=None):
    """
    For each text, predict next token at each position.
    Return accuracy of top-1 predictions.
    """
    correct = 0
    total = 0
    
    for text in tqdm(texts, desc="Calculating next-token accuracy"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=128).to(device)
        
        if inputs.input_ids.shape[1] < 2:
            continue
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            logits = outputs.logits
            
            # Predict token at position i given tokens 0..i-1
            predictions = logits[:, :-1, :].argmax(dim=-1)
            targets = inputs.input_ids[:, 1:]
            
            matches = (predictions == targets).sum().item()
            correct += matches
            total += targets.numel()
        
        # Update EAS if using it
        if intervener is not None:
            intervener.record_sample()
            # For language modeling, treat low-loss samples as "correct"
            loss_val = outputs.loss.item() if outputs.loss is not None else 5.0
            if loss_val < 3.0:  # Threshold for "good" prediction
                intervener.update_on_success()
    
    return correct / total if total > 0 else 0


def test_eas_on_language_modeling(
    model_key: str = "pythia-70m",
    num_samples: int = 100,
    warmup_samples: int = 30
):
    """Test whether EAS improves language modeling metrics."""
    device = get_device()
    model_info = MODEL_REGISTRY[model_key]
    
    print(f"\n{'='*60}")
    print(f"EAS Language Modeling Test: {model_key}")
    print(f"{'='*60}")
    
    # Load text data (WikiText for language modeling)
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in dataset["text"] if len(t.strip()) > 50][:num_samples]
    print(f"Using {len(texts)} text samples")
    
    # Split into warmup and test
    warmup_texts = texts[:warmup_samples]
    test_texts = texts[warmup_samples:]
    
    # --- BASELINE ---
    print("\n[1/2] Testing BASELINE (no EAS)...")
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    baseline_ppl = calculate_perplexity(model, tokenizer, test_texts, device)
    baseline_acc = calculate_next_token_accuracy(model, tokenizer, test_texts, device)
    
    print(f"  Baseline Perplexity: {baseline_ppl:.2f}")
    print(f"  Baseline Next-Token Accuracy: {baseline_acc:.2%}")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- WITH EAS ---
    print("\n[2/2] Testing WITH EAS...")
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    config = EASConfig(
        hidden_dim=model_info["hidden_dim"],
        num_attractors=10,
        base_alpha=0.3,
        warmup_samples=warmup_samples
    )
    model, intervener = wrap_model_with_eas(model, model_info["hidden_dim"], config=config)
    intervener.to(device)
    
    # Warmup phase: let EAS learn from successful predictions
    print(f"  Warming up EAS on {len(warmup_texts)} samples...")
    for text in tqdm(warmup_texts, desc="EAS warmup"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
        intervener.record_sample()
        if outputs.loss.item() < 3.0:  # Learn from good predictions
            intervener.update_on_success()
    
    print(f"  EAS warmup complete. Interventions ready: {intervener.intervention_count}")
    
    # Test with EAS
    eas_ppl = calculate_perplexity(model, tokenizer, test_texts, device)
    eas_acc = calculate_next_token_accuracy(model, tokenizer, test_texts, device, intervener)
    
    print(f"  EAS Perplexity: {eas_ppl:.2f}")
    print(f"  EAS Next-Token Accuracy: {eas_acc:.2%}")
    
    # --- RESULTS ---
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':<15} {'EAS':<15} {'Change':<15}")
    print("-" * 60)
    
    ppl_change = ((eas_ppl - baseline_ppl) / baseline_ppl) * 100
    acc_change = (eas_acc - baseline_acc) * 100
    
    ppl_emoji = "✅" if eas_ppl < baseline_ppl else "❌"
    acc_emoji = "✅" if eas_acc > baseline_acc else "❌"
    
    print(f"{'Perplexity':<25} {baseline_ppl:<15.2f} {eas_ppl:<15.2f} {ppl_emoji} {ppl_change:+.1f}%")
    print(f"{'Next-Token Accuracy':<25} {baseline_acc:<15.2%} {eas_acc:<15.2%} {acc_emoji} {acc_change:+.1f}pp")
    
    print(f"\nEAS Stats: {intervener.get_stats()}")
    
    if eas_ppl < baseline_ppl or eas_acc > baseline_acc:
        print("\n🎉 EAS shows IMPROVEMENT on language modeling!")
    else:
        print("\n⚠️ No improvement detected")
    
    return {
        "baseline_ppl": baseline_ppl,
        "eas_ppl": eas_ppl,
        "baseline_acc": baseline_acc,
        "eas_acc": eas_acc
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-70m")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()
    
    results = test_eas_on_language_modeling(args.model, args.samples)
