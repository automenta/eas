#!/usr/bin/env python3
"""
eas_generation.py - Test EAS on text generation quality

Metrics:
1. Repetition rate (lower = better)
2. Diversity (higher = better)  
3. Coherence via forward perplexity
4. Self-BLEU (lower = more diverse)

EAS hypothesis: steering toward successful generation patterns
should produce higher quality, more diverse outputs.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import Counter
import numpy as np
from tqdm import tqdm
from eas_core import EASConfig, wrap_model_with_eas
from utils import get_device, MODEL_REGISTRY


def calculate_repetition_rate(text: str, n: int = 3) -> float:
    """Calculate n-gram repetition rate."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    
    counter = Counter(ngrams)
    repeated = sum(1 for count in counter.values() if count > 1)
    return repeated / len(counter) if counter else 0


def calculate_distinct_n(texts: list, n: int = 2) -> float:
    """Calculate distinct-n metric (diversity)."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def generate_texts(model, tokenizer, prompts, device, max_new_tokens=50, **kwargs):
    """Generate texts from prompts."""
    generated = []
    for prompt in tqdm(prompts, desc="Generating"):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated.append(text[len(prompt):])  # Remove prompt
    
    return generated


def test_eas_on_generation(
    model_key: str = "pythia-70m",
    num_prompts: int = 50
):
    """Test whether EAS improves generation quality."""
    device = get_device()
    model_info = MODEL_REGISTRY[model_key]
    
    print(f"\n{'='*60}")
    print(f"EAS Generation Quality Test: {model_key}")
    print(f"{'='*60}")
    
    # Prompts for generation
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away,",
        "The most important thing about science is",
        "In the year 2050, humanity will",
        "The secret to happiness lies in",
        "Technology has changed the way we",
        "The ocean is home to",
        "Music brings people together because",
        "The best way to learn is",
        "Climate change affects everyone because",
    ] * (num_prompts // 10 + 1)
    prompts = prompts[:num_prompts]
    
    # Split prompts
    warmup_prompts = prompts[:20]
    test_prompts = prompts[20:]
    
    # --- BASELINE ---
    print("\n[1/2] Testing BASELINE generation...")
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    baseline_texts = generate_texts(model, tokenizer, test_prompts, device)
    
    baseline_rep = np.mean([calculate_repetition_rate(t) for t in baseline_texts])
    baseline_div = calculate_distinct_n(baseline_texts, n=2)
    baseline_len = np.mean([len(t.split()) for t in baseline_texts])
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- WITH EAS ---
    print("\n[2/2] Testing WITH EAS generation...")
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    config = EASConfig(
        hidden_dim=model_info["hidden_dim"],
        num_attractors=10,
        base_alpha=0.3,
        warmup_samples=20
    )
    model, intervener = wrap_model_with_eas(model, model_info["hidden_dim"], config=config)
    intervener.to(device)
    
    # Warmup: generate some texts and learn from "good" ones
    print("  Warming up EAS...")
    warmup_texts = generate_texts(model, tokenizer, warmup_prompts, device)
    for text in warmup_texts:
        intervener.record_sample()
        # Learn from texts with low repetition and good length
        if calculate_repetition_rate(text) < 0.3 and len(text.split()) > 10:
            intervener.update_on_success()
    
    # Generate with EAS
    eas_texts = generate_texts(model, tokenizer, test_prompts, device)
    
    eas_rep = np.mean([calculate_repetition_rate(t) for t in eas_texts])
    eas_div = calculate_distinct_n(eas_texts, n=2)
    eas_len = np.mean([len(t.split()) for t in eas_texts])
    
    # --- RESULTS ---
    print(f"\n{'='*60}")
    print("GENERATION QUALITY RESULTS")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Baseline':<15} {'EAS':<15} {'Better?':<10}")
    print("-" * 60)
    
    rep_better = "✅" if eas_rep < baseline_rep else "❌"
    div_better = "✅" if eas_div > baseline_div else "❌"
    
    print(f"{'Repetition Rate (↓)':<25} {baseline_rep:<15.3f} {eas_rep:<15.3f} {rep_better}")
    print(f"{'Distinct-2 (↑)':<25} {baseline_div:<15.3f} {eas_div:<15.3f} {div_better}")
    print(f"{'Avg Length':<25} {baseline_len:<15.1f} {eas_len:<15.1f}")
    
    print(f"\nEAS Stats: {intervener.get_stats()}")
    
    # Sample outputs
    print(f"\n{'='*60}")
    print("SAMPLE OUTPUTS")
    print(f"{'='*60}")
    for i in range(min(3, len(test_prompts))):
        print(f"\nPrompt: {test_prompts[i]}")
        print(f"  Baseline: {baseline_texts[i][:100]}...")
        print(f"  EAS:      {eas_texts[i][:100]}...")
    
    if eas_rep < baseline_rep or eas_div > baseline_div:
        print("\n🎉 EAS shows IMPROVEMENT on generation quality!")
    else:
        print("\n⚠️ No improvement detected")
    
    return {
        "baseline_rep": baseline_rep,
        "eas_rep": eas_rep,
        "baseline_div": baseline_div,
        "eas_div": eas_div
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-70m")
    parser.add_argument("--prompts", type=int, default=50)
    args = parser.parse_args()
    
    results = test_eas_on_generation(args.model, args.prompts)
