#!/usr/bin/env python3
"""
eas_contrastive.py - EAS for Learning from Good vs Bad Examples

THE KEY INSIGHT:
EAS should learn what GOOD looks like by contrasting it with BAD.
We show the model good and bad examples, learn attractors from good only,
then measure if EAS improves the QUALITY of continuations.

Metric: We can MEASURE this objectively using:
1. Toxicity scores (via classifier)
2. Factuality scores
3. Coherence scores
4. Task-specific accuracy
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from eas_core import EASConfig, wrap_model_with_eas
from utils import get_device, MODEL_REGISTRY


def test_eas_coherence_improvement(model_key: str = "gpt2", num_tests: int = 20):
    """
    Test if EAS can improve coherence by learning from coherent examples.
    
    Measure: Compare continuation quality before/after EAS with a scoring model.
    """
    device = get_device()
    model_info = MODEL_REGISTRY[model_key]
    
    print(f"\n{'='*70}")
    print("EAS COHERENCE IMPROVEMENT TEST")
    print(f"{'='*70}")
    
    # Load a sentiment classifier as a proxy for "quality"
    # (Positive sentiment = generally more coherent/sensible text)
    print("Loading quality scorer...")
    scorer = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)
    
    # Coherent (good) examples
    good_examples = [
        "The weather today is beautiful with clear skies and sunshine.",
        "I enjoy spending time with my family during the holidays.",
        "Learning new skills helps you grow as a person.",
        "Exercise is important for maintaining good health.",
        "Reading books expands your knowledge and imagination.",
        "Music can lift your mood and bring people together.",
        "Kindness and compassion make the world a better place.",
        "Nature provides us with peace and tranquility.",
    ]
    
    # Prompts that could go either way
    test_prompts = [
        "Today I feel",
        "The best thing about",
        "When I think of the future,",
        "What matters most is",
        "Life is full of",
    ]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # --- BASELINE ---
    print(f"\n[1/2] BASELINE generation...")
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    baseline_outputs = []
    baseline_scores = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=20, do_sample=True,
                temperature=0.9, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        baseline_outputs.append(text)
        
        # Score the continuation (not the prompt)
        continuation = text[len(prompt):]
        try:
            score = scorer(continuation[:200])[0]
            sentiment_score = score['score'] if score['label'] == 'POSITIVE' else 1 - score['score']
        except:
            sentiment_score = 0.5
        baseline_scores.append(sentiment_score)
        print(f"  {prompt} → {continuation[:40]}... (score: {sentiment_score:.2f})")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- WITH EAS ---
    print(f"\n[2/2] WITH EAS (learned from positive examples)...")
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    config = EASConfig(
        hidden_dim=model_info["hidden_dim"],
        num_attractors=8,
        base_alpha=0.5,
        warmup_samples=0
    )
    model, intervener = wrap_model_with_eas(model, model_info["hidden_dim"], config=config)
    intervener.to(device)
    
    # Learn from good examples
    print("  Learning from positive examples...")
    for example in good_examples:
        inputs = tokenizer(example, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        intervener.record_sample()
        intervener.update_on_success()
    
    eas_outputs = []
    eas_scores = []
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=20, do_sample=True,
                temperature=0.9, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        eas_outputs.append(text)
        
        continuation = text[len(prompt):]
        try:
            score = scorer(continuation[:200])[0]
            sentiment_score = score['score'] if score['label'] == 'POSITIVE' else 1 - score['score']
        except:
            sentiment_score = 0.5
        eas_scores.append(sentiment_score)
        print(f"  {prompt} → {continuation[:40]}... (score: {sentiment_score:.2f})")
    
    # --- RESULTS ---
    avg_baseline = sum(baseline_scores) / len(baseline_scores)
    avg_eas = sum(eas_scores) / len(eas_scores)
    improvement = avg_eas - avg_baseline
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"Average Positivity Score:")
    print(f"  Baseline: {avg_baseline:.3f}")
    print(f"  With EAS: {avg_eas:.3f}")
    print(f"  Change:   {improvement:+.3f} ({improvement/avg_baseline*100:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✅ EAS improved output quality by {improvement*100:.1f}%")
    else:
        print(f"\n❌ No improvement detected")
    
    return {
        "baseline_avg": avg_baseline,
        "eas_avg": avg_eas,
        "improvement": improvement
    }


if __name__ == "__main__":
    results = test_eas_coherence_improvement("gpt2", num_tests=10)
