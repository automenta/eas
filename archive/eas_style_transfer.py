#!/usr/bin/env python3
"""
eas_style_transfer.py - EAS Style Transfer with Neutral Prompts

Uses NEUTRAL prompts so baseline has no style bias.
Then shows EAS can inject specific style from examples.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eas_core import EASConfig, wrap_model_with_eas
from utils import get_device, MODEL_REGISTRY


# Style examples
STYLES = {
    "shakespeare": [
        "Hark! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        "To thine own self be true, and it must follow, as the night the day, thou canst not then be false to any man.",
        "All the world's a stage, and all the men and women merely players; they have their exits and their entrances.",
        "The quality of mercy is not strained; it droppeth as the gentle rain from heaven upon the place beneath.",
        "We are such stuff as dreams are made on, and our little life is rounded with a sleep.",
    ],
    "hemingway": [
        "The old man was thin and gaunt with deep wrinkles in the back of his neck.",
        "He was an old man who fished alone in a skiff in the Gulf Stream.",
        "The sun rose thinly from the sea and the old man could see the other boats.",
        "But man is not made for defeat. A man can be destroyed but not defeated.",
        "There is nothing noble in being superior to your fellow man; true nobility is being superior to your former self.",
    ],
    "news": [
        "WASHINGTON (AP) — Officials announced today that the new policy will take effect immediately.",
        "The spokesperson declined to comment on the ongoing investigation.",
        "Markets reacted sharply to the announcement, with the index falling 2.3% in early trading.",
        "Sources familiar with the matter said negotiations are expected to continue through the weekend.",
        "The company released a statement saying it would cooperate fully with authorities.",
    ],
    "code_comment": [
        "# Initialize the connection pool with max 10 concurrent connections",
        "// TODO: Refactor this function to handle edge cases properly",
        "/* This algorithm runs in O(n log n) time complexity */",
        "# WARNING: This modifies the input array in place",
        "// Returns null if the key is not found in the cache",
    ],
}

# Neutral prompts with no inherent style
NEUTRAL_PROMPTS = [
    "The man walked into the room and",
    "When the sun rose,",
    "The problem with this approach is",
    "After careful consideration,",
    "In the end,",
]


def style_transfer_demo(model_key: str = "pythia-70m", style: str = "shakespeare"):
    """Show EAS style transfer from neutral prompts."""
    device = get_device()
    model_info = MODEL_REGISTRY[model_key]
    
    print(f"\n{'='*70}")
    print(f"EAS STYLE TRANSFER: → {style.upper()}")
    print(f"Using NEUTRAL prompts (no inherent style)")
    print(f"{'='*70}")
    
    examples = STYLES[style]
    print(f"\nStyle examples:")
    for ex in examples[:3]:
        print(f"  • {ex[:70]}...")
    
    # --- BASELINE ---
    print(f"\n{'='*70}")
    print("BASELINE (no style adaptation)")
    print(f"{'='*70}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    
    baseline_outputs = []
    for prompt in NEUTRAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=30, do_sample=True,
                temperature=0.9, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        baseline_outputs.append(text)
        print(f"  {prompt} → {text[len(prompt):50]}...")
    
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # --- WITH EAS STYLE ---
    print(f"\n{'='*70}")
    print(f"WITH EAS → {style.upper()} STYLE")
    print(f"{'='*70}")
    
    model = AutoModelForCausalLM.from_pretrained(model_info["name"]).to(device).eval()
    config = EASConfig(
        hidden_dim=model_info["hidden_dim"],
        num_attractors=5,
        base_alpha=0.7,  # Strong style steering
        warmup_samples=0
    )
    model, intervener = wrap_model_with_eas(model, model_info["hidden_dim"], config=config)
    intervener.to(device)
    
    # Learn style
    print("\n  Learning style patterns...")
    for example in examples:
        inputs = tokenizer(example, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)
        intervener.record_sample()
        intervener.update_on_success()
    print(f"  Learned from {len(examples)} examples")
    
    # Generate with style
    adapted_outputs = []
    for prompt in NEUTRAL_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=30, do_sample=True,
                temperature=0.9, pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        adapted_outputs.append(text)
        print(f"  {prompt} → {text[len(prompt):50]}...")
    
    # --- COMPARISON ---
    print(f"\n{'='*70}")
    print(f"COMPARISON: Neutral prompt → {style} style?")
    print(f"{'='*70}")
    for i, prompt in enumerate(NEUTRAL_PROMPTS):
        print(f"\n📝 \"{prompt}\"")
        print(f"   BASELINE: {baseline_outputs[i][len(prompt):][:50]}...")
        print(f"   {style.upper()}: {adapted_outputs[i][len(prompt):][:50]}...")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="pythia-70m")
    parser.add_argument("--style", default="shakespeare", choices=list(STYLES.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all:
        for style in STYLES:
            style_transfer_demo(args.model, style)
    else:
        style_transfer_demo(args.model, args.style)
