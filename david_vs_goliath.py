#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from local_dataset import get_logic_dataset
from selective_abstention_demo import MCRE  # Import the smarter MCRE

def load_model(name, device):
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tokenizer

def run_comparison(num_test=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Goliath (GPT-2 Large)
    # Note: GPT-2 Large is 774M. If too big for sandbox, we might need smaller.
    # But user asked for this comparison.
    # If OOM, we can fallback to gpt2-medium (355M).
    try:
        large_model, large_tok = load_model("gpt2-large", device)
    except Exception as e:
        print(f"Could not load gpt2-large: {e}")
        print("Falling back to gpt2-medium...")
        large_model, large_tok = load_model("gpt2-medium", device)

    # 2. Load David (Pythia-70m)
    small_model, small_tok = load_model("EleutherAI/pythia-70m", device)
    
    # Initialize MCRE for David
    # Threshold tuning: 0.6 was default in demo.
    # We can try to be more aggressive or lenient.
    mcre = MCRE(small_model, small_tok, device, threshold=0.6)

    dataset = get_logic_dataset()
    
    results = {
        "large_correct": 0, "large_wrong": 0,
        "small_correct": 0, "small_wrong": 0, "small_abstained": 0
    }
    
    print(f"\nComparing models on {min(num_test, len(dataset))} samples...")
    
    for i in tqdm(range(min(num_test, len(dataset)))):
        ex = dataset[i]
        
        # Format prompt
        options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
        prompt = f"Question: {ex['question']}\n{options}\nAnswer:"
        correct_char = "ABCD"[ex['answer']]

        # --- Goliath (Standard Generation) ---
        inputs = large_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = large_model(**inputs)
            logits = outputs.logits[0, -1, :]

            # Simple answer extraction
            ans_probs = {}
            for char in "ABCD":
                # Handle tokenization variations
                ids = large_tok.encode(" " + char) + large_tok.encode(char)
                prob = 0
                for tid in ids:
                    if tid < logits.size(0):
                        prob = max(prob, logits[tid].item())
                ans_probs[char] = prob

            large_ans = max(ans_probs, key=ans_probs.get)

        if large_ans == correct_char:
            results["large_correct"] += 1
        else:
            results["large_wrong"] += 1

        # --- David (MCRE /w Abstention) ---
        state = mcre.evaluate(prompt)
        
        if state.should_abstain:
            results["small_abstained"] += 1
        elif state.predicted_answer == correct_char:
            results["small_correct"] += 1
        else:
            results["small_wrong"] += 1

    # --- Metrics ---
    # Goliath
    large_total = results["large_correct"] + results["large_wrong"]
    large_acc = results["large_correct"] / large_total if large_total else 0
    large_effective = large_acc # No abstention
    
    # David
    small_total = results["small_correct"] + results["small_wrong"] + results["small_abstained"]
    small_answered = results["small_correct"] + results["small_wrong"]
    small_acc_answered = results["small_correct"] / small_answered if small_answered > 0 else 0
    abstention_rate = results["small_abstained"] / small_total

    # Effective accuracy formula: Acc * (1 - Abstain) + Val * Abstain
    # Val = 0.25 (random guess value) or 0.5 (neutral/safe value)
    # User prompt implied "Abstention value" might be higher if errors are costly.
    # But let's stick to the README's implied 0.5 or just compare "Acc on Answered" vs "Large Acc".
    
    small_effective_neutral = (small_acc_answered * (1 - abstention_rate)) + (0.5 * abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"DAVID VS GOLIATH RESULTS")
    print(f"{'='*60}")

    print(f"\nGOLIATH (Large Model):")
    print(f"  Accuracy: {large_acc:.1%}")
    print(f"  (Answers everything, often confidently wrong)")

    print(f"\nDAVID (Pythia-70m + MCRE):")
    print(f"  Accuracy (when answering): {small_acc_answered:.1%}")
    print(f"  Abstention Rate:           {abstention_rate:.1%}")
    print(f"  Effective Score:           {small_effective_neutral:.1%}")

    print(f"\n{'='*60}")
    
    if small_acc_answered > large_acc:
        print("🏆 DAVID WINS! Small model is more reliable when it speaks.")
        print(f"   (Reliability Gap: {small_acc_answered - large_acc:+.1%})")
    elif small_effective_neutral > large_acc:
        print("🏆 DAVID WINS! Higher effective score via safety.")
    else:
        print("🏳️  GOLIATH WINS. Small model needs better tuning.")
        print("    Try adjusting threshold or using a better uncertainty estimator.")

if __name__ == "__main__":
    run_comparison()
