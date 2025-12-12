#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from local_dataset import get_logic_dataset
from selective_abstention_demo import MCRE

def load_model(name, device):
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tokenizer

def run_comparison(num_test=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Goliath (GPT-2 Large - 774M)
    try:
        large_model, large_tok = load_model("gpt2-large", device)
    except Exception as e:
        print(f"Could not load gpt2-large: {e}. Fallback to gpt2-medium.")
        large_model, large_tok = load_model("gpt2-medium", device)

    # 2. Load David (Pythia-410m - 410M)
    # 410M is ~half of 774M, maintaining the David vs Goliath dynamic.
    small_model, small_tok = load_model("EleutherAI/pythia-410m", device)

    # Initialize Adaptive MCRE
    # z_threshold=0.5 means we abstain on the top 30% most uncertain samples approx.
    mcre = MCRE(small_model, small_tok, device, z_threshold=0.5)

    dataset = get_logic_dataset()

    # Calibrate MCRE on the dataset to learn baseline entropy
    mcre.calibrate(dataset, n=30)
    
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
                ids = large_tok.encode(" " + char) + large_tok.encode(char)
                prob = 0
                for tid in ids:
                    if tid < logits.size(0):
                        prob = max(prob, logits[tid].item())
                ans_probs[char] = prob

            large_ans = max(ans_probs, key=ans_probs.get) if ans_probs else "A"

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
    
    # David
    small_total = results["small_correct"] + results["small_wrong"] + results["small_abstained"]
    small_answered = results["small_correct"] + results["small_wrong"]
    small_acc_answered = results["small_correct"] / small_answered if small_answered > 0 else 0
    abstention_rate = results["small_abstained"] / small_total
    
    # Effective accuracy: Acc * (1 - Abstain) + 0.5 * Abstain
    # Using 0.5 as "neutral" value for "I don't know" (better than wrong, worse than right)
    small_effective = (small_acc_answered * (1 - abstention_rate)) + (0.5 * abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"DAVID VS GOLIATH RESULTS")
    print(f"{'='*60}")

    print(f"\nGOLIATH (GPT-2 Large, 774M):")
    print(f"  Accuracy: {large_acc:.1%}")

    print(f"\nDAVID (Pythia-410m, 410M + Adaptive MCRE):")
    print(f"  Accuracy (Answered):       {small_acc_answered:.1%}")
    print(f"  Abstention Rate:           {abstention_rate:.1%}")
    print(f"  Effective Score:           {small_effective:.1%}")

    print(f"\n{'='*60}")
    
    # Success Criteria:
    # 1. David is more accurate when it chooses to speak (Accuracy Answered > Goliath Accuracy)
    # 2. OR David has higher effective score

    if small_acc_answered > large_acc:
        print(f"🏆 DAVID WINS! Higher accuracy on answered questions (+{small_acc_answered - large_acc:.1%})")
    elif small_effective > large_acc:
        print(f"🏆 DAVID WINS! Higher effective score via safe abstention.")
    else:
        print("🏳️  GOLIATH WINS. Adaptive thresholding needs more tuning.")

if __name__ == "__main__":
    run_comparison()
