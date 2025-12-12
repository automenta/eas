#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model (Scientifically Rigorous)"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from local_dataset import get_logic_dataset
from selective_abstention_demo import MCRE
import gc

def load_model(name, device):
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tokenizer

def run_multi_model_comparison(num_test=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = get_logic_dataset()
    num_test = min(num_test, len(dataset))

    # --- 1. Evaluate Goliath (Reference) ---
    print("\n" + "="*60)
    print("PHASE 1: GOLIATH (Baseline)")
    print("="*60)
    
    try:
        goliath_model, goliath_tok = load_model("gpt2-large", device)
    except Exception as e:
        print(f"Could not load gpt2-large: {e}. Fallback to gpt2-medium.")
        goliath_model, goliath_tok = load_model("gpt2-medium", device)

    goliath_results = []
    
    print(f"Evaluating Goliath on {num_test} samples...")
    for i in range(num_test):
        ex = dataset[i]
        options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
        prompt = f"Question: {ex['question']}\n{options}\nAnswer:"
        correct_char = "ABCD"[ex['answer']]

        inputs = goliath_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = goliath_model(**inputs).logits[0, -1, :]
            ids = [goliath_tok.encode(" " + c)[0] for c in "ABCD"]
            # Fallback if tokenization differs (usually gpt2 has leading space)
            if len(ids) != 4:
                ids = [goliath_tok.encode(c)[0] for c in "ABCD"]

            probs = [logits[tid].item() if tid < logits.size(0) else -999 for tid in ids]
            goliath_ans = "ABCD"[torch.tensor(probs).argmax().item()]

        goliath_results.append(goliath_ans == correct_char)

    goliath_acc = sum(goliath_results) / num_test
    print(f"Goliath Accuracy: {goliath_acc:.1%}")

    # Clear Goliath from memory to make space for Davids
    del goliath_model
    del goliath_tok
    torch.cuda.empty_cache()
    gc.collect()

    # --- 2. Evaluate Davids ---
    davids = [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-410m"
    ]

    summary = []

    for david_name in davids:
        print("\n" + "="*60)
        print(f"PHASE 2: DAVID ({david_name})")
        print("="*60)

        david_model, david_tok = load_model(david_name, device)

        # Initialize MCRE
        mcre = MCRE(david_model, david_tok, device, z_threshold=0.5)

        # Calibrate with correct prompt format
        def format_prompt(ex):
            options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
            return f"Question: {ex['question']}\n{options}\nAnswer:"

        mcre.calibrate(dataset, n=30, prompt_fn=format_prompt)

        stats = {
            "correct": 0,
            "mcre_correct": 0,
            "mcre_answered": 0,
            "abstained": 0
        }

        for i in tqdm(range(num_test)):
            ex = dataset[i]
            prompt = format_prompt(ex)
            correct_char = "ABCD"[ex['answer']]

            state = mcre.evaluate(prompt)

            # Baseline
            if state.predicted_answer == correct_char:
                stats["correct"] += 1

            # MCRE
            if state.should_abstain:
                stats["abstained"] += 1
            else:
                stats["mcre_answered"] += 1
                if state.predicted_answer == correct_char:
                    stats["mcre_correct"] += 1

        # Calculate Metrics
        base_acc = stats["correct"] / num_test

        if stats["mcre_answered"] > 0:
            mcre_acc = stats["mcre_correct"] / stats["mcre_answered"]
        else:
            mcre_acc = 0.0

        abst_rate = stats["abstained"] / num_test

        # Store for summary
        summary.append({
            "model": david_name,
            "base_acc": base_acc,
            "mcre_acc": mcre_acc,
            "abst_rate": abst_rate,
            "answered_count": stats["mcre_answered"]
        })

        print(f"Baseline Acc: {base_acc:.1%}")
        print(f"MCRE Acc:     {mcre_acc:.1%}")
        print(f"Abstention:   {abst_rate:.1%}")

        # Cleanup
        del david_model
        del david_tok
        torch.cuda.empty_cache()
        gc.collect()

    # --- 3. Final Report ---
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT: DAVIDS vs GOLIATH")
    print("="*80)
    print(f"GOLIATH (Reference): {goliath_acc:.1%} Accuracy")
    print("-" * 80)
    print(f"{'Model':<25} | {'Base Acc':<10} | {'MCRE Acc':<10} | {'Abst %':<8} | {'vs Goliath':<10}")
    print("-" * 80)
    
    for row in summary:
        base = row['base_acc']
        mcre = row['mcre_acc']
        abst = row['abst_rate']

        # Did we beat Goliath?
        if base > goliath_acc:
            vs_g = "WIN (Raw)"
        elif mcre > goliath_acc:
            vs_g = "WIN (MCRE)"
        else:
            vs_g = "LOSE"

        print(f"{row['model']:<25} | {base:<10.1%} | {mcre:<10.1%} | {abst:<8.1%} | {vs_g:<10}")

    print("-" * 80)

    # Return success if ANY model beats Goliath via MCRE or Raw
    wins = sum(1 for row in summary if row['mcre_acc'] > goliath_acc or row['base_acc'] > goliath_acc)
    return wins > 0

if __name__ == "__main__":
    run_multi_model_comparison()
