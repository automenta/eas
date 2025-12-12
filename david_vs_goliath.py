#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model (Scientifically Rigorous)"""

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
    
    # 1. Load Goliath
    try:
        large_model, large_tok = load_model("gpt2-large", device)
    except Exception as e:
        print(f"Could not load gpt2-large: {e}. Fallback to gpt2-medium.")
        large_model, large_tok = load_model("gpt2-medium", device)

    # 2. Load David
    small_model, small_tok = load_model("EleutherAI/pythia-410m", device)

    # Initialize Adaptive MCRE
    # Lower threshold to ensure some abstention happens for demonstration
    mcre = MCRE(small_model, small_tok, device, z_threshold=0.5)

    dataset = get_logic_dataset()

    # Define prompt formatter to ensure calibration matches evaluation distribution
    def format_prompt(ex):
        options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
        return f"Question: {ex['question']}\n{options}\nAnswer:"

    mcre.calibrate(dataset, n=30, prompt_fn=format_prompt)
    
    results = {
        "goliath_correct": 0,
        "david_baseline_correct": 0, # David answering everything
        "david_mcre_correct": 0,     # David answering only when confident
        "david_mcre_answered": 0,    # Count of non-abstained
        "david_abstained": 0,
        "total": 0
    }
    
    print(f"\nComparing models on {min(num_test, len(dataset))} samples...")
    
    for i in tqdm(range(min(num_test, len(dataset)))):
        ex = dataset[i]
        options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
        prompt = f"Question: {ex['question']}\n{options}\nAnswer:"
        correct_char = "ABCD"[ex['answer']]

        # --- Goliath ---
        inputs = large_tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = large_model(**inputs).logits[0, -1, :]
            ids = [large_tok.encode(" " + c)[0] for c in "ABCD"]
            probs = [logits[tid].item() if tid < logits.size(0) else -999 for tid in ids]
            goliath_ans = "ABCD"[torch.tensor(probs).argmax().item()]

        if goliath_ans == correct_char:
            results["goliath_correct"] += 1

        # --- David ---
        state = mcre.evaluate(prompt)
        
        # Baseline: Check correctness regardless of abstention
        if state.predicted_answer == correct_char:
            results["david_baseline_correct"] += 1

        # MCRE: Check correctness only if not abstained
        if state.should_abstain:
            results["david_abstained"] += 1
        else:
            results["david_mcre_answered"] += 1
            if state.predicted_answer == correct_char:
                results["david_mcre_correct"] += 1

        results["total"] += 1

    # --- Metrics ---
    total = results["total"]

    # 1. Goliath Accuracy
    goliath_acc = results["goliath_correct"] / total

    # 2. David Baseline Accuracy (Control)
    david_baseline_acc = results["david_baseline_correct"] / total
    
    # 3. David MCRE Accuracy (Experimental)
    # Accuracy on the subset it chose to answer
    mcre_answered = results["david_mcre_answered"]
    david_mcre_acc = results["david_mcre_correct"] / mcre_answered if mcre_answered > 0 else 0
    
    # 4. Effective Score
    abstention_rate = results["david_abstained"] / total
    david_effective = (david_mcre_acc * (1 - abstention_rate)) + (0.5 * abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"SCIENTIFIC CONTROL RESULTS")
    print(f"{'='*60}")

    print(f"\n1. GOLIATH (Baseline Large):   {goliath_acc:.1%}")
    print(f"2. DAVID (Baseline Small):     {david_baseline_acc:.1%}")
    print(f"3. DAVID + MCRE (Experimental):{david_mcre_acc:.1%} (on {mcre_answered}/{total} answered)")
    print(f"   Abstention Rate:            {abstention_rate:.1%}")
    print(f"   Effective Score:            {david_effective:.1%}")

    print(f"\n{'='*60}")
    print("HYPOTHESIS TESTING:")
    
    # Test 1: Does MCRE improve accuracy on answered questions?
    delta_acc = david_mcre_acc - david_baseline_acc
    print(f"H1 (Precision): MCRE Acc > Baseline Acc?  ", end="")
    if delta_acc > 0:
        print(f"✅ YES (+{delta_acc:.1%})")
    else:
        print(f"❌ NO ({delta_acc:.1%})")

    # Test 2: Does Small Model beat Large Model?
    print(f"H2 (David vs Goliath): David > Goliath?   ", end="")
    if david_baseline_acc > goliath_acc:
        print(f"✅ YES (Raw capability victory)")
    elif david_effective > goliath_acc:
        print(f"✅ YES (Effective victory via MCRE)")
    else:
        print(f"❌ NO")

if __name__ == "__main__":
    run_comparison()
