#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model"""

import torch
import re
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def load_model(name, device):
    print(f"Loading {name}...")
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return model, tokenizer

def get_answer_confidence(model, tokenizer, prompt, device):
    """Get model's answer and confidence via log probability."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        
        # Check confidence in answer tokens (A, B, C, D)
        # Note: Some tokenizers might need a space prefix
        answer_tokens = []
        for c in "ABCD":
            t1 = tokenizer.encode(f" {c}", add_special_tokens=False)
            t2 = tokenizer.encode(c, add_special_tokens=False)
            # Prefer the token that is a single token if possible, or the one with space
            # For GPT2/Pythia, " A" is often a different token than "A"
            # We'll sum probs of " A" and "A" just to be safe or pick the one that makes sense
            # Here we just pick " A" as is common for completion
            if len(t1) == 1:
                answer_tokens.append(t1[0])
            elif len(t2) == 1:
                answer_tokens.append(t2[0])
            else:
                answer_tokens.append(t1[0]) # Fallback

        answer_probs = [probs[t].item() for t in answer_tokens]
        
        best_idx = max(range(4), key=lambda i: answer_probs[i])
        confidence = answer_probs[best_idx]
        answer = "ABCD"[best_idx]
    
    return answer, confidence

def parse_heka_example(ex):
    """
    Parses heka-ai/logiqa format:
    input: "... question <!--input--> Question Text ['Opt A', 'Opt B', ...] <!--/input-->"
    expected_output: "Opt A"
    """
    input_text = ex['input']
    expected = ex['expected_output']

    # Extract context and question
    # This dataset seems to mix context and question in a complex way
    # Simplification: Extract the list of options

    # Find options list
    match = re.search(r"(\['.*'\])", input_text, re.DOTALL)
    if not match:
        return None

    options_str = match.group(1)
    try:
        options = ast.literal_eval(options_str)
    except:
        return None

    # Extract the question (text before the options)
    # The format is ... question <!--input--> Question text ['Opt...
    # It's messy. Let's try to grab everything before the list but after <!--input-->

    question_match = re.search(r"<!--input-->\s*(.*?)\s*\['", input_text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    else:
        # Fallback: just take everything before the options
        question = input_text.split("['")[0].strip()

    # Find correct answer index
    correct_idx = -1
    for i, opt in enumerate(options):
        if opt in expected or expected in opt:
            correct_idx = i
            break

    if correct_idx == -1:
        return None

    return {
        'question': question,
        'options': options,
        'answer': correct_idx
    }

def run_comparison(num_test=50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    small_model_name = "EleutherAI/pythia-70m"
    large_model_name = "gpt2-large"
    
    small_model, small_tok = load_model(small_model_name, device)
    large_model, large_tok = load_model(large_model_name, device)

    # Load dataset
    print("Loading heka-ai/logiqa...")
    ds = load_dataset("heka-ai/logiqa", split="train") # Only train exists
    
    results = {
        "small_correct": 0, "small_wrong": 0, "small_abstained": 0,
        "large_correct": 0, "large_wrong": 0
    }
    
    abstention_threshold = 0.30 # Slightly lower threshold for 70m

    count = 0
    pbar = tqdm(total=num_test)
    
    for ex in ds:
        if count >= num_test:
            break

        parsed = parse_heka_example(ex)
        if not parsed:
            continue

        count += 1
        pbar.update(1)
        
        # Format prompt
        options_text = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(parsed['options'])])
        prompt = f"Question: {parsed['question']}\n{options_text}\nAnswer:"

        correct_char = "ABCD"[parsed['answer']]

        # Large model
        large_ans, large_conf = get_answer_confidence(large_model, large_tok, prompt, device)
        if large_ans == correct_char:
            results["large_correct"] += 1
        else:
            results["large_wrong"] += 1

        # Small model
        small_ans, small_conf = get_answer_confidence(small_model, small_tok, prompt, device)
        if small_conf < abstention_threshold:
            results["small_abstained"] += 1
        elif small_ans == correct_char:
            results["small_correct"] += 1
        else:
            results["small_wrong"] += 1

    pbar.close()
    
    # Calculate metrics
    large_acc = results["large_correct"] / count
    small_answered = results["small_correct"] + results["small_wrong"]
    small_acc_answered = results["small_correct"] / small_answered if small_answered else 0
    small_abstention_rate = results["small_abstained"] / count

    # Effective accuracy (abstention = 0.25 value for random guess equivalent, or 0.5 for neutral)
    # Using 0.25 (random baseline for 4-choice) as the value of "I don't know" is debatable.
    # Usually abstention > random guess (0.25).
    # If we assume an abstention prevents a hallucination (negative utility), it's valuable.
    # Let's use 0.25 to be conservative (equivalent to random guessing).
    # Or follow the README which suggests 0.5 ("neutral").
    abstention_value = 0.25
    
    large_effective = large_acc
    small_effective = (small_acc_answered * (1 - small_abstention_rate) + 
                       abstention_value * small_abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"DAVID VS GOLIATH RESULTS (n={count})")
    print(f"{'='*60}")
    print(f"\nGPT-2-Large (774M params):")
    print(f"  Accuracy: {large_acc:.1%}")
    print(f"  Effective: {large_effective:.1%}")
    print(f"\nPythia-70m + Abstention (70M params, 11x smaller):")
    print(f"  Accuracy (answered): {small_acc_answered:.1%}")
    print(f"  Abstention rate: {small_abstention_rate:.1%}")
    print(f"  Effective (val={abstention_value}): {small_effective:.1%}")
    print(f"\n{'='*60}")
    
    if small_acc_answered > large_acc:
        print(f"🏆 DAVID WINS! Small model achieves higher accuracy on answered questions!")
    elif small_effective >= large_effective:
        print(f"🏆 DAVID WINS! Small model matches effective accuracy with 11x fewer params!")

if __name__ == "__main__":
    run_comparison()
