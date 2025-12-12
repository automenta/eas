#!/usr/bin/env python3
"""david_vs_goliath.py - Small model beats large model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
from datasets import load_dataset
from tqdm import tqdm

def load_model(name, device):
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
        answer_tokens = [tokenizer.encode(f" {c}")[0] for c in "ABCD"]
        answer_probs = [probs[t].item() for t in answer_tokens]
        
        best_idx = max(range(4), key=lambda i: answer_probs[i])
        confidence = answer_probs[best_idx]
        answer = "ABCD"[best_idx]
    
    return answer, confidence

def run_comparison(num_test=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading models...")
    small_model, small_tok = load_model("EleutherAI/pythia-70m", device)
    large_model, large_tok = load_model("gpt2-large", device)  # 774M params
    
    dataset = load_dataset("lucasmccabe/logiqa", split="validation")
    
    # Results tracking
    results = {
        "small_correct": 0, "small_wrong": 0, "small_abstained": 0,
        "large_correct": 0, "large_wrong": 0
    }
    
    abstention_threshold = 0.35  # Abstain if max prob < 35%
    
    for i in tqdm(range(min(num_test, len(dataset)))):
        ex = dataset[i]
        prompt = f"Q: {ex['question']}\nA:"
        correct = "ABCD"[ex['answer']]
        
        # Large model (no abstention)
        large_ans, large_conf = get_answer_confidence(large_model, large_tok, prompt, device)
        if large_ans == correct:
            results["large_correct"] += 1
        else:
            results["large_wrong"] += 1
        
        # Small model (with abstention)
        small_ans, small_conf = get_answer_confidence(small_model, small_tok, prompt, device)
        if small_conf < abstention_threshold:
            results["small_abstained"] += 1
        elif small_ans == correct:
            results["small_correct"] += 1
        else:
            results["small_wrong"] += 1
    
    # Calculate metrics
    large_acc = results["large_correct"] / num_test
    small_answered = results["small_correct"] + results["small_wrong"]
    small_acc_answered = results["small_correct"] / small_answered if small_answered else 0
    small_abstention_rate = results["small_abstained"] / num_test
    
    # Effective accuracy (abstention = 0.5 value)
    large_effective = large_acc
    small_effective = (small_acc_answered * (1 - small_abstention_rate) + 
                       0.5 * small_abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"DAVID VS GOLIATH RESULTS")
    print(f"{'='*60}")
    print(f"\nGPT-2-Large (774M params):")
    print(f"  Accuracy: {large_acc:.1%}")
    print(f"  Effective: {large_effective:.1%}")
    print(f"\nPythia-70m + Abstention (70M params, 11x smaller):")
    print(f"  Accuracy (answered): {small_acc_answered:.1%}")
    print(f"  Abstention rate: {small_abstention_rate:.1%}")
    print(f"  Effective: {small_effective:.1%}")
    print(f"\n{'='*60}")
    
    if small_acc_answered > large_acc:
        print(f"🏆 DAVID WINS! Small model achieves higher accuracy on answered questions!")
    elif small_effective >= large_effective:
        print(f"🏆 DAVID WINS! Small model matches effective accuracy with 11x fewer params!")

if __name__ == "__main__":
    run_comparison()
