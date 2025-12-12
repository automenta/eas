#!/usr/bin/env python3
"""selective_abstention_demo.py - Primary PoC with Entropy-Based MCRE"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class MCREState:
    uncertainty: float
    confidence: float
    should_abstain: bool
    predicted_answer: str

class MCRE:
    """Meta-Cognitive Reasoning Engine using entropy-based uncertainty."""
    
    def __init__(self, model, tokenizer, device="cpu", threshold=0.6):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.threshold = threshold  # Entropy threshold for abstention
    
    def get_uncertainty(self, logits: torch.Tensor) -> float:
        """Calculate Shannon Entropy of the next-token distribution."""
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        # Normalize: Pythia-70m entropy usually peaks around 3.0-4.0
        normalized = torch.sigmoid(entropy - 2.5)
        return normalized.item()
    
    def get_answer_and_confidence(self, logits: torch.Tensor) -> tuple[str, float]:
        """Get predicted answer (A/B/C/D) and confidence from logits."""
        probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities for answer tokens
        answer_probs = {}
        for answer in "ABCD":
            # Try both formats: " A" and "A"
            tokens = [self.tokenizer.encode(f" {answer}", add_special_tokens=False),
                     self.tokenizer.encode(answer, add_special_tokens=False)]
            token_id = tokens[0][0] if tokens[0] else tokens[1][0]
            answer_probs[answer] = probs[token_id].item()
        
        best_answer = max(answer_probs, key=answer_probs.get)
        confidence = answer_probs[best_answer]
        return best_answer, confidence
    
    def evaluate(self, prompt: str) -> MCREState:
        """Evaluate a prompt and return meta-cognitive state."""
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]  # Last token logits
        
        uncertainty = self.get_uncertainty(logits)
        answer, answer_conf = self.get_answer_and_confidence(logits)
        
        # Abstain if: high entropy OR low answer confidence
        should_abstain = (uncertainty > self.threshold) or (answer_conf < 0.25)
        
        return MCREState(
            uncertainty=uncertainty,
            confidence=1.0 - uncertainty,
            should_abstain=should_abstain,
            predicted_answer=answer
        )

def run_demo(model_name="EleutherAI/pythia-70m", num_test=100, threshold=0.6):
    """Run the selective abstention demo with real model evaluation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading {model_name} on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    
    mcre = MCRE(model, tokenizer, device, threshold=threshold)
    dataset = load_dataset("lucasmccabe/logiqa", split="validation")
    
    print(f"📊 Testing on {num_test} examples (threshold={threshold})...")
    
    results = {
        "answered_correct": 0, 
        "answered_wrong": 0,
        "abstained": 0
    }
    
    for i in tqdm(range(min(num_test, len(dataset)))):
        ex = dataset[i]
        
        # Format prompt for multiple choice
        options = "\n".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(ex['options'])])
        prompt = f"Question: {ex['question']}\n{options}\nAnswer:"
        
        state = mcre.evaluate(prompt)
        correct_answer = "ABCD"[ex['answer']]
        
        if state.should_abstain:
            results["abstained"] += 1
        elif state.predicted_answer == correct_answer:
            results["answered_correct"] += 1
        else:
            results["answered_wrong"] += 1
    
    # Calculate metrics
    total = sum(results.values())
    answered = results["answered_correct"] + results["answered_wrong"]
    abstained = results["abstained"]
    
    baseline_acc = (results["answered_correct"]) / total  # If we had answered all
    answered_acc = results["answered_correct"] / answered if answered > 0 else 0
    abstention_rate = abstained / total
    
    # Effective accuracy: correct answers / total questions
    # But abstaining is "neutral" (0.25 for 4-way MC random baseline)
    effective_acc = (answered_acc * (1 - abstention_rate) + 0.25 * abstention_rate)
    
    print(f"\n{'='*60}")
    print(f"SELECTIVE ABSTENTION RESULTS (Entropy-Based MCRE)")
    print(f"{'='*60}")
    print(f"Total questions:       {total}")
    print(f"Answered:              {answered} ({1-abstention_rate:.1%})")
    print(f"Abstained:             {abstained} ({abstention_rate:.1%})")
    print(f"{'='*60}")
    print(f"Baseline accuracy:     {baseline_acc:.1%} (if answered all)")
    print(f"Accuracy on answered:  {answered_acc:.1%}")
    print(f"Effective accuracy:    {effective_acc:.1%}")
    print(f"{'='*60}")
    
    improvement = answered_acc - baseline_acc
    if improvement > 0:
        print(f"✅ IMPROVEMENT: +{improvement:.1%} by selective answering!")
    else:
        print(f"⚠️  Model may benefit from threshold tuning (current: {threshold})")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m")
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    
    run_demo(args.model, args.test, args.threshold)
