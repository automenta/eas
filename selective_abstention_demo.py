#!/usr/bin/env python3
"""selective_abstention_demo.py - Adaptive MCRE with Calibration"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class MCREState:
    uncertainty: float
    z_score: float
    confidence: float
    should_abstain: bool
    predicted_answer: str

class MCRE:
    """
    Adaptive Meta-Cognitive Reasoning Engine.
    Uses Z-score based thresholding on entropy to adapt to different models.
    """
    
    def __init__(self, model, tokenizer, device="cpu", z_threshold=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold

        # Calibration stats (default to heuristics until calibrated)
        self.mean_entropy = 2.5
        self.std_entropy = 0.5
        self.is_calibrated = False

    def calibrate(self, dataset, n=50, prompt_fn=None):
        """
        Run on a small dataset to learn baseline entropy distribution.

        Args:
            dataset: The dataset to calibrate on.
            n: Number of samples to use.
            prompt_fn: A function that takes an example (dict) and returns a prompt (str).
                       If None, defaults to simple "Question: ...\nAnswer:" format.
        """
        print(f"🔧 Calibrating MCRE on {n} examples...")
        entropies = []

        model_training = self.model.training
        self.model.eval()

        for i in range(min(n, len(dataset))):
            ex = dataset[i]

            if prompt_fn:
                text = prompt_fn(ex)
            else:
                # Default behavior
                text = f"Question: {ex['question']}\nAnswer:"

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]
                e = self._calculate_entropy(logits)
                entropies.append(e)

        self.mean_entropy = np.mean(entropies)
        self.std_entropy = np.std(entropies) + 1e-6 # Avoid div by zero
        self.is_calibrated = True

        print(f"✅ Calibration complete: µ={self.mean_entropy:.2f}, σ={self.std_entropy:.2f}")
        self.model.train(model_training)

    def _calculate_entropy(self, logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-9)
        entropy = -torch.sum(probs * log_probs, dim=-1).item()
        return entropy

    def get_answer_and_confidence(self, logits: torch.Tensor) -> tuple[str, float]:
        probs = torch.softmax(logits, dim=-1)
        
        answer_probs = {}
        for answer in "ABCD":
            # Handle tokenization
            tokens = [self.tokenizer.encode(f" {answer}", add_special_tokens=False),
                     self.tokenizer.encode(answer, add_special_tokens=False)]
            # Use the first valid token found
            token_id = None
            for t in tokens:
                if t:
                    token_id = t[0]
                    break

            if token_id is not None and token_id < logits.size(0):
                answer_probs[answer] = probs[token_id].item()
            else:
                answer_probs[answer] = 0.0
        
        best_answer = max(answer_probs, key=answer_probs.get) if answer_probs else "A"
        confidence = answer_probs[best_answer]
        return best_answer, confidence
    
    def evaluate(self, prompt: str) -> MCREState:
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]
        entropy = self._calculate_entropy(logits)

        # Z-Score Calculation
        z_score = (entropy - self.mean_entropy) / self.std_entropy
        
        answer, answer_conf = self.get_answer_and_confidence(logits)
        
        # Adaptive Abstention Logic:
        # Abstain if entropy is significantly higher than model's baseline (high Z-score)
        # OR if direct answer confidence is very low.
        should_abstain = (z_score > self.z_threshold) or (answer_conf < 0.2)
        
        return MCREState(
            uncertainty=entropy,
            z_score=z_score,
            confidence=answer_conf,
            should_abstain=should_abstain,
            predicted_answer=answer
        )
