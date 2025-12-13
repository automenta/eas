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
    valid_mass: float # Total probability mass of A,B,C,D

class MCRE:
    """
    Adaptive Meta-Cognitive Reasoning Engine.
    Uses Z-score based thresholding on entropy to adapt to different models.
    Now uses Restricted Softmax (A/B/C/D) for "Smarter" evaluation.
    """
    
    def __init__(self, model, tokenizer, device="cpu", z_threshold=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold

        # Pre-compute candidate token IDs
        self.candidates = ["A", "B", "C", "D"]
        self.candidate_ids = {}
        for c in self.candidates:
            # Try with and without leading space
            ids = [self.tokenizer.encode(f" {c}", add_special_tokens=False),
                   self.tokenizer.encode(c, add_special_tokens=False)]
            # Pick the first valid non-empty one
            for i in ids:
                if i:
                    self.candidate_ids[c] = i[0]
                    break
            if c not in self.candidate_ids:
                print(f"Warning: Could not find token ID for {c}")
                self.candidate_ids[c] = 0 # Fallback

        # Calibration stats (default to heuristics until calibrated)
        self.mean_entropy = 1.0 # Lower default for restricted set (max ln(4) ≈ 1.38)
        self.std_entropy = 0.2
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
                probs, valid_mass = self._get_restricted_probs(logits)
                e = self._calculate_entropy(probs)
                entropies.append(e)

        self.mean_entropy = np.mean(entropies)
        self.std_entropy = np.std(entropies) + 1e-6 # Avoid div by zero
        self.is_calibrated = True

        print(f"✅ Calibration complete: µ={self.mean_entropy:.2f}, σ={self.std_entropy:.2f}")
        self.model.train(model_training)

    def _get_restricted_probs(self, logits: torch.Tensor) -> tuple[dict, float]:
        """Normalize logits over A, B, C, D only."""
        target_ids = [self.candidate_ids[c] for c in self.candidates]
        target_logits = logits[target_ids]
        
        # Calculate restricted softmax
        restricted_probs_tensor = torch.softmax(target_logits, dim=-1)
        restricted_probs = {c: restricted_probs_tensor[i].item() for i, c in enumerate(self.candidates)}

        # Calculate valid mass (from global softmax)
        global_probs = torch.softmax(logits, dim=-1)
        valid_mass = sum(global_probs[tid].item() for tid in target_ids)
        
        return restricted_probs, valid_mass

    def _calculate_entropy(self, probs: dict) -> float:
        """Calculate entropy of the restricted distribution."""
        p_values = np.array(list(probs.values()))
        # Clip to avoid log(0)
        p_values = np.clip(p_values, 1e-9, 1.0)
        entropy = -np.sum(p_values * np.log(p_values))
        return float(entropy)
    
    def evaluate(self, prompt: str) -> MCREState:
        inputs = self.tokenizer(prompt, return_tensors="pt", 
                               truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits[0, -1, :]

        # Get Restricted Distribution
        probs, valid_mass = self._get_restricted_probs(logits)
        entropy = self._calculate_entropy(probs)

        # Z-Score Calculation
        z_score = (entropy - self.mean_entropy) / self.std_entropy
        
        # Determine Answer
        best_answer = max(probs, key=probs.get)
        confidence = probs[best_answer]

        # Adaptive Abstention Logic (Smarter):
        # 1. Uncertainty: Is the distribution flat? (High Z-score)
        # 2. Format Failure: Is the model predicting something else entirely? (Low Valid Mass)
        # Note: We now check confidence relative to restricted set, so it sums to 1.
        # But if valid_mass is tiny (e.g. 0.001), the restricted distribution is noise.
        
        should_abstain = (z_score > self.z_threshold) or (confidence < 0.4) or (valid_mass < 0.01)
        
        return MCREState(
            uncertainty=entropy,
            z_score=z_score,
            confidence=confidence,
            should_abstain=should_abstain,
            predicted_answer=best_answer,
            valid_mass=valid_mass
        )
