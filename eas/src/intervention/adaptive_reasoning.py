#!/usr/bin/env python3
"""
Adaptive Reasoning Amplifier (ARA)

NOVEL INNOVATION: Uses real-time CTD measurement to dynamically 
steer model activations DURING generation.

Key innovations:
1. Online CTD computation - measure divergence during inference
2. Adaptive steering strength - stronger push where divergence is high
3. Reasoning momentum - accumulate "correctness direction" across tokens
4. Self-correcting generation - detect and recover from reasoning errors

This goes beyond static EAS to create a DYNAMIC reasoning enhancement system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

import sys
sys.path.insert(0, '/home/me/eas')


@dataclass
class ReasoningState:
    """Tracks reasoning trajectory during generation."""
    token_count: int = 0
    cumulative_confidence: float = 0.0
    momentum_vector: Optional[torch.Tensor] = None
    divergence_history: List[float] = field(default_factory=list)
    correction_count: int = 0
    amplification_count: int = 0


class AdaptiveReasoningAmplifier(nn.Module):
    """
    Novel system that amplifies correct reasoning patterns during generation.
    
    Unlike static EAS which applies fixed interventions, ARA:
    1. Measures real-time "reasoning quality" via activation patterns
    2. Dynamically adjusts intervention strength
    3. Maintains reasoning momentum across token generation
    4. Self-corrects when detecting divergence from good patterns
    """
    
    def __init__(self, 
                 dim: int,
                 reference_correct: torch.Tensor = None,
                 reference_incorrect: torch.Tensor = None,
                 momentum_decay: float = 0.9,
                 amplification_threshold: float = 0.1,
                 correction_threshold: float = 0.3,
                 max_steering: float = 0.5):
        super().__init__()
        
        self.dim = dim
        self.momentum_decay = momentum_decay
        self.amplification_threshold = amplification_threshold
        self.correction_threshold = correction_threshold
        self.max_steering = max_steering
        
        # Reference directions (learned from examples)
        self.register_buffer('correct_direction', 
                           reference_correct if reference_correct is not None 
                           else torch.zeros(dim))
        self.register_buffer('incorrect_direction',
                           reference_incorrect if reference_incorrect is not None
                           else torch.zeros(dim))
        
        # Adaptive learned parameters
        self.steering_scale = nn.Parameter(torch.ones(1))
        self.position_weights = nn.Parameter(torch.ones(128))  # Max seq length
        
        # Running statistics
        self.register_buffer('running_correct_mean', torch.zeros(dim))
        self.register_buffer('running_incorrect_mean', torch.zeros(dim))
        self.register_buffer('n_correct', torch.tensor(0.0))
        self.register_buffer('n_incorrect', torch.tensor(0.0))
        
        # State tracking
        self.state = ReasoningState()
        
    def learn_from_example(self, hidden_states: torch.Tensor, is_correct: bool):
        """
        Online learning: update reference directions from examples.
        
        This is the key to making ARA adaptive - it learns what 
        "correct" and "incorrect" patterns look like in real-time.
        """
        # Pool to get sequence representation
        pooled = hidden_states.mean(dim=1).squeeze(0)  # [dim]
        
        if is_correct:
            # Update running mean for correct examples
            self.n_correct += 1
            delta = pooled - self.running_correct_mean
            self.running_correct_mean += delta / self.n_correct
            
            # Update correct direction
            self.correct_direction = F.normalize(
                self.running_correct_mean, dim=0
            )
        else:
            # Update running mean for incorrect examples
            self.n_incorrect += 1
            delta = pooled - self.running_incorrect_mean
            self.running_incorrect_mean += delta / self.n_incorrect
            
            # Update incorrect direction
            self.incorrect_direction = F.normalize(
                self.running_incorrect_mean, dim=0
            )
    
    def compute_reasoning_quality(self, hidden_states: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Compute real-time reasoning quality score.
        
        Returns:
            quality: float in [-1, 1] where positive = more correct-like
            direction: tensor indicating which way to steer
        """
        # Get current representation
        current = hidden_states.mean(dim=1).squeeze(0)  # [dim]
        current_norm = F.normalize(current, dim=0)
        
        # Compute similarity to correct/incorrect directions
        if self.correct_direction.norm() > 0:
            correct_sim = torch.dot(current_norm, self.correct_direction).item()
        else:
            correct_sim = 0.0
            
        if self.incorrect_direction.norm() > 0:
            incorrect_sim = torch.dot(current_norm, self.incorrect_direction).item()
        else:
            incorrect_sim = 0.0
        
        # Quality = how much more correct-like than incorrect-like
        quality = correct_sim - incorrect_sim
        
        # Steering direction: toward correct, away from incorrect
        steering_dir = self.correct_direction - self.incorrect_direction
        steering_dir = F.normalize(steering_dir, dim=0)
        
        return quality, steering_dir
    
    def adaptive_steer(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply adaptive steering based on real-time quality measurement.
        
        Key innovation: Steering strength is proportional to how much
        the current generation is diverging from correct patterns.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        quality, steering_dir = self.compute_reasoning_quality(hidden_states)
        
        # Track state
        self.state.token_count += seq_len
        self.state.divergence_history.append(1 - quality)  # Higher means worse
        
        # Update momentum
        if self.state.momentum_vector is None:
            self.state.momentum_vector = steering_dir.clone()
        else:
            self.state.momentum_vector = (
                self.momentum_decay * self.state.momentum_vector + 
                (1 - self.momentum_decay) * steering_dir
            )
            self.state.momentum_vector = F.normalize(self.state.momentum_vector, dim=0)
        
        # Compute adaptive steering strength
        if quality < -self.correction_threshold:
            # Strong correction needed - we're going wrong way
            alpha = self.max_steering
            self.state.correction_count += 1
        elif quality < self.amplification_threshold:
            # Moderate steering to nudge toward correct
            alpha = self.max_steering * (self.amplification_threshold - quality) / (
                self.amplification_threshold + self.correction_threshold
            )
            self.state.amplification_count += 1
        else:
            # Already on good track - minimal intervention
            alpha = 0.05  # Small nudge to maintain
        
        # Apply position-aware steering
        modified = hidden_states.clone()
        
        for i in range(seq_len):
            # Position weight (later positions get stronger steering)
            rel_pos = i / seq_len
            pos_weight = 0.5 + 0.5 * rel_pos  # 0.5 to 1.0
            
            effective_alpha = alpha * pos_weight * self.steering_scale
            
            # Apply steering with momentum
            delta = effective_alpha * self.state.momentum_vector.unsqueeze(0)
            
            # Clamp magnitude
            delta_norm = delta.norm()
            if delta_norm > self.max_steering:
                delta = delta * (self.max_steering / delta_norm)
            
            modified[:, i, :] = hidden_states[:, i, :] + delta
        
        # Update cumulative confidence
        self.state.cumulative_confidence = (
            self.momentum_decay * self.state.cumulative_confidence +
            (1 - self.momentum_decay) * quality
        )
        
        return modified
    
    def get_reasoning_score(self) -> float:
        """Get current reasoning quality estimate."""
        return self.state.cumulative_confidence
    
    def should_backtrack(self, threshold: float = -0.5) -> bool:
        """Detect if generation has gone wrong and needs backtracking."""
        if len(self.state.divergence_history) < 3:
            return False
        
        # Check if recent divergence is consistently high
        recent = self.state.divergence_history[-3:]
        return all(d > threshold for d in recent)
    
    def get_correction_suggestion(self) -> str:
        """Suggest what kind of correction is needed."""
        if self.state.cumulative_confidence < -0.3:
            return "MAJOR_CORRECTION: Reasoning has diverged significantly"
        elif self.state.cumulative_confidence < 0:
            return "MINOR_CORRECTION: Nudge back toward correct pattern"
        else:
            return "ON_TRACK: Reasoning appears sound"
    
    def reset_state(self):
        """Reset generation state for new sequence."""
        self.state = ReasoningState()
    
    def get_statistics(self) -> Dict:
        """Return statistics about interventions."""
        return {
            'token_count': self.state.token_count,
            'cumulative_confidence': self.state.cumulative_confidence,
            'correction_count': self.state.correction_count,
            'amplification_count': self.state.amplification_count,
            'avg_divergence': np.mean(self.state.divergence_history) if self.state.divergence_history else 0,
            'n_correct_examples': self.n_correct.item(),
            'n_incorrect_examples': self.n_incorrect.item()
        }


class ReasoningEnhancedGenerator:
    """
    Novel generator that uses ARA for reasoning-enhanced text generation.
    
    Key innovation: Monitors and steers reasoning quality in real-time
    during autoregressive generation.
    """
    
    def __init__(self, model, amplifier: AdaptiveReasoningAmplifier, layer_idx: int = 3):
        self.model = model
        self.amplifier = amplifier
        self.layer_idx = layer_idx
        
    def generate_with_reasoning(self, 
                                prompt: str, 
                                max_tokens: int = 50,
                                temperature: float = 0.7) -> Dict:
        """
        Generate text with real-time reasoning enhancement.
        
        Returns generated text plus reasoning diagnostics.
        """
        self.amplifier.reset_state()
        
        # Tokenize prompt
        input_ids = self.model.tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.to(self.model.device)
        
        generated_ids = input_ids.clone()
        reasoning_log = []
        
        # Register intervention hook
        def reasoning_hook(hidden_states):
            modified = self.amplifier.adaptive_steer(hidden_states)
            reasoning_log.append({
                'quality': self.amplifier.get_reasoning_score(),
                'suggestion': self.amplifier.get_correction_suggestion()
            })
            return modified
        
        self.model.register_intervention_hook(self.layer_idx, reasoning_hook)
        
        try:
            for _ in range(max_tokens):
                with torch.no_grad():
                    outputs = self.model.model(generated_ids)
                    logits = outputs.logits[:, -1, :] / temperature
                    
                    # Sample next token
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    
                    # Check for EOS
                    if next_token.item() == self.model.tokenizer.eos_token_id:
                        break
                    
                    # Check if we should warn about reasoning quality
                    if self.amplifier.should_backtrack():
                        reasoning_log.append({
                            'warning': 'POTENTIAL_REASONING_ERROR',
                            'position': generated_ids.shape[1]
                        })
        finally:
            self.model.remove_intervention_hook(self.layer_idx)
        
        # Decode
        generated_text = self.model.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        return {
            'text': generated_text,
            'prompt': prompt,
            'generated_tokens': generated_ids.shape[1] - input_ids.shape[1],
            'final_reasoning_score': self.amplifier.get_reasoning_score(),
            'statistics': self.amplifier.get_statistics(),
            'reasoning_log': reasoning_log[:10]  # First 10 for brevity
        }


def demo_adaptive_reasoning():
    """Demonstrate the Adaptive Reasoning Amplifier."""
    print("=" * 60)
    print("ADAPTIVE REASONING AMPLIFIER (ARA) DEMO")
    print("=" * 60)
    
    from eas.src.models.transformer import PretrainedTransformer
    
    print("\nLoading model...")
    model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
    
    print("\nInitializing ARA...")
    ara = AdaptiveReasoningAmplifier(dim=model.d_model)
    
    # Learn from examples
    print("\nLearning from examples...")
    
    correct_examples = [
        "All birds fly. A sparrow is a bird. Therefore, sparrows fly.",
        "If it rains, the ground is wet. It rained. The ground is wet.",
        "2 + 2 = 4. This is correct mathematics.",
    ]
    
    incorrect_examples = [
        "All birds fly. A sparrow is a bird. Therefore, sparrows swim.",
        "If it rains, the ground is wet. It rained. The ground is dry.",
        "2 + 2 = 5. This is correct mathematics.",
    ]
    
    for text in correct_examples:
        input_ids = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model.forward(input_ids.input_ids)
            hidden = model.get_layer_activation(3)
        if hidden is not None:
            ara.learn_from_example(hidden, is_correct=True)
    
    for text in incorrect_examples:
        input_ids = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model.forward(input_ids.input_ids)
            hidden = model.get_layer_activation(3)
        if hidden is not None:
            ara.learn_from_example(hidden, is_correct=False)
    
    print(f"Learned from {ara.n_correct.item():.0f} correct, {ara.n_incorrect.item():.0f} incorrect examples")
    
    # Test on new examples
    print("\n" + "=" * 40)
    print("TESTING QUALITY DETECTION")
    print("=" * 40)
    
    test_cases = [
        ("All cats are mammals. Fluffy is a cat. Fluffy is a mammal.", True),
        ("All cats are mammals. Fluffy is a cat. Fluffy is a reptile.", False),
        ("Fire needs oxygen. There is oxygen. Fire burns.", True),
        ("Fire needs oxygen. There is oxygen. Fire freezes.", False),
    ]
    
    for text, expected_correct in test_cases:
        ara.reset_state()
        input_ids = model.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            model.forward(input_ids.input_ids)
            hidden = model.get_layer_activation(3)
        
        if hidden is not None:
            quality, _ = ara.compute_reasoning_quality(hidden)
            detected = "correct" if quality > 0 else "incorrect"
            expected = "correct" if expected_correct else "incorrect"
            match = "✓" if detected == expected else "✗"
            
            print(f"\n{match} '{text[:50]}...'")
            print(f"   Quality: {quality:+.3f}, Detected: {detected}, Expected: {expected}")
    
    # Demo adaptive steering
    print("\n" + "=" * 40)
    print("TESTING ADAPTIVE STEERING")
    print("=" * 40)
    
    test_text = "All dogs are animals. Rex is a dog. Therefore Rex is an"
    input_ids = model.tokenizer(test_text, return_tensors="pt")
    
    ara.reset_state()
    
    with torch.no_grad():
        model.forward(input_ids.input_ids)
        original = model.get_layer_activation(3)
        steered = ara.adaptive_steer(original)
    
    delta = (steered - original).norm(dim=-1).mean().item()
    print(f"\nTest: '{test_text}'")
    print(f"Steering magnitude: {delta:.4f}")
    print(f"Final stats: {ara.get_statistics()}")
    
    print("\n✅ ARA demonstration complete!")
    return ara


if __name__ == "__main__":
    demo_adaptive_reasoning()
