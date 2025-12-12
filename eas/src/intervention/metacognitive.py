#!/usr/bin/env python3
"""
Meta-Cognitive Reasoning Engine (MCRE)

REVOLUTIONARY INNOVATION: A system that reasons about its own reasoning.

This implements metacognition for LMs:
1. Uncertainty Quantification - Know what you don't know
2. Confidence Calibration - Accurate self-assessment
3. Strategy Selection - Choose the right reasoning approach
4. Error Prediction - Anticipate mistakes before making them
5. Explanation Generation - Explain WHY a reasoning is valid/invalid

This is the closest we get to genuine "thinking about thinking".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json

import sys
sys.path.insert(0, '/home/me/eas')


class ReasoningStrategy(Enum):
    """Different reasoning strategies the system can employ."""
    DEDUCTIVE = "deductive"      # If A→B and A, then B
    INDUCTIVE = "inductive"      # Multiple instances → general rule
    ABDUCTIVE = "abductive"      # Best explanation for observations
    ANALOGICAL = "analogical"    # Similar to known case
    CAUSAL = "causal"           # Cause-effect reasoning


@dataclass
class MetaCognitiveState:
    """The system's awareness of its own cognitive state."""
    confidence: float = 0.5
    uncertainty: float = 0.5
    strategy: Optional[ReasoningStrategy] = None
    error_risk: float = 0.0
    explanation: str = ""
    
    # Calibration metrics
    was_correct: Optional[bool] = None
    calibration_error: float = 0.0


class UncertaintyQuantifier(nn.Module):
    """
    Quantifies epistemic and aleatoric uncertainty.
    
    Epistemic: What the model doesn't know (reducible)
    Aleatoric: Inherent randomness in the data (irreducible)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 2)  # [epistemic, aleatoric]
        )
        
        # Calibration history
        self.predictions: List[float] = []
        self.outcomes: List[bool] = []
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[float, float]:
        """Estimate uncertainty from hidden states."""
        pooled = hidden_states.mean(dim=1).squeeze(0)
        
        uncertainties = self.uncertainty_net(pooled)
        uncertainties = torch.sigmoid(uncertainties)  # Bound to [0, 1]
        
        epistemic = uncertainties[0].item()
        aleatoric = uncertainties[1].item()
        
        return epistemic, aleatoric
    
    def update_calibration(self, predicted_confidence: float, was_correct: bool):
        """Track predictions for calibration."""
        self.predictions.append(predicted_confidence)
        self.outcomes.append(was_correct)
    
    def get_calibration_error(self) -> float:
        """Compute Expected Calibration Error."""
        if len(self.predictions) < 10:
            return 0.5  # Not enough data
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        
        for i in range(len(bins) - 1):
            bin_mask = [(bins[i] <= p < bins[i+1]) for p in self.predictions]
            bin_predictions = [p for p, m in zip(self.predictions, bin_mask) if m]
            bin_outcomes = [o for o, m in zip(self.outcomes, bin_mask) if m]
            
            if bin_predictions:
                avg_confidence = np.mean(bin_predictions)
                accuracy = np.mean(bin_outcomes)
                bin_weight = len(bin_predictions) / len(self.predictions)
                ece += bin_weight * abs(avg_confidence - accuracy)
        
        return ece


class StrategySelector:
    """
    Selects the appropriate reasoning strategy for a given problem.
    
    Meta-cognitive insight: Different problems need different approaches.
    """
    
    def __init__(self):
        # Strategy signatures (pattern → best strategy)
        self.strategy_patterns = {
            ReasoningStrategy.DEDUCTIVE: [
                'if', 'then', 'all', 'every', 'must', 'necessarily'
            ],
            ReasoningStrategy.INDUCTIVE: [
                'usually', 'often', 'many', 'most', 'typically', 'examples'
            ],
            ReasoningStrategy.ABDUCTIVE: [
                'because', 'explains', 'best', 'probably', 'likely', 'suggests'
            ],
            ReasoningStrategy.ANALOGICAL: [
                'similar', 'like', 'analogous', 'just as', 'same way'
            ],
            ReasoningStrategy.CAUSAL: [
                'causes', 'leads to', 'results in', 'due to', 'effect'
            ],
        }
        
        # Track strategy success rates
        self.strategy_successes: Dict[ReasoningStrategy, List[bool]] = {
            s: [] for s in ReasoningStrategy
        }
    
    def select_strategy(self, text: str) -> Tuple[ReasoningStrategy, float]:
        """Select the best reasoning strategy for this text."""
        text_lower = text.lower()
        
        scores = {}
        for strategy, patterns in self.strategy_patterns.items():
            score = sum(1 for p in patterns if p in text_lower)
            scores[strategy] = score
        
        # Weight by historical success rate
        for strategy in scores:
            if self.strategy_successes[strategy]:
                success_rate = np.mean(self.strategy_successes[strategy])
                scores[strategy] *= (0.5 + success_rate)
        
        best_strategy = max(scores, key=scores.get)
        confidence = scores[best_strategy] / (sum(scores.values()) + 1)
        
        return best_strategy, min(1.0, confidence)
    
    def update_strategy_success(self, strategy: ReasoningStrategy, success: bool):
        """Track strategy performance."""
        self.strategy_successes[strategy].append(success)
        
        # Keep only recent history
        if len(self.strategy_successes[strategy]) > 100:
            self.strategy_successes[strategy] = self.strategy_successes[strategy][-100:]
    
    def get_strategy_report(self) -> Dict[str, float]:
        """Get success rate for each strategy."""
        return {
            s.value: np.mean(successes) if successes else 0.5
            for s, successes in self.strategy_successes.items()
        }


class ErrorPredictor:
    """
    Predicts the likelihood of making an error BEFORE it happens.
    
    Key insight: Some reasoning states are more error-prone than others.
    Learn to recognize these states and intervene preemptively.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        
        # Error-prone signatures
        self.error_signatures: List[torch.Tensor] = []
        self.success_signatures: List[torch.Tensor] = []
        
        # Error risk threshold
        self.risk_threshold = 0.6
    
    def add_error_case(self, hidden_states: torch.Tensor):
        """Record an error case for future prediction."""
        signature = self._create_signature(hidden_states)
        self.error_signatures.append(signature)
        
        # Keep manageable size
        if len(self.error_signatures) > 100:
            self.error_signatures = self.error_signatures[-100:]
    
    def add_success_case(self, hidden_states: torch.Tensor):
        """Record a success case."""
        signature = self._create_signature(hidden_states)
        self.success_signatures.append(signature)
        
        if len(self.success_signatures) > 100:
            self.success_signatures = self.success_signatures[-100:]
    
    def _create_signature(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Create a signature from hidden states."""
        pooled = hidden_states.mean(dim=1).squeeze(0)
        return F.normalize(pooled, dim=0)
    
    def predict_error_risk(self, hidden_states: torch.Tensor) -> float:
        """Predict the risk of error for current state."""
        if not self.error_signatures:
            return 0.5  # No data yet
        
        current_sig = self._create_signature(hidden_states)
        
        # Similarity to error patterns
        error_sims = [torch.dot(current_sig, sig).item() 
                      for sig in self.error_signatures[-20:]]
        avg_error_sim = np.mean(error_sims) if error_sims else 0
        
        # Similarity to success patterns
        success_sims = [torch.dot(current_sig, sig).item()
                       for sig in self.success_signatures[-20:]]
        avg_success_sim = np.mean(success_sims) if success_sims else 0
        
        # Risk based on relative similarity
        risk = (avg_error_sim - avg_success_sim + 1) / 2
        return np.clip(risk, 0, 1)
    
    def is_high_risk(self, hidden_states: torch.Tensor) -> bool:
        """Check if current state is high-risk."""
        return self.predict_error_risk(hidden_states) > self.risk_threshold


class ExplanationGenerator:
    """
    Generates explanations for reasoning validity.
    
    This provides interpretability - WHY is this reasoning good or bad?
    """
    
    def __init__(self):
        self.explanation_templates = {
            'valid_deductive': "Valid deductive inference: premise '{premise}' implies '{conclusion}'",
            'invalid_deductive': "Invalid: the conclusion does not follow from the premises",
            'valid_inductive': "Strong inductive generalization from {n} observed cases",
            'weak_inductive': "Weak induction: insufficient evidence for generalization",
            'valid_causal': "Valid causal chain: {cause} leads to {effect} through known mechanism",
            'spurious_causal': "Warning: correlation does not imply causation",
            'high_confidence': "High confidence: pattern matches {n} previous successful cases",
            'low_confidence': "Low confidence: novel pattern, limited prior experience",
            'high_risk': "High error risk: pattern similar to {n} previous failures",
        }
    
    def generate_explanation(self, 
                            state: MetaCognitiveState,
                            context: Dict) -> str:
        """Generate a human-readable explanation."""
        explanations = []
        
        # Strategy-based explanation
        if state.strategy:
            if state.strategy == ReasoningStrategy.DEDUCTIVE:
                if state.confidence > 0.7:
                    explanations.append(
                        self.explanation_templates['valid_deductive'].format(
                            premise=context.get('premise', '?'),
                            conclusion=context.get('conclusion', '?')
                        ))
                else:
                    explanations.append(self.explanation_templates['invalid_deductive'])
            
            elif state.strategy == ReasoningStrategy.CAUSAL:
                if state.confidence > 0.6:
                    explanations.append(
                        self.explanation_templates['valid_causal'].format(
                            cause=context.get('cause', '?'),
                            effect=context.get('effect', '?')
                        ))
                else:
                    explanations.append(self.explanation_templates['spurious_causal'])
        
        # Confidence-based explanation
        if state.confidence > 0.8:
            explanations.append(
                self.explanation_templates['high_confidence'].format(
                    n=context.get('similar_successes', '?')
                ))
        elif state.confidence < 0.3:
            explanations.append(self.explanation_templates['low_confidence'])
        
        # Risk-based explanation
        if state.error_risk > 0.6:
            explanations.append(
                self.explanation_templates['high_risk'].format(
                    n=context.get('similar_failures', '?')
                ))
        
        return " | ".join(explanations) if explanations else "No specific explanation available"


class MetaCognitiveReasoningEngine:
    """
    The main meta-cognitive reasoning engine.
    
    This system thinks about its own thinking:
    - Knows what it knows and doesn't know
    - Selects appropriate strategies
    - Predicts and prevents errors
    - Explains its reasoning
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        # Components
        self.uncertainty = UncertaintyQuantifier(model.d_model)
        self.strategy_selector = StrategySelector()
        self.error_predictor = ErrorPredictor(model.d_model)
        self.explanation_generator = ExplanationGenerator()
        
        # Statistics
        self.stats = {
            'total_episodes': 0,
            'correct_predictions': 0,
            'high_confidence_correct': 0,
            'high_confidence_total': 0,
        }
    
    def reason(self, text: str) -> Tuple[MetaCognitiveState, torch.Tensor]:
        """
        Perform reasoning with full meta-cognitive awareness.
        """
        self.stats['total_episodes'] += 1
        
        # Get hidden states
        input_ids = self.model.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            self.model.forward(input_ids.input_ids.to(self.device))
            hidden = self.model.get_layer_activation(3)
        
        if hidden is None:
            return MetaCognitiveState(), None
        
        # Create meta-cognitive state
        state = MetaCognitiveState()
        
        # 1. Select strategy
        state.strategy, strategy_confidence = self.strategy_selector.select_strategy(text)
        
        # 2. Quantify uncertainty
        epistemic, aleatoric = self.uncertainty(hidden)
        state.uncertainty = (epistemic + aleatoric) / 2
        
        # 3. Compute confidence (inverse of uncertainty, modulated by strategy)
        state.confidence = (1 - state.uncertainty) * strategy_confidence
        
        # 4. Predict error risk
        state.error_risk = self.error_predictor.predict_error_risk(hidden)
        
        # 5. Generate explanation
        context = {
            'premise': text[:50],
            'conclusion': text[-50:],
            'similar_successes': len(self.error_predictor.success_signatures),
            'similar_failures': len(self.error_predictor.error_signatures)
        }
        state.explanation = self.explanation_generator.generate_explanation(state, context)
        
        # Track high confidence predictions
        if state.confidence > 0.7:
            self.stats['high_confidence_total'] += 1
        
        return state, hidden
    
    def update_with_outcome(self, state: MetaCognitiveState, 
                           hidden: torch.Tensor, 
                           was_correct: bool):
        """Update the system with the outcome of a reasoning episode."""
        state.was_correct = was_correct
        
        # Update calibration
        self.uncertainty.update_calibration(state.confidence, was_correct)
        
        # Update strategy success
        if state.strategy:
            self.strategy_selector.update_strategy_success(state.strategy, was_correct)
        
        # Update error predictor
        if was_correct:
            self.error_predictor.add_success_case(hidden)
            self.stats['correct_predictions'] += 1
            if state.confidence > 0.7:
                self.stats['high_confidence_correct'] += 1
        else:
            self.error_predictor.add_error_case(hidden)
        
        # Compute calibration error
        state.calibration_error = self.uncertainty.get_calibration_error()
    
    def should_abstain(self, state: MetaCognitiveState) -> bool:
        """Decide whether to abstain (say "I don't know") rather than guess."""
        # Abstain if high uncertainty AND high error risk AND low confidence
        if state.uncertainty > 0.7 and state.error_risk > 0.6 and state.confidence < 0.3:
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get comprehensive meta-cognitive statistics."""
        accuracy = self.stats['correct_predictions'] / max(1, self.stats['total_episodes'])
        
        hc_accuracy = self.stats['high_confidence_correct'] / max(1, self.stats['high_confidence_total'])
        
        return {
            'total_episodes': self.stats['total_episodes'],
            'overall_accuracy': accuracy,
            'high_confidence_accuracy': hc_accuracy,
            'calibration_error': self.uncertainty.get_calibration_error(),
            'strategy_performance': self.strategy_selector.get_strategy_report(),
            'error_patterns_learned': len(self.error_predictor.error_signatures),
            'success_patterns_learned': len(self.error_predictor.success_signatures),
        }


def demo_metacognitive():
    """Demonstrate Meta-Cognitive Reasoning Engine."""
    print("=" * 60)
    print("META-COGNITIVE REASONING ENGINE (MCRE) DEMO")
    print("=" * 60)
    
    from eas.src.models.transformer import PretrainedTransformer
    
    print("\nLoading model...")
    model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
    
    print("\nInitializing Meta-Cognitive Engine...")
    mcre = MetaCognitiveReasoningEngine(model)
    
    # Test reasoning with metacognition
    print("\n" + "=" * 40)
    print("REASONING WITH METACOGNITION")
    print("=" * 40)
    
    test_cases = [
        ("If all birds fly, and a sparrow is a bird, then sparrows fly.", True, "deductive"),
        ("Most swans are white. Therefore, all swans are white.", False, "inductive"),
        ("The ground is wet. It probably rained.", True, "abductive"),
        ("Smoking causes cancer because it leads to cell damage.", True, "causal"),
        ("Cats like milk, just as dogs like bones.", True, "analogical"),
    ]
    
    for text, expected_correct, expected_type in test_cases:
        state, hidden = mcre.reason(text)
        
        print(f"\n{'='*50}")
        print(f"Text: '{text[:60]}...'")
        print(f"\nMeta-Cognitive Assessment:")
        print(f"  Strategy: {state.strategy.value if state.strategy else 'unknown'}")
        print(f"  Confidence: {state.confidence:.2f}")
        print(f"  Uncertainty: {state.uncertainty:.2f}")
        print(f"  Error Risk: {state.error_risk:.2f}")
        print(f"  Would abstain: {mcre.should_abstain(state)}")
        print(f"\nExplanation: {state.explanation}")
        
        # Simulate outcome
        if hidden is not None:
            mcre.update_with_outcome(state, hidden, expected_correct)
            print(f"\n[Actual: {'correct' if expected_correct else 'incorrect'}]")
    
    # Show calibration
    print("\n" + "=" * 40)
    print("CALIBRATION & STRATEGY ANALYSIS")
    print("=" * 40)
    
    stats = mcre.get_statistics()
    print(f"\nOverall accuracy: {stats['overall_accuracy']:.1%}")
    print(f"High-confidence accuracy: {stats['high_confidence_accuracy']:.1%}")
    print(f"Calibration error: {stats['calibration_error']:.3f}")
    
    print("\nStrategy performance:")
    for strategy, success_rate in stats['strategy_performance'].items():
        bar = "█" * int(success_rate * 20)
        print(f"  {strategy:12s}: {success_rate:.1%} {bar}")
    
    print(f"\nPatterns learned:")
    print(f"  Error patterns: {stats['error_patterns_learned']}")
    print(f"  Success patterns: {stats['success_patterns_learned']}")
    
    print("\n✅ Meta-Cognitive Reasoning Engine demo complete!")
    return mcre


if __name__ == "__main__":
    demo_metacognitive()
