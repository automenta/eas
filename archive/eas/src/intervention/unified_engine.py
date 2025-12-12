#!/usr/bin/env python3
"""
Unified Reasoning Engine (URE)

NOVEL INNOVATION: Combines all EAS innovations into a single
powerful reasoning enhancement system.

Integrates:
1. Circuit Discovery - identify which components matter
2. Compositional Logic Grounding - primitive-specific attractors
3. Adaptive Reasoning Amplifier - real-time quality steering
4. Position-Aware Intervention - CTD-based targeting

This creates a complete reasoning enhancement pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

import sys
sys.path.insert(0, '/home/me/eas')


@dataclass
class ReasoningMetrics:
    """Comprehensive reasoning quality metrics."""
    logic_coherence: float = 0.0      # How well logic structure is maintained
    direction_quality: float = 0.0     # Alignment with correct reasoning direction
    circuit_activation: float = 0.0    # How strongly reasoning circuits are activated
    position_divergence: float = 0.0   # CTD-based divergence at critical positions
    overall_score: float = 0.0         # Combined score
    
    def compute_overall(self):
        """Compute weighted overall score."""
        self.overall_score = (
            0.3 * self.logic_coherence +
            0.3 * self.direction_quality +
            0.2 * self.circuit_activation +
            0.2 * self.position_divergence
        )


class UnifiedReasoningEngine:
    """
    Complete reasoning enhancement system.
    
    This is the production-ready integration of all EAS innovations,
    designed for both research and practical deployment.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        
        from eas.src.intervention.adaptive_reasoning import AdaptiveReasoningAmplifier
        from eas.src.intervention.compositional_logic import CompositionalLogicIntervention
        from eas.src.intervention.circuit_discovery import ReasoningCircuitDiscovery
        from eas.src.watcher.position_aware_watcher import PositionAwareWatcher
        
        # Initialize components
        self.amplifier = AdaptiveReasoningAmplifier(dim=model.d_model)
        self.logic_grounder = CompositionalLogicIntervention(dim=model.d_model)
        self.circuit_discoverer = ReasoningCircuitDiscovery(model)
        self.position_watcher = PositionAwareWatcher(
            dim=model.d_model,
            tokenizer=model.tokenizer
        )
        
        # Discovered circuits (populated after learning)
        self.reasoning_circuits = []
        self.is_calibrated = False
        
        # Intervention configuration
        self.config = {
            'use_circuit_targeting': True,
            'use_logic_grounding': True,
            'use_adaptive_steering': True,
            'use_position_awareness': True,
            'base_alpha': 0.3,
            'intervention_layers': [2, 3, 4],  # Middle layers by default
        }
        
        # Statistics
        self.stats = {
            'examples_learned': 0,
            'interventions_applied': 0,
            'avg_quality_improvement': []
        }
    
    def calibrate(self, correct_examples: List[str], incorrect_examples: List[str]):
        """
        Calibrate the engine from examples.
        
        This is the training phase where the engine learns:
        - What correct reasoning looks like
        - Which circuits are important for reasoning
        - How to detect logical structure
        """
        print("Calibrating Unified Reasoning Engine...")
        
        # 1. Learn correct/incorrect directions
        print("  Learning reasoning directions...")
        for text in correct_examples:
            input_ids = self.model.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                self.model.forward(input_ids.input_ids.to(self.device))
                hidden = self.model.get_layer_activation(3)
            
            if hidden is not None:
                self.amplifier.learn_from_example(hidden, is_correct=True)
                self.logic_grounder.learn_from_example(text, hidden, is_correct=True)
                self.position_watcher.update(hidden, input_ids.input_ids)
        
        for text in incorrect_examples:
            input_ids = self.model.tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                self.model.forward(input_ids.input_ids.to(self.device))
                hidden = self.model.get_layer_activation(3)
            
            if hidden is not None:
                self.amplifier.learn_from_example(hidden, is_correct=False)
        
        # 2. Discover reasoning circuits
        print("  Discovering reasoning circuits...")
        pairs = [
            {'correct': c, 'incorrect': i}
            for c, i in zip(correct_examples[:min(5, len(correct_examples))],
                          incorrect_examples[:min(5, len(incorrect_examples))])
        ]
        if pairs:
            self.reasoning_circuits = self.circuit_discoverer.discover_circuits(pairs)
            
            # Update intervention layers based on discovered circuits
            if self.reasoning_circuits:
                top_layers = list(set(c.layer_idx for c in self.reasoning_circuits[:3]))
                if top_layers:
                    self.config['intervention_layers'] = top_layers
        
        self.stats['examples_learned'] = len(correct_examples) + len(incorrect_examples)
        self.is_calibrated = True
        
        print(f"  Calibration complete!")
        print(f"    Learned from {len(correct_examples)} correct, {len(incorrect_examples)} incorrect examples")
        print(f"    Discovered {len(self.reasoning_circuits)} circuit components")
        print(f"    Intervention layers: {self.config['intervention_layers']}")
    
    def compute_metrics(self, text: str, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> ReasoningMetrics:
        """Compute comprehensive reasoning metrics."""
        metrics = ReasoningMetrics()
        
        # 1. Logic coherence from compositional analysis
        structure = self.logic_grounder.parser.parse(text)
        metrics.logic_coherence = structure.confidence
        
        # 2. Direction quality from amplifier
        if self.amplifier.correct_direction.norm() > 0:
            quality, _ = self.amplifier.compute_reasoning_quality(hidden_states)
            metrics.direction_quality = max(0, quality)
        
        # 3. Circuit activation (how different are activations at key layers)
        if self.reasoning_circuits:
            # Use the causal effect as a proxy for circuit activation
            metrics.circuit_activation = min(1.0, 
                self.reasoning_circuits[0].causal_effect / 5.0
            ) if self.reasoning_circuits else 0.0
        
        # 4. Position divergence (from position-aware analysis)
        weights = self.position_watcher.position_detector.detect_critical_positions(input_ids)
        critical_count = (weights > 1.0).sum().item()
        total_count = weights.numel()
        metrics.position_divergence = critical_count / max(1, total_count)
        
        metrics.compute_overall()
        return metrics
    
    def enhance(self, text: str, layer_idx: int = None) -> Tuple[torch.Tensor, ReasoningMetrics]:
        """
        Enhance reasoning for the given text.
        
        This is the main inference-time function that applies all
        innovations to improve reasoning quality.
        
        Returns enhanced hidden states and quality metrics.
        """
        if not self.is_calibrated:
            raise RuntimeError("Engine not calibrated. Call calibrate() first.")
        
        # Tokenize
        input_ids = self.model.tokenizer(text, return_tensors="pt").input_ids.to(self.device)
        
        # Get base hidden states
        if layer_idx is None:
            layer_idx = self.config['intervention_layers'][0]
        
        with torch.no_grad():
            self.model.forward(input_ids)
            original_hidden = self.model.get_layer_activation(layer_idx)
        
        if original_hidden is None:
            return None, ReasoningMetrics()
        
        enhanced = original_hidden.clone()
        
        # Apply interventions in order
        
        # 1. Position-aware snapping
        if self.config['use_position_awareness']:
            enhanced = self.position_watcher.snap(enhanced, input_ids)
        
        # 2. Compositional logic grounding
        if self.config['use_logic_grounding']:
            enhanced = self.logic_grounder.intervene(text, enhanced)
        
        # 3. Adaptive reasoning steering
        if self.config['use_adaptive_steering']:
            enhanced = self.amplifier.adaptive_steer(enhanced)
        
        # Compute metrics
        metrics = self.compute_metrics(text, enhanced, input_ids)
        
        self.stats['interventions_applied'] += 1
        
        return enhanced, metrics
    
    def create_intervention_hook(self) -> callable:
        """Create a hook that can be registered for live intervention."""
        
        def hook(hidden_states: torch.Tensor) -> torch.Tensor:
            # Note: We don't have text here, so we use position-only approach
            # In practice, you'd pass the input_ids through a closure
            
            # Apply adaptive steering
            if self.config['use_adaptive_steering']:
                hidden_states = self.amplifier.adaptive_steer(hidden_states)
            
            return hidden_states
        
        return hook
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        return {
            'is_calibrated': self.is_calibrated,
            'examples_learned': self.stats['examples_learned'],
            'interventions_applied': self.stats['interventions_applied'],
            'config': self.config,
            'num_discovered_circuits': len(self.reasoning_circuits),
            'top_circuits': [
                {'layer': c.layer_idx, 'type': c.component_type, 'effect': c.causal_effect}
                for c in self.reasoning_circuits[:5]
            ],
            'amplifier_stats': self.amplifier.get_statistics(),
            'logic_stats': self.logic_grounder.get_statistics(),
            'position_stats': self.position_watcher.get_statistics()
        }
    
    def save(self, path: str):
        """Save engine state."""
        state = {
            'config': self.config,
            'stats': self.stats,
            'is_calibrated': self.is_calibrated,
            'circuits': [
                {'layer': c.layer_idx, 'type': c.component_type, 
                 'effect': c.causal_effect, 'direction': c.direction}
                for c in self.reasoning_circuits
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved engine state to {path}")
    
    def load(self, path: str):
        """Load engine state."""
        with open(path) as f:
            state = json.load(f)
        
        self.config = state['config']
        self.stats = state['stats']
        self.is_calibrated = state['is_calibrated']
        
        from eas.src.intervention.circuit_discovery import CircuitComponent
        self.reasoning_circuits = [
            CircuitComponent(**c) for c in state['circuits']
        ]
        
        print(f"Loaded engine state from {path}")


def demo_unified_engine():
    """Demonstrate the Unified Reasoning Engine."""
    print("=" * 60)
    print("UNIFIED REASONING ENGINE (URE) DEMO")
    print("=" * 60)
    
    from eas.src.models.transformer import PretrainedTransformer
    
    print("\nLoading model...")
    model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
    
    print("\nInitializing Unified Reasoning Engine...")
    engine = UnifiedReasoningEngine(model)
    
    # Calibration examples
    correct_examples = [
        "If it rains, the ground gets wet. It is raining. Therefore, the ground is wet.",
        "All mammals are warm-blooded. Dogs are mammals. Thus, dogs are warm-blooded.",
        "Every bird has feathers. Sparrows are birds. Hence, sparrows have feathers.",
        "2 + 2 = 4. This is correct arithmetic.",
        "Fire requires oxygen to burn. There is oxygen here. So fire can burn here.",
    ]
    
    incorrect_examples = [
        "If it rains, the ground gets wet. It is raining. Therefore, the ground is dry.",
        "All mammals are warm-blooded. Dogs are mammals. Thus, dogs are cold-blooded.",
        "Every bird has feathers. Sparrows are birds. Hence, sparrows have scales.",
        "2 + 2 = 5. This is correct arithmetic.",
        "Fire requires oxygen to burn. There is oxygen here. So fire freezes.",
    ]
    
    # Calibrate
    engine.calibrate(correct_examples, incorrect_examples)
    
    # Test enhancement
    print("\n" + "=" * 40)
    print("TESTING REASONING ENHANCEMENT")
    print("=" * 40)
    
    test_cases = [
        "All cats are animals. Whiskers is a cat. Therefore, Whiskers is an animal.",
        "If study then pass. I studied. Hence I passed.",
        "Every fruit has seeds. An apple is a fruit. Thus apples have seeds.",
    ]
    
    for text in test_cases:
        enhanced, metrics = engine.enhance(text)
        
        print(f"\nText: '{text[:50]}...'")
        print(f"  Logic coherence:     {metrics.logic_coherence:.3f}")
        print(f"  Direction quality:   {metrics.direction_quality:.3f}")
        print(f"  Circuit activation:  {metrics.circuit_activation:.3f}")
        print(f"  Position divergence: {metrics.position_divergence:.3f}")
        print(f"  ───────────────────────────────")
        print(f"  OVERALL SCORE:       {metrics.overall_score:.3f}")
    
    # Show comprehensive statistics
    print("\n" + "=" * 40)
    print("ENGINE STATISTICS")
    print("=" * 40)
    
    stats = engine.get_statistics()
    print(f"\nCalibrated: {stats['is_calibrated']}")
    print(f"Examples learned: {stats['examples_learned']}")
    print(f"Interventions applied: {stats['interventions_applied']}")
    print(f"\nDiscovered {stats['num_discovered_circuits']} reasoning circuits:")
    for circuit in stats['top_circuits'][:3]:
        print(f"  Layer {circuit['layer']} ({circuit['type']}): effect={circuit['effect']:.3f}")
    
    # Save engine state
    save_path = "/home/me/eas/eas/analysis/results/ure_state.json"
    engine.save(save_path)
    
    print("\n✅ Unified Reasoning Engine demo complete!")
    return engine


if __name__ == "__main__":
    demo_unified_engine()
