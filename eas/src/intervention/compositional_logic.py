#!/usr/bin/env python3
"""
Compositional Logic Grounding (CLG)

NOVEL INNOVATION: Decompose reasoning into atomic operations and 
learn grounded representations for each.

Key innovations:
1. Logic primitives extraction (implication, conjunction, negation, etc.)
2. Grounded attractor learning for each primitive
3. Compositional intervention that respects logical structure
4. Symbolic-neural bridge for interpretable steering

This creates a neuro-symbolic system where interventions have 
clear logical semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import re

import sys
sys.path.insert(0, '/home/me/eas')


class LogicPrimitive(Enum):
    """Atomic logical operations."""
    IMPLICATION = "implication"      # A → B
    CONJUNCTION = "conjunction"      # A ∧ B  
    DISJUNCTION = "disjunction"      # A ∨ B
    NEGATION = "negation"            # ¬A
    UNIVERSAL = "universal"          # ∀x
    EXISTENTIAL = "existential"      # ∃x
    MODUS_PONENS = "modus_ponens"    # A, A→B ⊢ B
    MODUS_TOLLENS = "modus_tollens"  # ¬B, A→B ⊢ ¬A
    TRANSITIVITY = "transitivity"    # A→B, B→C ⊢ A→C


@dataclass
class LogicStructure:
    """Parsed logical structure of a statement."""
    primitives: List[LogicPrimitive]
    variables: List[str]
    propositions: List[str]
    is_valid: bool
    confidence: float


class LogicParser:
    """
    Parse natural language into logical structure.
    
    This bridges the symbolic-neural gap by identifying
    what logical operations are being performed.
    """
    
    def __init__(self):
        # Pattern matchers for logical structures
        self.patterns = {
            LogicPrimitive.IMPLICATION: [
                r'if\s+(.+?)\s+then\s+(.+)',
                r'(.+?)\s+implies\s+(.+)',
                r'when\s+(.+?),?\s+(.+)',
            ],
            LogicPrimitive.MODUS_PONENS: [
                r'(.+?)\.?\s+therefore\s+(.+)',
                r'(.+?)\.?\s+thus\s+(.+)',
                r'(.+?)\.?\s+hence\s+(.+)',
                r'(.+?)\.?\s+so\s+(.+)',
            ],
            LogicPrimitive.UNIVERSAL: [
                r'all\s+(\w+)\s+are\s+(.+)',
                r'every\s+(\w+)\s+is\s+(.+)',
                r'each\s+(\w+)\s+has\s+(.+)',
            ],
            LogicPrimitive.CONJUNCTION: [
                r'(.+)\s+and\s+(.+)',
                r'both\s+(.+)\s+and\s+(.+)',
            ],
            LogicPrimitive.NEGATION: [
                r'not\s+(.+)',
                r'(.+)\s+is\s+false',
                r'(.+)\s+is\s+incorrect',
            ],
        }
    
    def parse(self, text: str) -> LogicStructure:
        """Parse text into logical structure."""
        text_lower = text.lower()
        found_primitives = []
        propositions = []
        variables = []
        
        for primitive, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    found_primitives.append(primitive)
                    propositions.extend(match.groups())
                    break
        
        # Extract potential variables (single letters or short words)
        for word in re.findall(r'\b[A-Z]\b|\b\w{1,2}\b', text):
            if word not in variables:
                variables.append(word)
        
        # Determine if reasoning appears valid
        # Simple heuristic: modus ponens chains should be consistent
        is_valid = True
        if LogicPrimitive.MODUS_PONENS in found_primitives:
            # Check for contradiction markers
            if 'but' in text_lower or 'however' in text_lower:
                is_valid = False
        
        confidence = len(found_primitives) / 3.0  # Rough confidence
        confidence = min(1.0, confidence)
        
        return LogicStructure(
            primitives=found_primitives,
            variables=variables,
            propositions=propositions,
            is_valid=is_valid,
            confidence=confidence
        )


class GroundedPrimitiveMemory(nn.Module):
    """
    Stores grounded neural representations for each logic primitive.
    
    Each primitive has associated attractors that represent
    "what correct use of this primitive looks like" in neural space.
    """
    
    def __init__(self, dim: int, k_per_primitive: int = 5):
        super().__init__()
        self.dim = dim
        self.k = k_per_primitive
        
        # One set of attractors per primitive
        self.primitive_attractors = nn.ParameterDict()
        
        for primitive in LogicPrimitive:
            self.primitive_attractors[primitive.value] = nn.Parameter(
                torch.randn(k_per_primitive, dim)
            )
        
        # Usage statistics
        self.usage_counts = {p.value: torch.zeros(k_per_primitive) for p in LogicPrimitive}
        
        self._normalize_all()
    
    def _normalize_all(self):
        """Normalize all attractors to unit sphere."""
        with torch.no_grad():
            for key in self.primitive_attractors:
                self.primitive_attractors[key].data = F.normalize(
                    self.primitive_attractors[key].data, dim=-1
                )
    
    def get_attractors(self, primitive: LogicPrimitive) -> torch.Tensor:
        """Get attractors for a specific primitive."""
        return self.primitive_attractors[primitive.value]
    
    def update_from_example(self, primitive: LogicPrimitive, 
                           activation: torch.Tensor,
                           momentum: float = 0.1):
        """Update attractors for a primitive from a successful example."""
        with torch.no_grad():
            attractors = self.primitive_attractors[primitive.value]
            activation_norm = F.normalize(activation, dim=-1)
            
            # Find nearest attractor
            sims = torch.mm(activation_norm.unsqueeze(0), attractors.t())
            nearest_idx = sims.argmax().item()
            
            # Update with momentum
            attractors[nearest_idx] = (1 - momentum) * attractors[nearest_idx] + \
                                      momentum * activation_norm
            attractors[nearest_idx] = F.normalize(attractors[nearest_idx], dim=0)
            
            self.usage_counts[primitive.value][nearest_idx] += 1


class CompositionalLogicIntervention(nn.Module):
    """
    Apply interventions that respect logical structure.
    
    Key innovation: Instead of uniform steering, we apply
    primitive-specific attractors based on the detected
    logical structure of the input.
    """
    
    def __init__(self, dim: int, k_per_primitive: int = 5, alpha: float = 0.3):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        
        self.parser = LogicParser()
        self.memory = GroundedPrimitiveMemory(dim, k_per_primitive)
        
        # Statistics
        self.intervention_counts = {p.value: 0 for p in LogicPrimitive}
        self.total_interventions = 0
    
    def learn_from_example(self, text: str, hidden_states: torch.Tensor, is_correct: bool):
        """
        Learn grounded representations from an example.
        
        Only updates from correct examples - we want attractors
        to represent "correct" usage of each primitive.
        """
        if not is_correct:
            return
        
        structure = self.parser.parse(text)
        
        # Pool hidden states
        pooled = hidden_states.mean(dim=1).squeeze(0)  # [dim]
        
        # Update attractors for each detected primitive
        for primitive in structure.primitives:
            self.memory.update_from_example(primitive, pooled)
    
    def intervene(self, text: str, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply compositional intervention based on logical structure.
        
        This is the key innovation: we steer toward the right
        attractors for the specific logical operations being performed.
        """
        structure = self.parser.parse(text)
        
        if not structure.primitives:
            return hidden_states  # No recognized structure, don't intervene
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        modified = hidden_states.clone()
        
        # Compute composite steering direction from relevant primitives
        steering_directions = []
        
        for primitive in structure.primitives:
            attractors = self.memory.get_attractors(primitive)
            
            # Find nearest attractor for current state
            pooled = hidden_states.mean(dim=1).squeeze(0)
            pooled_norm = F.normalize(pooled, dim=-1)
            
            sims = torch.mm(pooled_norm.unsqueeze(0), attractors.t())
            nearest_idx = sims.argmax().item()
            
            steering_directions.append(attractors[nearest_idx])
            self.intervention_counts[primitive.value] += 1
        
        self.total_interventions += 1
        
        # Combine steering directions (weighted by primitive importance)
        combined_steering = torch.stack(steering_directions).mean(dim=0)
        combined_steering = F.normalize(combined_steering, dim=-1)
        
        # Apply intervention with position weighting
        for i in range(seq_len):
            rel_pos = i / seq_len
            
            # Stronger intervention at conclusion positions
            pos_weight = 0.5 + 0.5 * rel_pos
            effective_alpha = self.alpha * pos_weight * structure.confidence
            
            current_norm = F.normalize(modified[0, i], dim=-1)
            delta = effective_alpha * (combined_steering - current_norm)
            
            modified[:, i, :] = hidden_states[:, i, :] + delta
        
        return modified
    
    def get_statistics(self) -> Dict:
        """Get intervention statistics by primitive."""
        return {
            'total_interventions': self.total_interventions,
            'by_primitive': dict(self.intervention_counts),
            'memory_usage': {
                p.value: self.memory.usage_counts[p.value].sum().item()
                for p in LogicPrimitive
            }
        }


def demo_clg():
    """Demonstrate Compositional Logic Grounding."""
    print("=" * 60)
    print("COMPOSITIONAL LOGIC GROUNDING (CLG) DEMO")
    print("=" * 60)
    
    from eas.src.models.transformer import PretrainedTransformer
    
    print("\nLoading model...")
    model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
    
    print("\nInitializing CLG...")
    clg = CompositionalLogicIntervention(dim=model.d_model)
    
    # Parse some examples
    print("\n" + "=" * 40)
    print("LOGICAL STRUCTURE PARSING")
    print("=" * 40)
    
    test_texts = [
        "If it rains then the ground is wet. It rained. Therefore the ground is wet.",
        "All birds have feathers. A sparrow is a bird. Thus sparrows have feathers.",
        "Every mammal is warm-blooded. Dogs are mammals. Hence dogs are warm-blooded.",
        "Not all statements are true. This is false.",
    ]
    
    for text in test_texts:
        structure = clg.parser.parse(text)
        print(f"\nText: '{text[:50]}...'")
        print(f"  Primitives: {[p.value for p in structure.primitives]}")
        print(f"  Confidence: {structure.confidence:.2f}")
        print(f"  Valid: {structure.is_valid}")
    
    # Learn from correct examples
    print("\n" + "=" * 40)
    print("LEARNING GROUNDED REPRESENTATIONS")
    print("=" * 40)
    
    correct_examples = [
        "If A then B. A is true. Therefore B is true.",
        "All dogs are mammals. Rex is a dog. Thus Rex is a mammal.",
        "Every bird has wings. Sparrows are birds. Hence sparrows have wings.",
        "If rain then wet. It rained. So the ground is wet.",
    ]
    
    for text in correct_examples:
        input_ids = model.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            model.forward(input_ids.input_ids)
            hidden = model.get_layer_activation(3)
        
        if hidden is not None:
            clg.learn_from_example(text, hidden, is_correct=True)
    
    print(f"Learned from {len(correct_examples)} correct examples")
    
    # Test intervention
    print("\n" + "=" * 40)
    print("COMPOSITIONAL INTERVENTION")
    print("=" * 40)
    
    test_text = "All cats are animals. Whiskers is a cat. Therefore Whiskers is"
    input_ids = model.tokenizer(test_text, return_tensors="pt")
    
    with torch.no_grad():
        model.forward(input_ids.input_ids)
        hidden = model.get_layer_activation(3)
    
    if hidden is not None:
        modified = clg.intervene(test_text, hidden)
        
        delta = (modified - hidden).norm(dim=-1)
        print(f"\nTest: '{test_text}'")
        print(f"Mean steering magnitude: {delta.mean().item():.4f}")
        print(f"Max steering magnitude: {delta.max().item():.4f}")
    
    # Show statistics
    print("\n" + "=" * 40)
    print("INTERVENTION STATISTICS")
    print("=" * 40)
    
    stats = clg.get_statistics()
    print(f"\nTotal interventions: {stats['total_interventions']}")
    print("\nBy primitive:")
    for primitive, count in stats['by_primitive'].items():
        if count > 0:
            print(f"  {primitive}: {count}")
    
    print("\n✅ CLG demonstration complete!")
    return clg


if __name__ == "__main__":
    demo_clg()
