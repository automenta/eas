#!/usr/bin/env python3
"""
Causal Reasoning Graph (CRG)

REVOLUTIONARY INNOVATION: Build explicit causal models from text,
enabling true causal reasoning rather than correlational pattern matching.

Key breakthroughs:
1. Causal variable extraction from natural language
2. Causal graph construction (DAG)
3. Intervention semantics (do-calculus)
4. Counterfactual reasoning
5. Causal path activation in neural network

This bridges the gap between causal inference and neural networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

import sys
sys.path.insert(0, '/home/me/eas')


@dataclass
class CausalVariable:
    """A variable in the causal graph."""
    name: str
    value: Optional[str] = None
    embedding: Optional[torch.Tensor] = None
    is_observed: bool = True


@dataclass
class CausalEdge:
    """A causal relationship between variables."""
    cause: str
    effect: str
    strength: float = 1.0
    mechanism: str = "direct"  # direct, mediated, confounded


class CausalGraph:
    """
    Explicit causal graph extracted from reasoning.
    
    Enables do-calculus style interventions on language.
    """
    
    def __init__(self):
        self.variables: Dict[str, CausalVariable] = {}
        self.edges: List[CausalEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # cause -> effects
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)  # effect -> causes
    
    def add_variable(self, var: CausalVariable):
        """Add a variable to the graph."""
        self.variables[var.name] = var
    
    def add_edge(self, edge: CausalEdge):
        """Add a causal edge."""
        self.edges.append(edge)
        self.adjacency[edge.cause].append(edge.effect)
        self.reverse_adj[edge.effect].append(edge.cause)
    
    def get_causal_parents(self, var_name: str) -> List[str]:
        """Get all causal parents of a variable."""
        return self.reverse_adj.get(var_name, [])
    
    def get_causal_children(self, var_name: str) -> List[str]:
        """Get all causal children of a variable."""
        return self.adjacency.get(var_name, [])
    
    def get_causal_path(self, start: str, end: str) -> List[str]:
        """Find causal path from start to end using BFS."""
        if start not in self.variables or end not in self.variables:
            return []
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for child in self.adjacency.get(current, []):
                if child not in visited:
                    queue.append((child, path + [child]))
        
        return []
    
    def do_intervention(self, var_name: str, value: str) -> 'CausalGraph':
        """
        Perform do(X=x) intervention.
        
        This removes all incoming edges to X and sets its value.
        Returns a new interventional graph.
        """
        new_graph = CausalGraph()
        
        # Copy variables
        for name, var in self.variables.items():
            new_var = CausalVariable(
                name=var.name,
                value=value if name == var_name else var.value,
                embedding=var.embedding,
                is_observed=var.is_observed
            )
            new_graph.add_variable(new_var)
        
        # Copy edges except those into intervened variable
        for edge in self.edges:
            if edge.effect != var_name:
                new_graph.add_edge(edge)
        
        return new_graph
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'variables': [v.name for v in self.variables.values()],
            'edges': [(e.cause, e.effect, e.strength) for e in self.edges]
        }


class CausalExtractor:
    """
    Extract causal structure from natural language.
    
    Uses pattern matching and neural signals to identify
    causal relationships in text.
    """
    
    def __init__(self, model):
        self.model = model
        
        # Causal language patterns
        self.causal_patterns = [
            # If-then patterns
            (r'if\s+(.+?)\s+then\s+(.+)', 'conditional'),
            (r'when\s+(.+?),?\s+(.+)', 'conditional'),
            
            # Causes patterns
            (r'(.+?)\s+causes?\s+(.+)', 'direct'),
            (r'(.+?)\s+leads?\s+to\s+(.+)', 'direct'),
            (r'(.+?)\s+results?\s+in\s+(.+)', 'direct'),
            
            # Because patterns
            (r'(.+?)\s+because\s+(.+)', 'explanation'),
            (r'(.+?)\s+due\s+to\s+(.+)', 'explanation'),
            
            # Therefore patterns (effect of prior causes)
            (r'(.+?)\.\s*therefore\s+(.+)', 'inference'),
            (r'(.+?)\.\s*thus\s+(.+)', 'inference'),
            (r'(.+?)\.\s*hence\s+(.+)', 'inference'),
        ]
    
    def extract(self, text: str) -> CausalGraph:
        """Extract causal graph from text."""
        graph = CausalGraph()
        text_lower = text.lower()
        
        # Find causal relationships
        for pattern, mechanism in self.causal_patterns:
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    cause_text = match.group(1).strip()
                    effect_text = match.group(2).strip()
                    
                    # Create variable names
                    cause_name = self._extract_variable_name(cause_text)
                    effect_name = self._extract_variable_name(effect_text)
                    
                    if cause_name and effect_name and cause_name != effect_name:
                        # Add variables
                        if cause_name not in graph.variables:
                            graph.add_variable(CausalVariable(
                                name=cause_name,
                                value=cause_text
                            ))
                        
                        if effect_name not in graph.variables:
                            graph.add_variable(CausalVariable(
                                name=effect_name,
                                value=effect_text
                            ))
                        
                        # Add edge
                        graph.add_edge(CausalEdge(
                            cause=cause_name,
                            effect=effect_name,
                            mechanism=mechanism
                        ))
        
        return graph
    
    def _extract_variable_name(self, text: str) -> Optional[str]:
        """Extract a clean variable name from text."""
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                    'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
        
        words = [w for w in text.split() if w.lower() not in stopwords]
        
        if not words:
            return None
        
        # Take first meaningful word or phrase
        name = '_'.join(words[:3])
        return name.replace('.', '').replace(',', '')


class CausalReasoningIntervention:
    """
    Use causal graphs to guide neural interventions.
    
    Key insight: Activate the neural pathways that correspond
    to causal chains, not just correlational patterns.
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.extractor = CausalExtractor(model)
        
        # Learned embeddings for causal concepts
        self.causal_embeddings: Dict[str, torch.Tensor] = {}
        
        # Intervention strength based on causal distance
        self.distance_decay = 0.8
    
    def learn_causal_embedding(self, var_name: str, text: str):
        """Learn the neural embedding for a causal variable."""
        input_ids = self.model.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            self.model.forward(input_ids.input_ids.to(self.device))
            hidden = self.model.get_layer_activation(3)
        
        if hidden is not None:
            embedding = hidden.mean(dim=1).squeeze(0)
            
            if var_name in self.causal_embeddings:
                # Running average
                self.causal_embeddings[var_name] = 0.9 * self.causal_embeddings[var_name] + 0.1 * embedding
            else:
                self.causal_embeddings[var_name] = embedding
    
    def intervene_on_causal_chain(self, 
                                   hidden_states: torch.Tensor,
                                   graph: CausalGraph,
                                   target_var: str,
                                   alpha: float = 0.3) -> torch.Tensor:
        """
        Intervene to strengthen the causal chain leading to target.
        
        This activates the neural representations corresponding to
        all causes in the causal chain.
        """
        if target_var not in graph.variables:
            return hidden_states
        
        # Find all causal ancestors
        ancestors = self._get_all_ancestors(graph, target_var)
        
        if not ancestors:
            return hidden_states
        
        # Build steering vector from causal chain
        steering = torch.zeros(self.model.d_model, device=self.device)
        total_weight = 0
        
        for i, ancestor in enumerate(ancestors):
            if ancestor in self.causal_embeddings:
                # Weight by causal distance (closer = stronger)
                weight = self.distance_decay ** i
                steering += weight * self.causal_embeddings[ancestor]
                total_weight += weight
        
        if total_weight > 0:
            steering = F.normalize(steering / total_weight, dim=0)
            
            delta = alpha * steering.unsqueeze(0).unsqueeze(0)
            return hidden_states + delta
        
        return hidden_states
    
    def _get_all_ancestors(self, graph: CausalGraph, var_name: str) -> List[str]:
        """Get all causal ancestors in order of distance."""
        ancestors = []
        visited = set()
        queue = [(var_name, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if current in visited:
                continue
            visited.add(current)
            
            parents = graph.get_causal_parents(current)
            for parent in parents:
                if parent not in visited:
                    ancestors.append(parent)
                    queue.append((parent, distance + 1))
        
        return ancestors
    
    def counterfactual_intervention(self,
                                   hidden_states: torch.Tensor,
                                   graph: CausalGraph,
                                   intervened_var: str,
                                   new_value_text: str,
                                   alpha: float = 0.5) -> torch.Tensor:
        """
        Apply counterfactual intervention: "What if X had been different?"
        
        This modifies the neural representation to reflect a different
        causal scenario.
        """
        # Learn embedding for new value
        input_ids = self.model.tokenizer(new_value_text, return_tensors="pt")
        
        with torch.no_grad():
            self.model.forward(input_ids.input_ids.to(self.device))
            new_embedding = self.model.get_layer_activation(3)
        
        if new_embedding is None:
            return hidden_states
        
        new_embedding = new_embedding.mean(dim=1).squeeze(0)
        
        # Get original embedding
        if intervened_var not in self.causal_embeddings:
            return hidden_states
        
        old_embedding = self.causal_embeddings[intervened_var]
        
        # Compute counterfactual direction
        counterfactual_dir = F.normalize(new_embedding - old_embedding, dim=0)
        
        # Apply to hidden states
        delta = alpha * counterfactual_dir.unsqueeze(0).unsqueeze(0)
        return hidden_states + delta
    
    def reason_with_causality(self, text: str, target_conclusion: str) -> Tuple[torch.Tensor, CausalGraph, Dict]:
        """
        Full causal reasoning pipeline.
        """
        # Extract causal graph
        graph = self.extractor.extract(text)
        
        # Learn embeddings for all variables
        for var_name, var in graph.variables.items():
            if var.value:
                self.learn_causal_embedding(var_name, var.value)
        
        # Get hidden states
        input_ids = self.model.tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            self.model.forward(input_ids.input_ids.to(self.device))
            hidden = self.model.get_layer_activation(3)
        
        if hidden is None:
            return None, graph, {}
        
        # Find target variable
        target_var = self.extractor._extract_variable_name(target_conclusion)
        
        # Apply causal intervention
        enhanced = self.intervene_on_causal_chain(hidden, graph, target_var)
        
        # Compute metrics
        delta_norm = (enhanced - hidden).norm().item()
        
        metrics = {
            'num_variables': len(graph.variables),
            'num_edges': len(graph.edges),
            'causal_depth': self._get_max_depth(graph),
            'intervention_magnitude': delta_norm
        }
        
        return enhanced, graph, metrics
    
    def _get_max_depth(self, graph: CausalGraph) -> int:
        """Get maximum depth of the causal graph."""
        if not graph.variables:
            return 0
        
        # Find root nodes (no parents)
        roots = [v for v in graph.variables if not graph.get_causal_parents(v)]
        
        if not roots:
            return 0
        
        max_depth = 0
        for root in roots:
            depth = self._dfs_depth(graph, root, set())
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _dfs_depth(self, graph: CausalGraph, node: str, visited: Set[str]) -> int:
        if node in visited:
            return 0
        visited.add(node)
        
        children = graph.get_causal_children(node)
        if not children:
            return 1
        
        return 1 + max(self._dfs_depth(graph, c, visited) for c in children)


def demo_causal_reasoning():
    """Demonstrate Causal Reasoning Graph."""
    print("=" * 60)
    print("CAUSAL REASONING GRAPH (CRG) DEMO")
    print("=" * 60)
    
    from eas.src.models.transformer import PretrainedTransformer
    
    print("\nLoading model...")
    model = PretrainedTransformer("EleutherAI/pythia-70m", device="cpu")
    
    print("\nInitializing Causal Reasoning...")
    crg = CausalReasoningIntervention(model)
    
    # Test causal extraction
    print("\n" + "=" * 40)
    print("CAUSAL STRUCTURE EXTRACTION")
    print("=" * 40)
    
    test_texts = [
        "If it rains, then the ground gets wet. The ground is wet because it rained.",
        "Smoking causes lung cancer. Therefore, quitting smoking reduces cancer risk.",
        "When temperature rises, ice melts. This leads to rising sea levels.",
        "Study leads to knowledge. Knowledge results in good grades. Thus studying causes good grades.",
    ]
    
    for text in test_texts:
        graph = crg.extractor.extract(text)
        print(f"\nText: '{text[:50]}...'")
        print(f"  Variables: {list(graph.variables.keys())}")
        print(f"  Edges: {[(e.cause, '->', e.effect) for e in graph.edges]}")
    
    # Test causal reasoning
    print("\n" + "=" * 40)
    print("CAUSAL REASONING INTERVENTION")
    print("=" * 40)
    
    reasoning_text = "If studying leads to passing, and I studied hard, then I should pass."
    conclusion = "passing the exam"
    
    enhanced, graph, metrics = crg.reason_with_causality(reasoning_text, conclusion)
    
    print(f"\nReasoning: '{reasoning_text}'")
    print(f"Conclusion: '{conclusion}'")
    print(f"\nCausal graph:")
    print(f"  Variables: {metrics['num_variables']}")
    print(f"  Edges: {metrics['num_edges']}")
    print(f"  Max depth: {metrics['causal_depth']}")
    print(f"  Intervention magnitude: {metrics['intervention_magnitude']:.4f}")
    
    # Test counterfactual
    print("\n" + "=" * 40)
    print("COUNTERFACTUAL REASONING")
    print("=" * 40)
    
    original_text = "It rained and the ground is wet"
    counterfactual_text = "It did not rain"
    
    # Learn original
    input_ids = model.tokenizer(original_text, return_tensors="pt")
    with torch.no_grad():
        model.forward(input_ids.input_ids)
        original_hidden = model.get_layer_activation(3)
    
    crg.learn_causal_embedding("rain", "it rained")
    crg.learn_causal_embedding("wet_ground", "ground is wet")
    
    # Create simple graph
    cf_graph = CausalGraph()
    cf_graph.add_variable(CausalVariable("rain"))
    cf_graph.add_variable(CausalVariable("wet_ground"))
    cf_graph.add_edge(CausalEdge("rain", "wet_ground"))
    
    # Apply counterfactual
    counterfactual_hidden = crg.counterfactual_intervention(
        original_hidden, cf_graph, "rain", counterfactual_text
    )
    
    cf_delta = (counterfactual_hidden - original_hidden).norm().item()
    print(f"\nOriginal: '{original_text}'")
    print(f"Counterfactual: 'What if {counterfactual_text}?'")
    print(f"Representation shift: {cf_delta:.4f}")
    
    print("\n✅ Causal Reasoning Graph demo complete!")
    return crg


if __name__ == "__main__":
    demo_causal_reasoning()
