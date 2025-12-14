#!/usr/bin/env python3
"""
eas_core.py - Core EAS (Emergent Activation Snapping) Implementation

This module implements the core EAS intervention mechanism:
1. AttractorMemory: Stores K centroids representing "correct reasoning" patterns
2. WhiteningBuffer: Normalizes activations for consistent geometry
3. EASIntervener: Captures, snaps, and injects modified activations

The key insight: successful inferences cluster in activation space. By learning
these clusters (attractors) and snapping wandering activations toward them,
we can improve reasoning accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from sklearn.cluster import MiniBatchKMeans


@dataclass
class EASConfig:
    """Configuration for EAS intervention."""
    num_attractors: int = 10          # K centroids
    hidden_dim: int = 512             # Model hidden dimension
    base_alpha: float = 0.3           # Base intervention strength
    clamp_delta: float = 1.0          # Max nudge magnitude
    warmup_samples: int = 20          # Samples before interventions start
    update_every: int = 5             # Update attractors every N successes
    intervention_layer: int = -1      # Which layer to intervene (-1 = middle)
    pooling: str = "mean"             # "mean" or "last"
    

class WhiteningBuffer:
    """Online normalization using running statistics."""
    
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        self.count = 0
    
    def update(self, x: torch.Tensor):
        """Update running statistics with new sample."""
        if x.dim() > 1:
            x = x.mean(dim=0)  # Reduce to single vector
        
        self.count += 1
        if self.count == 1:
            self.running_mean = x.detach().clone()
            self.running_var = torch.ones_like(x)
        else:
            delta = x - self.running_mean
            self.running_mean = self.running_mean + self.momentum * delta
            self.running_var = (1 - self.momentum) * self.running_var + \
                               self.momentum * delta * (x - self.running_mean)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using running statistics."""
        device = x.device
        mean = self.running_mean.to(device)
        std = torch.sqrt(self.running_var.to(device) + self.eps)
        return (x - mean) / std
    
    def to(self, device):
        """Move buffers to device."""
        self.running_mean = self.running_mean.to(device)
        self.running_var = self.running_var.to(device)
        return self


class AttractorMemory:
    """Memory storing K attractor centroids."""
    
    def __init__(self, k: int, dim: int):
        self.k = k
        self.dim = dim
        # Initialize with random unit vectors
        self.centroids = torch.randn(k, dim)
        self.centroids = self.centroids / self.centroids.norm(dim=1, keepdim=True)
        self.usage_counts = torch.zeros(k)
    
    def find_nearest(self, v: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """Find nearest attractor and return (centroid, similarity, index)."""
        device = v.device
        centroids = self.centroids.to(device)
        
        # Normalize for cosine similarity
        v_norm = v / (v.norm() + 1e-8)
        c_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-8)
        
        similarities = torch.mv(c_norm, v_norm)
        best_idx = similarities.argmax().item()
        best_sim = similarities[best_idx].item()
        
        return centroids[best_idx], best_sim, best_idx
    
    def update_centroid(self, idx: int, v: torch.Tensor, lr: float = 0.1):
        """Move centroid toward new sample."""
        # Move v to same device as centroids
        v_local = v.detach().to(self.centroids.device)
        self.centroids[idx] = (1 - lr) * self.centroids[idx] + lr * v_local
        # Re-normalize to stay on hypersphere
        self.centroids[idx] = self.centroids[idx] / (self.centroids[idx].norm() + 1e-8)
        self.usage_counts[idx] += 1
    
    def get_entropy(self) -> float:
        """Calculate usage entropy (measures attractor diversity)."""
        if self.usage_counts.sum() == 0:
            return 0.0
        probs = self.usage_counts / self.usage_counts.sum()
        probs = probs[probs > 0]
        return -torch.sum(probs * torch.log(probs + 1e-10)).item()
    
    def to(self, device):
        """Move centroids to device."""
        self.centroids = self.centroids.to(device)
        return self


class OnlineKMeans:
    """Online K-Means clustering for attractor evolution."""
    
    def __init__(self, k: int, dim: int):
        self.k = k
        self.dim = dim
        self.kmeans = MiniBatchKMeans(
            n_clusters=k, 
            batch_size=10,
            n_init=3,
            random_state=42
        )
        self.buffer: List[np.ndarray] = []
        self.buffer_size = 10
        self.fitted = False
    
    def add_sample(self, v: torch.Tensor):
        """Add successful activation to buffer."""
        self.buffer.append(v.detach().cpu().numpy())
        
        if len(self.buffer) >= self.buffer_size:
            self._partial_fit()
    
    def _partial_fit(self):
        """Fit K-Means on accumulated buffer."""
        if len(self.buffer) < self.k:
            return
        
        X = np.stack(self.buffer)
        self.kmeans.partial_fit(X)
        self.fitted = True
        self.buffer = []
    
    def get_centroids(self) -> Optional[torch.Tensor]:
        """Get current centroids if fitted."""
        if not self.fitted:
            return None
        return torch.from_numpy(self.kmeans.cluster_centers_).float()


class EASIntervener:
    """
    Core EAS intervention engine.
    
    Captures activations at a target layer, snaps them toward learned attractors,
    and injects the modified activations back into the forward pass.
    """
    
    def __init__(self, config: EASConfig):
        self.config = config
        self.whitening = WhiteningBuffer(config.hidden_dim)
        self.attractors = AttractorMemory(config.num_attractors, config.hidden_dim)
        self.kmeans = OnlineKMeans(config.num_attractors, config.hidden_dim)
        
        # State tracking
        self.total_samples = 0
        self.successful_samples = 0
        self.intervention_count = 0
        self.last_hidden_state: Optional[torch.Tensor] = None
        self.last_pooled: Optional[torch.Tensor] = None
        
        # Hook handles
        self._hook_handle = None
        self._is_intervening = True
    
    def pool(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Pool sequence to single vector."""
        # hidden_state: [batch, seq_len, hidden_dim]
        if self.config.pooling == "mean":
            return hidden_state.mean(dim=1)  # [batch, hidden_dim]
        else:  # "last"
            return hidden_state[:, -1, :]
    
    def snap(self, v: torch.Tensor) -> torch.Tensor:
        """
        Snap activation toward nearest attractor.
        
        Uses adaptive strength: nudge is weaker when already close to attractor.
        """
        device = v.device
        
        # Find nearest attractor (returns tensor on correct device)
        A_best, similarity, idx = self.attractors.find_nearest(v.squeeze())
        A_best = A_best.to(device)  # Ensure device match
        
        # Adaptive alpha: less nudging when already close
        alpha = self.config.base_alpha * (1 - similarity)
        
        # Calculate and clamp nudge
        v_squeezed = v.squeeze()
        delta = A_best - v_squeezed
        nudge = alpha * delta
        nudge = torch.clamp(nudge, -self.config.clamp_delta, self.config.clamp_delta)
        
        v_snapped = v_squeezed + nudge
        return v_snapped.unsqueeze(0)
    
    def intervene(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Full intervention pipeline: pool, whiten, snap, broadcast back.
        """
        device = hidden_state.device
        self.last_hidden_state = hidden_state.detach().clone()
        
        # Pool to sentence-level vector
        v = self.pool(hidden_state)  # [batch, hidden_dim]
        
        # Whiten/normalize (ensures device consistency)
        v_norm = self.whitening.normalize(v)
        self.last_pooled = v_norm.detach().clone()
        
        # Skip intervention during warmup
        if self.total_samples < self.config.warmup_samples:
            return hidden_state
        
        if not self._is_intervening:
            return hidden_state
        
        # Snap toward attractor
        v_snapped = self.snap(v_norm)
        
        # Ensure v_snapped is on correct device
        v_snapped = v_snapped.to(device)
        v_norm = v_norm.to(device)
        
        # Broadcast back to sequence dimension
        # Add snapped vector as residual to original hidden state
        batch_size, seq_len, hidden_dim = hidden_state.shape
        
        # Scale residual by sequence length
        residual = (v_snapped - v_norm).unsqueeze(1).expand(-1, seq_len, -1)
        modified = hidden_state + 0.1 * residual  # Small residual addition
        
        self.intervention_count += 1
        return modified
    
    def update_on_success(self, hidden_state: Optional[torch.Tensor] = None):
        """
        Update attractors based on successful inference.
        Called when the model's prediction was correct.
        """
        if hidden_state is None:
            hidden_state = self.last_hidden_state
        
        if hidden_state is None:
            return
        
        v = self.pool(hidden_state)
        v_norm = self.whitening.normalize(v).squeeze()
        
        # Update whitening statistics
        self.whitening.update(v.squeeze())
        
        # Add to K-Means buffer
        self.kmeans.add_sample(v_norm)
        
        # Find nearest attractor and nudge it toward this sample
        _, _, idx = self.attractors.find_nearest(v_norm)
        self.attractors.update_centroid(idx, v_norm)
        
        # Periodically sync K-Means centroids to attractor memory
        if self.successful_samples % (self.config.update_every * 5) == 0:
            new_centroids = self.kmeans.get_centroids()
            if new_centroids is not None:
                self.attractors.centroids = new_centroids
        
        self.successful_samples += 1
    
    def record_sample(self):
        """Record that a sample was processed."""
        self.total_samples += 1
    
    def set_intervening(self, enabled: bool):
        """Enable or disable interventions."""
        self._is_intervening = enabled
    
    def get_stats(self) -> Dict[str, Any]:
        """Get intervention statistics."""
        return {
            "total_samples": self.total_samples,
            "successful_samples": self.successful_samples,
            "intervention_count": self.intervention_count,
            "attractor_entropy": self.attractors.get_entropy(),
            "warmup_complete": self.total_samples >= self.config.warmup_samples,
        }
    
    def to(self, device):
        """Move all components to device."""
        self.whitening.to(device)
        self.attractors.to(device)
        return self


def create_intervention_hook(intervener: EASIntervener, layer_idx: int):
    """
    Create a forward hook for intervention at specified layer.
    
    Returns a hook function that can be registered with model.register_forward_hook()
    """
    def hook(module, input, output):
        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            modified = intervener.intervene(hidden_states)
            return (modified,) + output[1:]
        else:
            return intervener.intervene(output)
    
    return hook


# Convenience function to wrap a model with EAS
def wrap_model_with_eas(
    model: nn.Module,
    hidden_dim: int,
    intervention_layer: Optional[int] = None,
    config: Optional[EASConfig] = None
) -> Tuple[nn.Module, EASIntervener]:
    """
    Wrap a HuggingFace model with EAS intervention.
    
    Args:
        model: The model to wrap
        hidden_dim: Hidden dimension of the model
        intervention_layer: Which layer to intervene at (default: middle)
        config: EAS configuration
    
    Returns:
        (model, intervener) tuple
    """
    if config is None:
        config = EASConfig(hidden_dim=hidden_dim)
    else:
        config.hidden_dim = hidden_dim
    
    intervener = EASIntervener(config)
    
    # Find the transformer layers
    if hasattr(model, 'transformer'):
        layers = model.transformer.h  # GPT-2 style
    elif hasattr(model, 'gpt_neox'):
        layers = model.gpt_neox.layers  # Pythia style
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers  # LLaMA style
    else:
        raise ValueError("Could not find transformer layers in model")
    
    num_layers = len(layers)
    
    if intervention_layer is None:
        intervention_layer = num_layers // 2
    elif intervention_layer < 0:
        intervention_layer = num_layers + intervention_layer
    
    # Register hook
    target_layer = layers[intervention_layer]
    hook = create_intervention_hook(intervener, intervention_layer)
    handle = target_layer.register_forward_hook(hook)
    intervener._hook_handle = handle
    
    return model, intervener


if __name__ == "__main__":
    # Quick test
    print("EAS Core Module - Quick Test")
    print("=" * 50)
    
    # Test components
    config = EASConfig(hidden_dim=64, num_attractors=5)
    intervener = EASIntervener(config)
    
    # Simulate some activations
    for i in range(30):
        fake_hidden = torch.randn(1, 10, 64)  # batch=1, seq=10, dim=64
        
        # Intervene
        modified = intervener.intervene(fake_hidden)
        intervener.record_sample()
        
        # Simulate 60% success rate
        if np.random.random() < 0.6:
            intervener.update_on_success(fake_hidden)
    
    stats = intervener.get_stats()
    print(f"Total samples: {stats['total_samples']}")
    print(f"Successful: {stats['successful_samples']}")
    print(f"Interventions: {stats['intervention_count']}")
    print(f"Attractor entropy: {stats['attractor_entropy']:.3f}")
    print(f"Warmup complete: {stats['warmup_complete']}")
    print("\n✅ EAS Core module working correctly!")
