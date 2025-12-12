"""
Self-Supervised Watcher Implementation for EAS
Uses model confidence as supervision signal - no labels needed!

Key Innovation: When the model is confident (low entropy output), 
it's naturally near an attractor. We bootstrap from this signal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Dict, List, Tuple
from eas.src.watcher import EmergentWatcher, WhiteningBuffer, AttractorMemory


class ConfidenceTracker:
    """
    Tracks model confidence over time to calibrate thresholds.
    Uses running statistics to adapt to the model's natural confidence distribution.
    """
    def __init__(self, momentum: float = 0.1):
        self.momentum = momentum
        self.running_mean_confidence = 0.5
        self.running_std_confidence = 0.2
        self.confidence_history: List[float] = []
        
    def update(self, confidence: float):
        """Update running confidence statistics"""
        self.confidence_history.append(confidence)
        
        # Keep history bounded
        if len(self.confidence_history) > 1000:
            self.confidence_history = self.confidence_history[-500:]
        
        # Update running stats
        self.running_mean_confidence = (
            (1 - self.momentum) * self.running_mean_confidence + 
            self.momentum * confidence
        )
        
        if len(self.confidence_history) > 10:
            std = np.std(self.confidence_history[-100:])
            self.running_std_confidence = (
                (1 - self.momentum) * self.running_std_confidence + 
                self.momentum * std
            )
    
    def is_high_confidence(self, confidence: float, threshold_sigmas: float = 0.5) -> bool:
        """Check if confidence is above adaptive threshold"""
        threshold = self.running_mean_confidence + threshold_sigmas * self.running_std_confidence
        return confidence > threshold
    
    def is_low_confidence(self, confidence: float, threshold_sigmas: float = -0.5) -> bool:
        """Check if confidence is below adaptive threshold"""
        threshold = self.running_mean_confidence + threshold_sigmas * self.running_std_confidence
        return confidence < threshold
    
    def get_confidence_weight(self, confidence: float) -> float:
        """
        Returns a weight [0, 1] based on how confident the model is.
        Higher confidence = higher weight for attractor learning.
        """
        # Normalize to [0, 1] using running stats
        z_score = (confidence - self.running_mean_confidence) / (self.running_std_confidence + 1e-8)
        weight = torch.sigmoid(torch.tensor(z_score)).item()
        return weight


class UncertaintyRegionMemory:
    """
    Tracks regions of high uncertainty in activation space.
    These regions should be avoided or treated with care during snapping.
    """
    def __init__(self, dim: int, max_regions: int = 50):
        self.dim = dim
        self.max_regions = max_regions
        self.uncertainty_regions: List[np.ndarray] = []
        self.uncertainty_scores: List[float] = []
        
    def add_region(self, activation: np.ndarray, uncertainty_score: float):
        """Record an uncertain activation pattern"""
        self.uncertainty_regions.append(activation)
        self.uncertainty_scores.append(uncertainty_score)
        
        # Keep bounded
        if len(self.uncertainty_regions) > self.max_regions:
            # Remove oldest low-uncertainty regions
            sorted_indices = np.argsort(self.uncertainty_scores)
            keep_indices = sorted_indices[-(self.max_regions // 2):]
            self.uncertainty_regions = [self.uncertainty_regions[i] for i in keep_indices]
            self.uncertainty_scores = [self.uncertainty_scores[i] for i in keep_indices]
    
    def get_uncertainty_penalty(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Returns a penalty score based on proximity to uncertain regions.
        Used to reduce snapping strength when near uncertain areas.
        """
        if len(self.uncertainty_regions) == 0:
            return torch.zeros(activation.size(0), device=activation.device)
        
        # Stack all uncertainty regions
        regions = torch.tensor(np.vstack(self.uncertainty_regions), 
                              dtype=torch.float32, device=activation.device)
        
        # Compute similarity to uncertainty regions
        act_norm = F.normalize(activation, p=2, dim=1)
        reg_norm = F.normalize(regions, p=2, dim=1)
        
        similarities = torch.mm(act_norm, reg_norm.t())  # [batch, num_regions]
        max_similarity = similarities.max(dim=1).values
        
        return max_similarity


class SelfSupervisedWatcher(nn.Module):
    """
    Watcher that learns attractors using model confidence as supervision.
    
    Key Innovation: No oracle labels needed!
    - High confidence outputs → the model is near a natural attractor → reinforce
    - Low confidence outputs → the model is wandering → be careful with intervention
    
    This enables:
    1. Unlimited scaling (no labeling cost)
    2. Task-agnostic learning (works on any generation task)
    3. Self-improving loop (more confidence → stronger attractors → more confidence)
    """
    def __init__(self, dim: int, k: int = 10, alpha_base: float = 0.3,
                 max_delta: float = 0.5, update_frequency: int = 5,
                 use_whitening: bool = True, confidence_threshold: float = 0.7):
        super().__init__()
        
        self.dim = dim
        self.k = k
        self.alpha_base = alpha_base
        self.max_delta = max_delta
        self.update_frequency = update_frequency
        self.use_whitening = use_whitening
        self.confidence_threshold = confidence_threshold
        
        # Core components
        self.attractor_memory = AttractorMemory(dim, k)
        self.whitening_buffer = WhiteningBuffer(dim)
        self.kmeans = MiniBatchKMeans(n_clusters=k, batch_size=update_frequency)
        
        # Self-supervised components
        self.confidence_tracker = ConfidenceTracker()
        self.uncertainty_memory = UncertaintyRegionMemory(dim)
        
        # Weighted activation buffer
        self.activation_buffer: List[Tuple[np.ndarray, float]] = []  # (activation, weight)
        
        # Statistics
        self.intervention_count = 0
        self.high_confidence_updates = 0
        self.low_confidence_skips = 0
        self.update_count = 0
        self.snap_history = []
        
        self.kmeans_fitted = False
        
    def _preprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool and optionally whiten activations"""
        pooled = hidden_states.mean(dim=1)
        self.whitening_buffer.update(pooled)
        
        if self.use_whitening:
            return self.whitening_buffer.whiten(pooled)
        return pooled
    
    def compute_confidence(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute model confidence from output logits.
        Confidence = 1 - normalized_entropy
        """
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute entropy
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log(logits.size(-1))
        normalized_entropy = entropy / max_entropy
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - normalized_entropy
        
        return confidence
    
    def snap(self, hidden_states: torch.Tensor, 
             logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-supervised snapping with confidence-aware strength.
        
        If logits are provided, we modulate snapping strength based on
        model confidence. High confidence = stronger snapping.
        """
        v_raw = hidden_states.mean(dim=1)
        self.whitening_buffer.update(v_raw)
        
        if self.use_whitening:
            v_norm = self.whitening_buffer.whiten(v_raw)
        else:
            v_norm = v_raw
            
        v_unit = F.normalize(v_norm, p=2, dim=1)
        
        # Normalize attractors
        self.attractor_memory.normalize_attractors()
        attractors = F.normalize(self.attractor_memory.attractors, p=2, dim=1)
        
        # Find closest attractor
        similarities = torch.mm(v_unit, attractors.t())
        best_scores, best_idx = torch.max(similarities, dim=1)
        closest_att = self.attractor_memory.attractors[best_idx]
        
        # Base adaptive alpha
        alpha = self.alpha_base * (1.0 - best_scores.unsqueeze(1))
        
        # Confidence-based modulation (if logits provided)
        if logits is not None:
            confidence = self.compute_confidence(logits[:, -1, :])  # Last token
            
            # Update tracker
            for c in confidence.cpu().numpy():
                self.confidence_tracker.update(float(c))
            
            # Scale alpha by confidence
            confidence_weight = confidence.unsqueeze(1)
            alpha = alpha * confidence_weight
        
        # Uncertainty penalty (reduce snapping if near uncertain regions)
        uncertainty_penalty = self.uncertainty_memory.get_uncertainty_penalty(v_norm)
        alpha = alpha * (1.0 - uncertainty_penalty.unsqueeze(1) * 0.5)
        
        # Compute and clamp delta
        delta = closest_att - v_unit
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        
        # Apply steering
        steering = alpha * delta
        
        if self.use_whitening:
            steering_raw = self.whitening_buffer.unwhiten_grad(steering)
        else:
            steering_raw = steering
        
        # Track
        self.snap_history.extend(best_idx.cpu().numpy())
        self.intervention_count += hidden_states.size(0)
        
        return hidden_states + steering_raw.unsqueeze(1)
    
    def self_supervised_update(self, hidden_states: torch.Tensor, 
                                logits: torch.Tensor):
        """
        Update attractors using model confidence as supervision signal.
        
        This is the key breakthrough: no labels needed!
        - High confidence examples get high weight in clustering
        - Low confidence examples are recorded as uncertainty regions
        """
        if hidden_states.numel() == 0 or logits.numel() == 0:
            return
        
        # Compute confidence for each example
        confidence = self.compute_confidence(logits[:, -1, :])  # Last token logits
        
        # Pool and preprocess activations
        pooled = hidden_states.mean(dim=1).detach()
        
        if self.use_whitening:
            self.whitening_buffer.update(pooled)
            normed = self.whitening_buffer.whiten(pooled)
        else:
            normed = pooled
        
        # Process each example based on confidence
        for i in range(hidden_states.size(0)):
            conf = float(confidence[i].cpu())
            act = normed[i].cpu().numpy()
            
            # Update confidence tracker
            self.confidence_tracker.update(conf)
            
            if self.confidence_tracker.is_high_confidence(conf):
                # High confidence → add to attractor buffer with weight
                weight = self.confidence_tracker.get_confidence_weight(conf)
                self.activation_buffer.append((act, weight))
                self.high_confidence_updates += 1
                
            elif self.confidence_tracker.is_low_confidence(conf):
                # Low confidence → record as uncertainty region
                uncertainty_score = 1.0 - conf
                self.uncertainty_memory.add_region(act, uncertainty_score)
                self.low_confidence_skips += 1
            
            # Medium confidence → ignore (neither strong signal)
        
        # Update attractors when buffer is full
        if len(self.activation_buffer) >= self.update_frequency:
            self._update_attractors_weighted()
    
    def _update_attractors_weighted(self):
        """Update attractors using weighted examples"""
        if len(self.activation_buffer) < self.k:
            return
        
        # Extract activations and weights
        activations = np.vstack([a for a, w in self.activation_buffer[:self.update_frequency]])
        weights = np.array([w for a, w in self.activation_buffer[:self.update_frequency]])
        
        # Weighted K-means (approximate by replicating high-weight samples)
        # This is simpler than true weighted K-means but effective
        weighted_samples = []
        for act, weight in zip(activations, weights):
            # Replicate proportional to weight
            n_replicates = max(1, int(weight * 5))
            weighted_samples.extend([act] * n_replicates)
        
        weighted_samples = np.array(weighted_samples)
        
        # Fit K-means
        if len(weighted_samples) >= self.k:
            self.kmeans.partial_fit(weighted_samples)
            self.kmeans_fitted = True
            
            # Update attractors
            with torch.no_grad():
                self.attractor_memory.attractors.copy_(
                    torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32)
                )
            self.attractor_memory.normalize_attractors()
        
        # Clear buffer
        self.activation_buffer = self.activation_buffer[self.update_frequency:]
        self.update_count += 1
    
    def update(self, successful_hidden_states: torch.Tensor):
        """
        Fallback: standard update for compatibility with validation suite.
        Treats all examples as high confidence.
        """
        if successful_hidden_states.numel() == 0:
            return
        
        pooled = successful_hidden_states.mean(dim=1).detach()
        
        if self.use_whitening:
            self.whitening_buffer.update(pooled)
            normed = self.whitening_buffer.whiten(pooled)
        else:
            normed = pooled
        
        # Add with high weight
        for act in normed.cpu().numpy():
            self.activation_buffer.append((act, 1.0))
        
        if len(self.activation_buffer) >= self.update_frequency:
            self._update_attractors_weighted()
    
    def get_statistics(self) -> Dict[str, float]:
        """Return tracking statistics"""
        total_decisions = self.high_confidence_updates + self.low_confidence_skips
        return {
            "interventions": self.intervention_count,
            "updates": self.update_count,
            "high_confidence_updates": self.high_confidence_updates,
            "low_confidence_skips": self.low_confidence_skips,
            "high_confidence_ratio": self.high_confidence_updates / max(1, total_decisions),
            "mean_confidence": self.confidence_tracker.running_mean_confidence,
            "uncertainty_regions": len(self.uncertainty_memory.uncertainty_regions),
            "attractor_entropy": self._compute_attractor_entropy()
        }
    
    def _compute_attractor_entropy(self) -> float:
        """Compute entropy of attractor usage"""
        if len(self.snap_history) < 2:
            return 0.0
        unique, counts = np.unique(self.snap_history, return_counts=True)
        probs = counts / len(self.snap_history)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        max_entropy = np.log2(self.k)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def reset_stats(self):
        """Reset statistics"""
        self.intervention_count = 0
        self.high_confidence_updates = 0
        self.low_confidence_skips = 0
        self.update_count = 0
        self.snap_history = []


# Factory function
def create_self_supervised_watcher(dim: int, k: int = 10,
                                    alpha: float = 0.3) -> SelfSupervisedWatcher:
    """Create a self-supervised watcher with sensible defaults"""
    return SelfSupervisedWatcher(
        dim=dim,
        k=k,
        alpha_base=alpha,
        max_delta=0.5,
        update_frequency=5,
        use_whitening=True
    )
