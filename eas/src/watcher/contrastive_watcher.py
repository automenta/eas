"""
Contrastive Watcher Implementation for EAS
Learns from the contrast between successful and failed reasoning trajectories.

Key Innovation: Instead of clustering only successes, we learn directional attractors
that represent the "correction vector" from failure to success activation patterns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Tuple, List, Dict
from eas.src.watcher import EmergentWatcher, WhiteningBuffer, AttractorMemory


class ContrastiveAttractorMemory(nn.Module):
    """
    Stores both positive attractors (success directions) and 
    negative attractors (failure regions to avoid).
    
    The key insight is that attractors are now DIRECTIONAL:
    they represent the vector from failure → success, not just success locations.
    """
    def __init__(self, dim: int, k: int = 10):
        super().__init__()
        self.dim = dim
        self.k = k
        
        # Positive attractors: directions toward correct reasoning
        self.positive_attractors = nn.Parameter(torch.randn(k, dim) * 0.1)
        
        # Anti-attractors: regions to repel away from (learned from failures)
        self.anti_attractors = nn.Parameter(torch.randn(k, dim) * 0.1)
        
        # Confidence weights for each attractor (learned importance)
        self.attractor_weights = nn.Parameter(torch.ones(k))
        
    def normalize(self):
        """Normalize attractors to unit sphere for cosine geometry"""
        with torch.no_grad():
            self.positive_attractors.data = F.normalize(self.positive_attractors.data, p=2, dim=1)
            self.anti_attractors.data = F.normalize(self.anti_attractors.data, p=2, dim=1)
            # Softmax weights for interpretability
            self.attractor_weights.data = F.softmax(self.attractor_weights.data, dim=0)


class ContrastiveClusteringEngine:
    """
    Online clustering that learns from success/failure pairs.
    
    Instead of clustering raw activations, we cluster the DIFFERENCE vectors:
    contrast = success_activation - failure_activation
    
    This captures "what makes reasoning correct" rather than just "where correct reasoning happens".
    """
    def __init__(self, n_clusters: int, batch_size: int = 5, margin: float = 0.3):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.margin = margin
        
        # Cluster contrast vectors (success - failure)
        self.contrast_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        
        # Separately track failure patterns for anti-attractors
        self.failure_kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        
        self.fitted = False
        
    def partial_fit_contrastive(self, success_batch: np.ndarray, failure_batch: np.ndarray):
        """Learn from paired success/failure examples"""
        # Check for sufficient samples
        if len(success_batch) < self.n_clusters:
            # Not enough samples yet - accumulate until we have enough
            return False
            
        # Compute contrast vectors
        contrast_vectors = success_batch - failure_batch
        
        # Normalize for directional clustering
        norms = np.linalg.norm(contrast_vectors, axis=1, keepdims=True)
        contrast_vectors = contrast_vectors / (norms + 1e-8)
        
        # Cluster the correction directions
        self.contrast_kmeans.partial_fit(contrast_vectors)
        
        # Also learn failure regions for repulsion
        self.failure_kmeans.partial_fit(failure_batch)
        
        self.fitted = True
        return True
        
    def get_positive_attractors(self) -> np.ndarray:
        """Return the learned 'correction direction' centroids"""
        if not self.fitted:
            return None
        return self.contrast_kmeans.cluster_centers_
    
    def get_anti_attractors(self) -> np.ndarray:
        """Return the learned 'failure region' centroids"""
        if not self.fitted:
            return None
        return self.failure_kmeans.cluster_centers_


class ContrastiveWatcher(nn.Module):
    """
    Enhanced Watcher that learns from contrastive success/failure pairs.
    
    Key differences from EmergentWatcher:
    1. update_contrastive() learns from pairs, not just successes
    2. snap() now includes a REPULSION term from anti-attractors
    3. Attractors represent correction DIRECTIONS, not locations
    
    This addresses the GPT-2 degradation problem by learning what to AVOID,
    not just what to approach.
    """
    def __init__(self, dim: int, k: int = 10, alpha_base: float = 0.3, 
                 repulsion_strength: float = 0.15, max_delta: float = 0.5,
                 update_frequency: int = 5, use_whitening: bool = True):
        super().__init__()
        
        self.dim = dim
        self.k = k
        self.alpha_base = alpha_base
        self.repulsion_strength = repulsion_strength
        self.max_delta = max_delta
        self.update_frequency = update_frequency
        self.use_whitening = use_whitening
        
        # Components
        self.attractor_memory = ContrastiveAttractorMemory(dim, k)
        self.whitening_buffer = WhiteningBuffer(dim)
        self.clustering_engine = ContrastiveClusteringEngine(n_clusters=k)
        
        # Paired buffers for contrastive learning
        self.success_buffer: List[np.ndarray] = []
        self.failure_buffer: List[np.ndarray] = []
        
        # Statistics
        self.intervention_count = 0
        self.repulsion_count = 0
        self.update_count = 0
        self.snap_history = []
        
    def _preprocess(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool and optionally whiten activations"""
        # Pool over sequence dimension
        pooled = hidden_states.mean(dim=1)
        
        # Update whitening statistics
        self.whitening_buffer.update(pooled)
        
        # Apply whitening if enabled
        if self.use_whitening:
            return self.whitening_buffer.whiten(pooled)
        return pooled
    
    def snap(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Contrastive snapping: attract toward positive attractors,
        repel from anti-attractors.
        
        This dual mechanism prevents the model from drifting toward
        failure patterns, even if they're geometrically close to success.
        """
        v_raw = hidden_states.mean(dim=1)
        
        # Update whitening buffer
        self.whitening_buffer.update(v_raw)
        
        # Preprocess
        if self.use_whitening:
            v_norm = self.whitening_buffer.whiten(v_raw)
        else:
            v_norm = v_raw
            
        v_unit = F.normalize(v_norm, p=2, dim=1)
        
        # Normalize attractors
        self.attractor_memory.normalize()
        pos_attractors = self.attractor_memory.positive_attractors
        anti_attractors = self.attractor_memory.anti_attractors
        
        # --- ATTRACTION toward positive attractors ---
        pos_similarities = torch.mm(v_unit, pos_attractors.t())
        best_pos_scores, best_pos_idx = torch.max(pos_similarities, dim=1)
        closest_pos = pos_attractors[best_pos_idx]
        
        # Adaptive attraction strength (less if already close)
        alpha_attract = self.alpha_base * (1.0 - best_pos_scores.unsqueeze(1))
        attraction_delta = closest_pos - v_unit
        
        # --- REPULSION from anti-attractors ---
        anti_similarities = torch.mm(v_unit, anti_attractors.t())
        worst_anti_scores, worst_anti_idx = torch.max(anti_similarities, dim=1)
        closest_anti = anti_attractors[worst_anti_idx]
        
        # Repel MORE strongly if close to anti-attractor
        alpha_repel = self.repulsion_strength * worst_anti_scores.unsqueeze(1)
        repulsion_delta = v_unit - closest_anti  # Point away from anti-attractor
        
        # Track repulsions
        repulsion_mask = worst_anti_scores > 0.5
        self.repulsion_count += repulsion_mask.sum().item()
        
        # --- COMBINE attraction and repulsion ---
        total_delta = attraction_delta * alpha_attract + repulsion_delta * alpha_repel
        
        # Safety clamp
        total_delta = torch.clamp(total_delta, -self.max_delta, self.max_delta)
        
        # Track which attractors were used
        self.snap_history.extend(best_pos_idx.cpu().numpy())
        self.intervention_count += hidden_states.size(0)
        
        # Convert back to raw space if using whitening
        if self.use_whitening:
            steering_raw = self.whitening_buffer.unwhiten_grad(total_delta)
        else:
            steering_raw = total_delta
            
        # Broadcast and add
        return hidden_states + steering_raw.unsqueeze(1)
    
    def update_contrastive(self, success_states: torch.Tensor, failure_states: torch.Tensor):
        """
        Learn from a paired success/failure example.
        
        This is the key innovation: we learn the CONTRAST between
        what works and what doesn't, not just what works.
        """
        if success_states.numel() == 0 or failure_states.numel() == 0:
            return
            
        # Pool activations
        success_pooled = success_states.mean(dim=1).detach()
        failure_pooled = failure_states.mean(dim=1).detach()
        
        # Apply whitening if enabled
        if self.use_whitening:
            self.whitening_buffer.update(success_pooled)
            self.whitening_buffer.update(failure_pooled)
            success_norm = self.whitening_buffer.whiten(success_pooled)
            failure_norm = self.whitening_buffer.whiten(failure_pooled)
        else:
            success_norm = success_pooled
            failure_norm = failure_pooled
        
        # Add to buffers
        self.success_buffer.append(success_norm.cpu().numpy())
        self.failure_buffer.append(failure_norm.cpu().numpy())
        
        # Update when we have enough pairs
        if len(self.success_buffer) >= self.update_frequency:
            success_batch = np.vstack(self.success_buffer[:self.update_frequency])
            failure_batch = np.vstack(self.failure_buffer[:self.update_frequency])
            
            # Contrastive clustering
            self.clustering_engine.partial_fit_contrastive(success_batch, failure_batch)
            
            # Update attractors from cluster centers
            pos_centers = self.clustering_engine.get_positive_attractors()
            anti_centers = self.clustering_engine.get_anti_attractors()
            
            if pos_centers is not None:
                with torch.no_grad():
                    self.attractor_memory.positive_attractors.copy_(
                        torch.tensor(pos_centers, dtype=torch.float32)
                    )
            
            if anti_centers is not None:
                with torch.no_grad():
                    self.attractor_memory.anti_attractors.copy_(
                        torch.tensor(anti_centers, dtype=torch.float32)
                    )
            
            # Normalize
            self.attractor_memory.normalize()
            
            # Clear processed pairs
            self.success_buffer = self.success_buffer[self.update_frequency:]
            self.failure_buffer = self.failure_buffer[self.update_frequency:]
            self.update_count += 1
    
    def update(self, successful_hidden_states: torch.Tensor):
        """
        Fallback: standard update using only successes.
        For compatibility with existing validation suite.
        """
        if successful_hidden_states.numel() == 0:
            return
            
        pooled = successful_hidden_states.mean(dim=1).detach()
        
        if self.use_whitening:
            self.whitening_buffer.update(pooled)
            normed = self.whitening_buffer.whiten(pooled)
        else:
            normed = pooled
            
        self.success_buffer.append(normed.cpu().numpy())
        
        if len(self.success_buffer) >= self.update_frequency:
            # Without failure data, just update positive attractors
            success_batch = np.vstack(self.success_buffer[:self.update_frequency])
            
            # Only cluster if we have enough samples
            if len(success_batch) >= self.k:
                # Use standard K-means on successes
                temp_kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=min(len(success_batch), self.update_frequency))
                temp_kmeans.fit(success_batch)
                
                with torch.no_grad():
                    self.attractor_memory.positive_attractors.copy_(
                        torch.tensor(temp_kmeans.cluster_centers_, dtype=torch.float32)
                    )
                
                self.attractor_memory.normalize()
                self.update_count += 1
            
            self.success_buffer = self.success_buffer[self.update_frequency:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Return tracking statistics for analysis"""
        return {
            "interventions": self.intervention_count,
            "repulsions": self.repulsion_count,
            "updates": self.update_count,
            "repulsion_ratio": self.repulsion_count / max(1, self.intervention_count),
            "attractor_entropy": self._compute_attractor_entropy()
        }
    
    def _compute_attractor_entropy(self) -> float:
        """Compute entropy of attractor usage (diversity measure)"""
        if len(self.snap_history) < 2:
            return 0.0
        unique, counts = np.unique(self.snap_history, return_counts=True)
        probs = counts / len(self.snap_history)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        max_entropy = np.log2(self.k)
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.intervention_count = 0
        self.repulsion_count = 0
        self.update_count = 0
        self.snap_history = []


# Convenience function for creating contrastive watcher with standard config
def create_contrastive_watcher(dim: int, k: int = 10, 
                                alpha: float = 0.3,
                                repulsion: float = 0.15) -> ContrastiveWatcher:
    """Factory function with sensible defaults for contrastive learning"""
    return ContrastiveWatcher(
        dim=dim,
        k=k,
        alpha_base=alpha,
        repulsion_strength=repulsion,
        max_delta=0.5,
        update_frequency=5,
        use_whitening=True
    )
