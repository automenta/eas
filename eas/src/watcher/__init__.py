"""
Emergent Watcher Implementation for EAS
Implements Attractor Memory, Whitening Buffer, and Clustering Engine
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Optional, Tuple, List


class WhiteningBuffer:
    """Running statistics buffer for normalization (whitening) of input activations"""
    def __init__(self, dim: int, momentum: float = 0.1):
        self.dim = dim
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.ones(dim))
        self.count = 0
        self.initialized = False
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register a buffer (for compatibility without nn.Module)"""
        setattr(self, name, tensor)
    
    def update(self, x: torch.Tensor):
        """Update running statistics with new batch"""
        # Calculate batch statistics
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        
        if not self.initialized:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.initialized = True
        else:
            # Exponential moving average update
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
    
    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        """Apply whitening transformation"""
        return (x - self.running_mean) / torch.sqrt(self.running_var + 1e-8)


class AttractorMemory(nn.Module):
    """Attractor Memory storing K centroids in R^(K×D)"""
    def __init__(self, dim: int, k: int = 10):
        super().__init__()
        self.dim = dim
        self.k = k
        # Initialize with random normal attractors
        self.attractors = nn.Parameter(torch.randn(k, dim))
        
    def normalize_attractors(self):
        """Normalize attractors to maintain hypersphere consistency (Sandwich Normalization)"""
        with torch.no_grad():
            # Normalize each attractor to unit length
            norms = torch.norm(self.attractors, p=2, dim=1, keepdim=True)
            self.attractors.data = self.attractors.data / norms.clamp(min=1e-8)


class ClusteringEngine:
    """Online K-Means clustering engine for attractor evolution"""
    def __init__(self, n_clusters: int, batch_size: int = 5):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)
        self.fitted = False
    
    def partial_fit(self, X: np.ndarray):
        """Partially fit the clustering model with new data"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.kmeans.partial_fit(X)
        self.fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments"""
        if not self.fitted:
            # If not fitted yet, return random cluster assignments
            return np.random.randint(0, self.n_clusters, size=X.shape[0])
        return self.kmeans.predict(X)


class EmergentWatcher(nn.Module):
    """The Emergent Watcher module that manages attractors and performs interventions"""
    def __init__(self, dim: int, k: int = 10, alpha_base: float = 0.3, max_delta: float = 0.5, 
                 update_frequency: int = 5):
        super().__init__()
        
        self.dim = dim
        self.k = k
        self.alpha_base = alpha_base
        self.max_delta = max_delta
        self.update_frequency = update_frequency
        
        # Components
        self.attractor_memory = AttractorMemory(dim, k)
        self.whitening_buffer = WhiteningBuffer(dim)
        self.clustering_engine = ClusteringEngine(n_clusters=k)
        
        # Internal buffers
        self.successful_activations = []  # Buffer to store successful activations
        self.activation_buffer = []  # Buffer for computing whitening statistics
        self.intervention_count = 0
        self.update_count = 0
        self.snap_history = []  # Track which attractors were used for snaps
        
    def snap(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Phase 2: Adaptive Snapping (Intervention)
        Identifies nearest logical structure and nudges activation toward it
        """
        # Phase 1: Signal Preprocessing (Pooling and Whitening)
        # Pool over sequence dimension to get sentence-level vector
        v_raw = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        
        # Update whitening buffer with raw activations
        self.whitening_buffer.update(v_raw)
        
        # Apply whitening normalization
        v_norm = self.whitening_buffer.whiten(v_raw)
        
        # Ensure attractors are normalized
        self.attractor_memory.normalize_attractors()
        attractors_norm = F.normalize(self.attractor_memory.attractors, p=2, dim=1)

        # Compute cosine similarity between normalized activations and attractors
        cosine_similarities = torch.mm(v_norm, attractors_norm.t())  # [batch, k]

        # Find best matching attractor for each sample in the batch
        best_scores, best_indices = torch.max(cosine_similarities, dim=1)

        # Record which attractors were used
        self.snap_history.extend(best_indices.cpu().numpy())

        # Get the closest attractors
        closest_att = self.attractor_memory.attractors[best_indices]  # [batch, hidden_dim]
        
        # Compute adaptive alpha (dynamic strength)
        # If activation is already very close to an attractor, the nudge approaches zero
        alpha_dyn = self.alpha_base * (1.0 - best_scores.unsqueeze(1))
        
        # Compute the nudge vector
        delta = closest_att - v_norm
        # Clamp the delta to prevent excessive changes (safety clamp)
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        
        # Apply the nudge
        v_snapped = v_norm + (alpha_dyn * delta)

        # Update intervention count
        self.intervention_count += hidden_states.size(0)
        
        # Broadcast back to sequence length (add as residual to original hidden states)
        # Calculate the difference to add as residual
        v_diff = v_snapped.unsqueeze(1) - v_raw.unsqueeze(1)
        
        return hidden_states + v_diff
    
    def adapt(self, hidden_states: torch.Tensor):
        """
        Unsupervised Domain Adaptation
        Updates ONLY the whitening buffer to align the coordinate space
        Does NOT update attractors or clustering
        """
        if hidden_states.numel() == 0:
            return

        # Pool activations over sequence dimension
        pooled = hidden_states.mean(dim=1).detach()

        # Update whitening buffer
        self.whitening_buffer.update(pooled)

    def update(self, successful_hidden_states: torch.Tensor):
        """
        Phase 3: Attractor Evolution (Update)
        Updates attractors only when model succeeds
        """
        if successful_hidden_states.numel() == 0:
            return

        # Pool successful activations over sequence dimension
        successful_pooled = successful_hidden_states.mean(dim=1).detach()

        # Store in buffer for later clustering update
        success_array = successful_pooled.cpu().numpy()
        if success_array.ndim == 1:
            success_array = success_array.reshape(1, -1)  # Reshape single sample

        self.successful_activations.extend(success_array)

        # Update whitening buffer with successful activations
        self.whitening_buffer.update(successful_pooled)

        # Perform clustering update if we have enough samples
        if len(self.successful_activations) >= self.update_frequency:
            # Convert to numpy array for sklearn
            # Only use up to update_frequency samples to avoid clustering issues
            samples_to_process = min(self.update_frequency, len(self.successful_activations))
            success_array = np.array(self.successful_activations[:samples_to_process])

            # Make sure we have at least as many samples as clusters
            if success_array.shape[0] >= self.k:
                # Update clustering with successful activations
                self.clustering_engine.partial_fit(success_array)

                # Update attractors based on cluster centers (this is a simplified approach)
                # In a more sophisticated implementation, we'd use the cluster centers directly
                try:
                    cluster_centers = self.clustering_engine.kmeans.cluster_centers_
                    if cluster_centers.shape[0] == self.k:
                        # Update attractor parameters based on cluster centers
                        with torch.no_grad():
                            self.attractor_memory.attractors.copy_(torch.tensor(cluster_centers,
                                                                              dtype=torch.float32))

                        # Normalize attractors
                        self.attractor_memory.normalize_attractors()
                except:
                    # If clustering fails, just continue without updating attractors
                    pass

            # Remove processed activations from buffer
            self.successful_activations = self.successful_activations[samples_to_process:]
            self.update_count += 1
    
    def get_attractor_stability(self) -> float:
        """Calculate attractor stability as a metric"""
        if len(self.snap_history) < 2:
            return 0.0
            
        # Calculate entropy of attractor usage (0 = all same attractor, high = evenly distributed)
        unique, counts = np.unique(self.snap_history, return_counts=True)
        probs = counts / len(self.snap_history)
        entropy = -np.sum(probs * np.log2(probs + 1e-8))
        max_entropy = np.log2(len(self.attractor_memory.attractors))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def get_intervention_frequency(self) -> float:
        """Calculate how often snapping occurs"""
        return self.intervention_count
    
    def reset_stats(self):
        """Reset tracking statistics"""
        self.intervention_count = 0
        self.update_count = 0
        self.snap_history = []