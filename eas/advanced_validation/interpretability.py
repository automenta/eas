"""
Interpretability Tools for EAS Research
Visualization and analysis utilities for attractor geometry and reasoning trajectories.
"""
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import json
import os

# Optional matplotlib import (may have compatibility issues)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, visualizations disabled")

# Optional sklearn imports
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, dimensionality reduction disabled")


class AttractorVisualizer:
    """
    Visualizes attractor geometry and activation trajectories.
    Essential for paper figures and interpretability analysis.
    """
    def __init__(self, save_dir: str = "eas/advanced_validation/results/visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Trajectory tracking
        self.trajectories: Dict[str, List[np.ndarray]] = {}
        self.attractor_history: List[np.ndarray] = []
        self.labels: List[str] = []
        
    def record_trajectory(self, activations: np.ndarray, label: str, step: int):
        """Record a single point in an activation trajectory"""
        if label not in self.trajectories:
            self.trajectories[label] = []
        self.trajectories[label].append({
            'step': step,
            'activation': activations.copy()
        })
    
    def record_attractors(self, attractors: np.ndarray, step: int):
        """Record attractor positions over time"""
        self.attractor_history.append({
            'step': step,
            'attractors': attractors.copy()
        })
    
    def plot_attractor_evolution(self, method: str = 'pca', 
                                  filename: str = 'attractor_evolution.png') -> str:
        """
        Visualize how attractors evolve over training.
        Shows the trajectory of each attractor centroid.
        """
        if len(self.attractor_history) < 2:
            print("Not enough attractor history to plot evolution")
            return None
        
        # Stack all attractors
        all_attractors = np.vstack([h['attractors'] for h in self.attractor_history])
        steps = []
        for h in self.attractor_history:
            steps.extend([h['step']] * len(h['attractors']))
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=min(30, len(all_attractors) - 1))
        else:
            reducer = PCA(n_components=2)
        
        reduced = reducer.fit_transform(all_attractors)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        n_attractors = len(self.attractor_history[0]['attractors'])
        colors = plt.cm.viridis(np.linspace(0, 1, n_attractors))
        
        for i in range(n_attractors):
            # Extract trajectory for this attractor
            traj_x = []
            traj_y = []
            idx = 0
            for h in self.attractor_history:
                traj_x.append(reduced[idx + i, 0])
                traj_y.append(reduced[idx + i, 1])
                idx += len(h['attractors'])
            
            ax.plot(traj_x, traj_y, 'o-', color=colors[i], alpha=0.7, 
                   markersize=8, label=f'Attractor {i}')
            # Mark start and end
            ax.scatter(traj_x[0], traj_y[0], marker='s', s=100, 
                      color=colors[i], edgecolors='black', zorder=5)
            ax.scatter(traj_x[-1], traj_y[-1], marker='^', s=100, 
                      color=colors[i], edgecolors='black', zorder=5)
        
        ax.set_xlabel(f'{method.upper()} Dimension 1')
        ax.set_ylabel(f'{method.upper()} Dimension 2')
        ax.set_title('Attractor Evolution During Training\n(□ = start, △ = end)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        filepath = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_success_failure_contrast(self, 
                                       success_activations: np.ndarray,
                                       failure_activations: np.ndarray,
                                       attractors: Optional[np.ndarray] = None,
                                       filename: str = 'success_failure_contrast.png') -> str:
        """
        Visualize the separation between successful and failed reasoning.
        This is the key figure for contrastive learning papers.
        """
        # Combine for joint embedding
        combined = np.vstack([success_activations, failure_activations])
        if attractors is not None:
            combined = np.vstack([combined, attractors])
        
        # Reduce
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)
        
        n_success = len(success_activations)
        n_failure = len(failure_activations)
        
        success_reduced = reduced[:n_success]
        failure_reduced = reduced[n_success:n_success + n_failure]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot success points
        ax.scatter(success_reduced[:, 0], success_reduced[:, 1], 
                  c='green', alpha=0.6, s=50, label='Correct Reasoning')
        
        # Plot failure points
        ax.scatter(failure_reduced[:, 0], failure_reduced[:, 1], 
                  c='red', alpha=0.6, s=50, label='Incorrect Reasoning')
        
        # Plot attractors if provided
        if attractors is not None:
            attractor_reduced = reduced[n_success + n_failure:]
            ax.scatter(attractor_reduced[:, 0], attractor_reduced[:, 1], 
                      c='blue', marker='*', s=200, edgecolors='black',
                      label='Attractors', zorder=5)
        
        # Draw arrows from failure centroids to success centroids
        success_centroid = success_reduced.mean(axis=0)
        failure_centroid = failure_reduced.mean(axis=0)
        ax.annotate('', xy=success_centroid, xytext=failure_centroid,
                   arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        
        ax.set_xlabel('PCA Dimension 1')
        ax.set_ylabel('PCA Dimension 2')
        ax.set_title('Success vs Failure in Activation Space')
        ax.legend()
        
        filepath = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_snapping_effect(self,
                             before_snap: np.ndarray,
                             after_snap: np.ndarray,
                             attractors: np.ndarray,
                             labels: List[bool],
                             filename: str = 'snapping_effect.png') -> str:
        """
        Visualize the effect of snapping: before and after activation positions.
        Shows how EAS moves activations toward attractors.
        """
        # Combine all for consistent embedding
        combined = np.vstack([before_snap, after_snap, attractors])
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(combined)
        
        n = len(before_snap)
        before_reduced = reduced[:n]
        after_reduced = reduced[n:2*n]
        attractor_reduced = reduced[2*n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Color by success/failure
        colors = ['green' if l else 'red' for l in labels]
        
        # Draw arrows from before to after
        for i in range(n):
            ax.annotate('', xy=after_reduced[i], xytext=before_reduced[i],
                       arrowprops=dict(arrowstyle='->', color=colors[i], alpha=0.5))
        
        # Plot before points (hollow)
        ax.scatter(before_reduced[:, 0], before_reduced[:, 1], 
                  c='none', edgecolors=colors, s=50, alpha=0.7, label='Before Snap')
        
        # Plot after points (filled)
        ax.scatter(after_reduced[:, 0], after_reduced[:, 1], 
                  c=colors, s=50, alpha=0.7, label='After Snap')
        
        # Plot attractors
        ax.scatter(attractor_reduced[:, 0], attractor_reduced[:, 1], 
                  c='blue', marker='*', s=200, edgecolors='black',
                  label='Attractors', zorder=5)
        
        ax.set_xlabel('PCA Dimension 1')
        ax.set_ylabel('PCA Dimension 2')
        ax.set_title('Effect of Activation Snapping\n(Arrows show movement)')
        ax.legend()
        
        filepath = os.path.join(self.save_dir, filename)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        return filepath


class LogitLensAnalyzer:
    """
    Projects attractors back to vocabulary space to interpret what they represent.
    
    Key question: Do attractors map to logical words like "therefore", "true", "follows"?
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def decode_attractor(self, attractor: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Project an attractor vector to vocabulary space.
        Returns top-k tokens that the attractor "represents".
        """
        attractor_tensor = torch.tensor(attractor, dtype=torch.float32)
        
        # Get the unembedding matrix (transpose of embedding)
        if hasattr(self.model, 'lm_head'):
            unembed = self.model.lm_head.weight.data
        elif hasattr(self.model, 'embed_out'):
            unembed = self.model.embed_out.weight.data
        else:
            # Try to find it
            for name, param in self.model.named_parameters():
                if 'lm_head' in name or 'embed_out' in name:
                    unembed = param.data
                    break
            else:
                raise ValueError("Could not find unembedding matrix")
        
        # Project attractor to logits
        if attractor_tensor.dim() == 1:
            attractor_tensor = attractor_tensor.unsqueeze(0)
        
        # May need to match dimensions
        if attractor_tensor.size(-1) != unembed.size(-1):
            print(f"Dimension mismatch: attractor {attractor_tensor.size(-1)} vs unembed {unembed.size(-1)}")
            return []
        
        logits = torch.mm(attractor_tensor, unembed.t())
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        results = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = self.tokenizer.decode([idx])
            results.append((token, prob))
        
        return results
    
    def analyze_all_attractors(self, attractors: np.ndarray, 
                                top_k: int = 5) -> Dict[int, List[Tuple[str, float]]]:
        """Decode all attractors and return their vocabulary projections"""
        results = {}
        for i, attractor in enumerate(attractors):
            results[i] = self.decode_attractor(attractor, top_k)
        return results
    
    def generate_report(self, attractors: np.ndarray, 
                         save_path: str = None) -> str:
        """Generate a human-readable report of attractor meanings"""
        analysis = self.analyze_all_attractors(attractors)
        
        report = "# Attractor Vocabulary Analysis\n\n"
        report += "This report shows what words/tokens each attractor represents "
        report += "when projected to vocabulary space.\n\n"
        
        for idx, tokens in analysis.items():
            report += f"## Attractor {idx}\n"
            report += "| Token | Probability |\n"
            report += "|-------|-------------|\n"
            for token, prob in tokens:
                token_clean = token.replace('\n', '\\n').replace('|', '\\|')
                report += f"| `{token_clean}` | {prob:.4f} |\n"
            report += "\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report


class ProbeClassifier:
    """
    Train linear probes to determine if different attractors encode different logical operations.
    
    Key hypothesis: Attractor k=0 encodes "modus ponens", k=1 encodes "modus tollens", etc.
    """
    def __init__(self, n_attractors: int):
        self.n_attractors = n_attractors
        self.probe_weights = None
        
    def fit(self, activations: np.ndarray, logic_types: List[str]):
        """
        Train a probe to predict logic type from nearest attractor.
        
        Args:
            activations: Activation vectors [n_samples, dim]
            logic_types: Logic type labels for each sample
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        
        # Encode logic types
        le = LabelEncoder()
        y = le.fit_transform(logic_types)
        self.label_encoder = le
        
        # Train probe
        self.probe = LogisticRegression(max_iter=1000)
        self.probe.fit(activations, y)
        
        return self.probe.score(activations, y)
    
    def predict_logic_type(self, activation: np.ndarray) -> str:
        """Predict the logic type from an activation"""
        if self.probe is None:
            raise ValueError("Probe not trained")
        
        pred = self.probe.predict(activation.reshape(1, -1))
        return self.label_encoder.inverse_transform(pred)[0]
    
    def get_attractor_logic_mapping(self, 
                                     attractors: np.ndarray,
                                     activations: np.ndarray,
                                     logic_types: List[str]) -> Dict[int, str]:
        """
        Determine which logic type each attractor primarily encodes.
        """
        from collections import Counter
        
        # Find nearest attractor for each activation
        norm_act = activations / (np.linalg.norm(activations, axis=1, keepdims=True) + 1e-8)
        norm_att = attractors / (np.linalg.norm(attractors, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(norm_act, norm_att.T)
        nearest_attractors = similarities.argmax(axis=1)
        
        # Count logic types per attractor
        attractor_logics = {i: [] for i in range(len(attractors))}
        for att_idx, logic_type in zip(nearest_attractors, logic_types):
            attractor_logics[att_idx].append(logic_type)
        
        # Find dominant logic type for each attractor
        mapping = {}
        for att_idx, types in attractor_logics.items():
            if types:
                counter = Counter(types)
                mapping[att_idx] = counter.most_common(1)[0][0]
            else:
                mapping[att_idx] = 'unused'
        
        return mapping


def create_summary_figure(visualizer: AttractorVisualizer,
                          success_acts: np.ndarray,
                          failure_acts: np.ndarray,
                          attractors: np.ndarray,
                          save_path: str = 'eas_summary_figure.png'):
    """
    Create a publication-ready summary figure with multiple panels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Success vs Failure
    combined = np.vstack([success_acts, failure_acts, attractors])
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined)
    
    n_s, n_f = len(success_acts), len(failure_acts)
    
    axes[0].scatter(reduced[:n_s, 0], reduced[:n_s, 1], c='green', alpha=0.6, label='Correct')
    axes[0].scatter(reduced[n_s:n_s+n_f, 0], reduced[n_s:n_s+n_f, 1], c='red', alpha=0.6, label='Incorrect')
    axes[0].scatter(reduced[n_s+n_f:, 0], reduced[n_s+n_f:, 1], c='blue', marker='*', s=200, label='Attractors')
    axes[0].set_title('(A) Activation Space')
    axes[0].legend()
    
    # Panel 2: Attractor distances
    ax = axes[1]
    att_reduced = reduced[n_s+n_f:]
    for i, (x, y) in enumerate(att_reduced):
        ax.scatter(x, y, c='blue', s=200, marker='*')
        ax.annotate(f'A{i}', (x, y), xytext=(5, 5), textcoords='offset points')
    ax.set_title('(B) Attractor Positions')
    
    # Panel 3: Histogram of similarities
    ax = axes[2]
    success_norm = success_acts / (np.linalg.norm(success_acts, axis=1, keepdims=True) + 1e-8)
    att_norm = attractors / (np.linalg.norm(attractors, axis=1, keepdims=True) + 1e-8)
    sims = np.dot(success_norm, att_norm.T).max(axis=1)
    ax.hist(sims, bins=20, color='green', alpha=0.7, label='Correct')
    
    failure_norm = failure_acts / (np.linalg.norm(failure_acts, axis=1, keepdims=True) + 1e-8)
    sims_f = np.dot(failure_norm, att_norm.T).max(axis=1)
    ax.hist(sims_f, bins=20, color='red', alpha=0.7, label='Incorrect')
    ax.set_title('(C) Similarity to Nearest Attractor')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path
