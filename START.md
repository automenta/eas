# Emergent Activation Snapping (EAS)

## 1. Executive Summary

Emergent Activation Snapping (EAS) is a neuro-symbolic intervention framework designed to bridge the gap between distributed neural representations and crisp logical reasoning. Unlike traditional methods that rely on supervised steering vectors, explicit symbolic encoders, or extensive fine-tuning, EAS enables a frozen language model to "bootstrap" its own logical geometry at runtime.

The system employs a lightweight, unsupervised **Watcher** module that observes internal activations during inference. By clustering the latent patterns of successful inferences ("wins"), the Watcher discovers emergent **Attractors**—geometric centroids representing implicit logical states (e.g., valid deductive steps). In subsequent inferences, the Watcher dynamically "snaps" (nudges) wandering activations toward these attractors. This creates a self-reinforcing loop where the model’s latent space becomes increasingly structured without gradient updates to the base parameters.

## 2. System Architecture

The architecture consists of two distinct components: the **Frozen Base Model** (the subject) and the **Emergent Watcher** (the intervener).

### 2.1 Base Neural Network

- **Role:** The primary reasoning engine.
- **Architecture:** Minimal Autoregressive Transformer.
    - **Scale:** ~1M parameters (2 layers, 8 heads, 512 hidden dimension).
    - **Vocabulary:** ~500 tokens (specialized for logic/syllogism corpora).
- **State:** **Frozen.** Weights are locked after initial pre-training. No gradients flow through the base model during the EAS lifecycle.
- **Interface:** Exposes a read/write hook at the middle layer (Layer 1) to allow the Watcher to intercept and modify the hidden state tensor $H$.

### 2.2 The Emergent Watcher

- **Role:** A runtime-only sidecar module that manages the lifecycle of Attractors and performs interventions.
- **Components:**
    1. **Attractor Memory:** A dynamic tensor $A \\in \\mathbb{R}^{K \\times D}$ storing $K$ centroids (default $K=10$).
    2. **Whitening Buffer:** A running statistics module to normalize input activations, reducing noise and "distractor features."
    3. **Clustering Engine:** An online K-Means algorithm responsible for evolving $A$ based on successful outcomes.

---

## 3. Functional Specification

### 3.1 Phase 1: Signal Preprocessing

Raw activations often contain noise or high-frequency components irrelevant to logic.

- **Input:** Raw hidden state $H\_{raw}$ of shape `[batch, seq_len, hidden_dim]`.
- **Pooling:** Apply attention-weighted pooling or simple mean pooling over the sequence dimension to obtain a sentence-level vector $v$.
- **Whitening:** Apply on-the-fly normalization using running mean $\\mu$ and variance $\\sigma^2$ to center and scale the signal:
$$v\_{norm} = \\frac{v - \\mu}{\\sigma + \\epsilon}$$

### 3.2 Phase 2: Adaptive Snapping (Intervention)

The Watcher identifies the nearest logical structure and nudges the activation toward it.

- **Selection:** Compute Cosine Similarity between $v\_{norm}$ and all Attractors in $A$. Identify the best matching attractor $A\_{best}$ and the similarity score $S\_{max}$.
- **Adaptive Strength ($\\alpha$):** Unlike fixed steering, $\\alpha$ is dynamic to prevent over-steering confident states or under-steering ambiguous ones.
$$\\alpha\_{dyn} = \\alpha\_{base} \\times (1 - S\_{max})$$
    - _Logic:_ If the activation is already very close to an attractor ($S\_{max} \\approx 1$), the nudge approaches zero.
- **Safety Clamp:** To prevent "off-manifold" hallucinations, the magnitude of the nudge is clamped.
$$\\Delta = A\_{best} - v\_{norm}$$
$$v\_{snapped} = v\_{norm} + \\text{clamp}(\\alpha\_{dyn} \\times \\Delta, -\\delta, \\delta)$$
- **Re-injection:** The modified vector $v\_{snapped}$ is broadcast back to the sequence length (or added as a residual) and fed into the next layer of the Base Model.

### 3.3 Phase 3: Attractor Evolution (Update)

Attractors evolve only when the model succeeds, reinforcing valid reasoning paths.

- **Trigger:** Post-inference, the system checks the prediction against an Oracle Label.
- **Condition:** Update occurs **only** if `Prediction == Label`.
- **Mechanism:**
    1. Add the successful $v\_{norm}$ to a batch buffer.
    2. Every $N$ wins (e.g., $N=5$), perform a partial fit using Online K-Means.
    3. **Sandwich Normalization:** Normalize centroids after the update to ensure they remain on the hypersphere, maintaining consistent cosine geometry.

---

## 4. Implementation Guidelines

### 4.1 Technology Stack

- **Core Framework:** PyTorch (CPU-compatible).
- **Math/Clustering:** Scikit-Learn (`MiniBatchKMeans`), NumPy.
- **Profiling:** `torch.profiler` to ensure latency overhead remains \<5%.

### 4.2 Synthetic Logic Corpus

To validate logical bootstrapping, use a dataset of 1,200 samples:

- **Structure:** Syllogisms (`All X are Y. Z is X. -> Z is Y`) and Propositional Logic (`If P then Q. P. -> Q`).
- **Splits:** 1,000 for Pre-training (Base Model), 200 for Online Evaluation.

### 4.3 Core Logic Pseudocode

```python
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

class EmergentWatcher(torch.nn.Module):
    def __init__(self, dim, k=10, alpha_base=0.3, max_delta=0.5):
        super().__init__()
        self.attractors = torch.nn.Parameter(torch.randn(k, dim)) # Random init
        self.kmeans = MiniBatchKMeans(n_clusters=k, batch_size=5)
        self.alpha_base = alpha_base
        self.max_delta = max_delta
        self.buffer = []

    def snap(self, hidden_states):
        # 1. Pooling & Normalization (Simplified Whitening)
        flat = hidden_states.mean(dim=1)
        flat_norm = F.normalize(flat, p=2, dim=1)
        att_norm = F.normalize(self.attractors, p=2, dim=1)

        # 2. Selection
        scores = torch.mm(flat_norm, att_norm.t())
        best_scores, best_indices = scores.max(dim=-1)
        closest_att = self.attractors[best_indices]

        # 3. Adaptive Alpha
        # If score is high (close), alpha drops. If low, alpha increases.
        alpha = self.alpha_base * (1.0 - best_scores.unsqueeze(1))
        
        # 4. Nudge with Clamp
        delta = closest_att - flat_norm
        delta = torch.clamp(delta, -self.max_delta, self.max_delta)
        
        snapped_flat = flat_norm + (alpha * delta)
        
        # 5. Re-inject (Broadcast back to sequence)
        return hidden_states + (snapped_flat.unsqueeze(1) - flat.unsqueeze(1))

    def update(self, successful_act):
        # Only called on WIN
        flat = successful_act.mean(dim=1).detach().cpu().numpy()
        self.buffer.append(flat)
        if len(self.buffer) >= 5:
            self.kmeans.partial_fit(self.buffer)
            self.attractors.data = torch.tensor(self.kmeans.cluster_centers_).to(self.attractors.device)
            self.buffer = []
```

---

## 5. Experimental Protocol

The experiment follows an **Online Learning** paradigm. There is no separate training phase for the Watcher.

1. **Pre-requisite:** Train Base Model to 60-70% accuracy. Freeze weights.
2. **Initialization:** Instantiate Watcher with random normal attractors.
3. **Evaluation Loop (200 Iterations):**
    - **Step A:** Forward pass with `Watcher.snap()`.
    - **Step B:** Check correctness via Oracle.
    - **Step C:** If Correct $\\rightarrow$ `Watcher.update()`.
    - **Step D:** Log Accuracy, Latency, and Attractor Entropy.
4. **Baselines:**
    - _Base:_ No Watcher.
    - _Random Control:_ Watcher enabled, but `update()` is disabled (snapping to static random noise).

---

## 6. Success Metrics & Failure Conditions

### 6.1 Primary Metrics

- **Online Learning Curve:** Accuracy must improve by $\\ge 20%$ over the baseline within 50-100 updates.
- **Attractor Stability:** Centroid variance (Euclidean shift per update) must converge to $\< 0.05$.
- **Latency Overhead:** Total inference time increase must be $\< 5%$.

### 6.2 Safety & Robustness Metrics

- **Collapse Detection:** Calculate the entropy of attractor usage.
    - _Failure:_ If $> 80%$ of snaps map to a single attractor (Mode Collapse).
- **Hallucination Rate:** Monitor "off-manifold" drifts.
    - _Metric:_ If the distance between $v\_{snapped}$ and $v\_{raw}$ consistently exceeds a safety threshold (e.g., Euclidean distance > 1.0), the system is destabilizing.

### 6.3 Qualitative Analysis

- **t-SNE Visualization:** Plot the trajectory of activations. Successful emergence is defined by the formation of distinct "islands" (attractors) corresponding to logical types (e.g., Transitivity vs. Negation) rather than a single amorphous cloud.

---

## 7. Optional Extensions

### 7.1 Symbolic Verifier (Safety Module)

- **Description:** A lightweight logical checker (e.g., mini-SAT).
- **Integration:** Before the final output generation, the snapped activation is decoded into a discrete logical form. If the form is invalid (e.g., $A \\land \\neg A$), the snap is rejected, and the raw activation is restored.

### 7.2 Hyperbolic Geometry (Advanced)

- **Description:** Replace Euclidean operations with Hyperbolic (Poincaré ball) arithmetic.
- **Rationale:** Better suited for hierarchical logic (trees/graphs).
- **Change:** Use Möbius addition for the "nudge" step to respect manifold curvature.

---

## 8. Conclusion

This specification defines a complete, closed-loop system for **Emergent Activation Snapping**. By combining unsupervised clustering with adaptive, geometry-aware interventions, EAS provides a pathway to infuse frozen neural networks with crisp, evolving logical structures, ensuring interpretability and improved reasoning with minimal computational cost.
