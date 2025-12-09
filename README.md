# Master Specification: Emergent Activation Snapping (EAS)

## 1. Executive Summary

Emergent Activation Snapping (EAS) is a neuro-symbolic intervention framework designed to bridge the gap between distributed neural representations and crisp logical reasoning. Unlike traditional methods that rely on supervised steering vectors, explicit symbolic encoders, or extensive fine-tuning, EAS enables a frozen language model to "bootstrap" its own logical geometry at runtime.

The system employs a lightweight, unsupervised **Watcher** module that observes internal activations during inference. By clustering the latent patterns of successful inferences ("wins"), the Watcher discovers emergent **Attractors**—geometric centroids representing implicit logical states (e.g., valid deductive steps). In subsequent inferences, the Watcher dynamically "snaps" (nudges) wandering activations toward these attractors. This creates a self-reinforcing loop where the model's latent space becomes increasingly structured without gradient updates to the base parameters.

**Pivot (Current Phase):** We have moved from a toy model to using **EleutherAI/pythia-70m** as the base model to validate the mechanism on a more capable foundation.

**Key Finding (Warmup Strategy):** To overcome the "Cold Start" problem (where a weak model cannot generate enough initial correct samples to form attractors), we implemented a **Supervised Warmup** phase. By initializing the Watcher with activations from a small set of correct logical examples, we successfully "primed" the geometric space, leading to a **+16% accuracy improvement** on synthetic logic tasks compared to the baseline.

## 2. System Architecture

The architecture consists of two distinct components: the **Frozen Base Model** (the subject) and the **Emergent Watcher** (the intervener).

### 2.1 Base Neural Network

- **Role:** The primary reasoning engine.
- **Model:** **EleutherAI/pythia-70m** (Pre-trained Causal LM).
- **Architecture:** Transformer Decoder (6 layers, 512 hidden dimension).
- **State:** **Frozen.** Weights are locked. No gradients flow through the base model during the EAS lifecycle.
- **Interface:** Exposes a read/write hook at the middle layer (Layer 3) to allow the Watcher to intercept and modify the hidden state tensor $H$.
- **Activation Access:** The base model provides access to the hidden state tensor $H$ of shape `[batch, seq_len, hidden_dim]`.

### 2.2 The Emergent Watcher

- **Role:** A runtime-only sidecar module that manages the lifecycle of Attractors and performs interventions.
- **Components:**
    1. **Attractor Memory:** A dynamic tensor $A \in \mathbb{R}^{K \times D}$ storing $K$ centroids (default $K=10$).
    2. **Whitening Buffer:** A running statistics module to normalize input activations, reducing noise and "distractor features."
    3. **Clustering Engine:** An online K-Means algorithm responsible for evolving $A$ based on successful outcomes.
    4. **Supervised Warmup:** A new initialization routine that populates $A$ with high-quality centroids derived from a small "Golden Set" of correct reasoning examples before the unsupervised loop begins.

---

## 3. Functional Specification

### 3.1 Phase 1: Signal Preprocessing

Raw activations often contain noise or high-frequency components irrelevant to logic.

- **Input:** Raw hidden state $H_{raw}$ of shape `[batch, seq_len, hidden_dim]`.
- **Pooling:** Apply attention-weighted pooling or simple mean pooling over the sequence dimension to obtain a sentence-level vector $v$.
- **Whitening:** Apply on-the-fly normalization using running mean $\mu$ and variance $\sigma^2$ to center and scale the signal:
$$v_{norm} = \frac{v - \mu}{\sigma + \epsilon}$$

### 3.2 Phase 2: Adaptive Snapping (Intervention)

The Watcher identifies the nearest logical structure and nudges the activation toward it.

- **Selection:** Compute Cosine Similarity between $v_{norm}$ and all Attractors in $A$. Identify the best matching attractor $A_{best}$ and the similarity score $S_{max}$.
- **Adaptive Strength ($\alpha$):** Unlike fixed steering, $\alpha$ is dynamic to prevent over-steering confident states or under-steering ambiguous ones.
$$\alpha_{dyn} = \alpha_{base} \times (1 - S_{max})$$
    - _Logic:_ If the activation is already very close to an attractor ($S_{max} \approx 1$), the nudge approaches zero.
- **Safety Clamp:** To prevent "off-manifold" hallucinations, the magnitude of the nudge is clamped.
$$\Delta = A_{best} - v_{norm}$$
$$v_{snapped} = v_{norm} + \text{clamp}(\alpha_{dyn} \times \Delta, -\delta, \delta)$$
- **Re-injection:** The modified vector $v_{snapped}$ is broadcast back to the sequence length (or added as a residual) and fed into the next layer of the Base Model.

### 3.3 Phase 3: Attractor Evolution (Update)

Attractors evolve only when the model succeeds, reinforcing valid reasoning paths.

- **Trigger:** Post-inference, the system checks the prediction against an Oracle Label.
- **Condition:** Update occurs **only** if `Prediction == Label`.
- **Mechanism:**
    1. Add the successful $v_{norm}$ to a batch buffer.
    2. Every $N$ wins (e.g., $N=5$), perform a partial fit using Online K-Means.
    3. **Sandwich Normalization:** Normalize centroids after the update to ensure they remain on the hypersphere, maintaining consistent cosine geometry.

---

## 4. Implementation

### 4.1 Models Module (`eas/src/models/`)
- **`transformer.py`**: Contains the `PretrainedTransformer` implementation wrapping Hugging Face models.
- **`legacy/`**: Contains the original toy model implementation (`AutoregressiveTransformer`, `LogicTokenizer`).

### 4.2 Watcher Module (`eas/src/watcher/`)
- **`__init__.py`**: Contains the `EmergentWatcher` implementation.

### 4.3 Datasets Module (`eas/src/datasets/`)
- **`__init__.py`**: Contains the `LogicCorpusGenerator`.

### 4.4 Experiments Module (`eas/src/experiments/`)
- **`__init__.py`**: Contains the `EASEvaluator` adapted for pre-trained models.

### 4.5 Advanced Validation (`eas/advanced_validation/`)
- **`train_and_evaluate.py`**: Main script for running validation on the pre-trained model.

---

## 5. Running the Experiments

### 5.1 Setup
```bash
pip install -r requirements.txt
pip install transformers accelerate scikit-learn
```

### 5.2 Running the Validation
```bash
./evaluate.sh
```

### 5.3 Quick Experiment
```python
from eas.src.main import main
main()
```
