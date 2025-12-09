# Master Specification: Emergent Activation Snapping (EAS)

## 1. Executive Summary

Emergent Activation Snapping (EAS) is a neuro-symbolic intervention framework designed to bridge the gap between distributed neural representations and crisp logical reasoning. Unlike traditional methods that rely on supervised steering vectors, explicit symbolic encoders, or extensive fine-tuning, EAS enables a frozen language model to "bootstrap" its own logical geometry at runtime.

The system employs a lightweight, unsupervised **Watcher** module that observes internal activations during inference. By clustering the latent patterns of successful inferences ("wins"), the Watcher discovers emergent **Attractors**—geometric centroids representing implicit logical states (e.g., valid deductive steps). In subsequent inferences, the Watcher dynamically "snaps" (nudges) wandering activations toward these attractors. This creates a self-reinforcing loop where the model's latent space becomes increasingly structured without gradient updates to the base parameters.

**Pivot (Current Phase):** We have moved from a toy model to using **Pre-trained Foundation Models** (Pythia-70m, GPT-2) as the base model to validate the mechanism on a more capable foundation.

**Key Findings:**
1.  **Supervised Warmup:** To overcome the "Cold Start" problem, we implemented a Supervised Warmup phase. This successfully primed the geometric space for Pythia-70m.
2.  **Model Sensitivity:** Validation on **Pythia-70m** showed a **+17% accuracy improvement** on synthetic logic. However, the same setup on **GPT-2** resulted in performance degradation (-18%). This indicates that the EAS mechanism (specifically layer selection and attractor formation) is highly sensitive to the underlying model architecture and requires model-specific tuning.

## 2. System Architecture

The architecture consists of two distinct components: the **Frozen Base Model** (the subject) and the **Emergent Watcher** (the intervener).

### 2.1 Base Neural Network

- **Role:** The primary reasoning engine.
- **Models Tested:**
    - **EleutherAI/pythia-70m** (Success case).
    - **openai-community/gpt2** (Failure case / Negative control).
- **State:** **Frozen.** Weights are locked. No gradients flow through the base model during the EAS lifecycle.
- **Interface:** Exposes a read/write hook at the middle layer to allow the Watcher to intercept and modify the hidden state tensor $H$.

### 2.2 The Emergent Watcher

- **Role:** A runtime-only sidecar module that manages the lifecycle of Attractors and performs interventions.
- **Components:**
    1. **Attractor Memory:** A dynamic tensor $A \in \mathbb{R}^{K \times D}$ storing $K$ centroids (default $K=10$).
    2. **Whitening Buffer:** A running statistics module to normalize input activations, reducing noise and "distractor features."
    3. **Clustering Engine:** An online K-Means algorithm responsible for evolving $A$ based on successful outcomes.
    4. **Supervised Warmup:** A new initialization routine that populates $A$ with high-quality centroids derived from a small "Golden Set" of correct reasoning examples before the unsupervised loop begins.

---

## 3. Implementation & Pivot

### 3.1 New Components
- **`eas/src/models/transformer.py`**: Contains the `PretrainedTransformer` implementation wrapping Hugging Face models.
- **`eas/advanced_validation/suite.py`**: Updated `AdvancedValidationSuite` supporting:
    - Multi-trial robustness testing (Mean ± Std Dev).
    - Data splitting (Warmup vs. Test).
    - Dynamic model injection.

### 3.2 Legacy Components
- **`eas/src/models/legacy/`**: Contains the original toy model implementation, preserved for reference.

---

## 4. Running the Validation

### 4.1 Setup
```bash
pip install -r requirements.txt
pip install transformers accelerate scikit-learn
```

### 4.2 Running the Multi-Model Validation
```bash
./evaluate.sh
```
This script runs the comprehensive 3-trial validation suite on both Pythia-70m and GPT-2, generating `VALIDATION_REPORT.md`.

---

## 5. Experimental Results (Multi-Model)

The latest rigorous validation (3 trials per model) yielded the following:

### EleutherAI/pythia-70m
- **Baseline Accuracy:** ~2.6%
- **EAS Standard Accuracy:** ~20.0%
- **Improvement:** **+17.3%**
- **Conclusion:** Strong validation of the EAS mechanism when properly warmed up.

### openai-community/gpt2
- **Baseline Accuracy:** ~28.6%
- **EAS Standard Accuracy:** ~10.0%
- **Improvement:** **-18.6%**
- **Conclusion:** Negative result. The intervention disrupted the model's native reasoning capabilities. This suggests that the "middle layer" heuristic may not apply universally, or that GPT-2's activation space requires different normalization or alpha parameters.

**Scientific Significance:**
We have demonstrated that EAS is **not a fluke** (it works reliably on Pythia across trials) but also **not a magic bullet** (it requires tuning for different architectures). This establishes a clear path for future research: conducting a Sensitivity Analysis on intervention layers and alpha strength across model families.
