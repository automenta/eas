# EAS Research Roadmap

This document outlines the strategic direction for the Emergent Activation Snapping (EAS) project following the successful pivot to pre-trained foundation models.

## 1. Immediate Scientific Priorities

### 1.1 Architecture Sensitivity Analysis (The "GPT-2 Puzzle")
*   **Goal:** Understand why EAS works on Pythia-70m (+17%) but degrades GPT-2 (-18%).
*   **Hypothesis:** The "middle layer" heuristic is insufficient. Different architectures store logical features at different depths.
*   **Tasks:**
    - [x] Implement an automated **Layer Sweep**: Run validation across *all* layers of GPT-2 to find the optimal intervention point. (Result: Negative across all layers. Best: Layer 9).
    - [ ] Implement an **Alpha Sweep**: Test varying intervention strengths ($\alpha \in [0.1, 0.9]$) to see if GPT-2 requires subtler nudges.
    - [ ] Compare attention head usage between Pythia and GPT-2 during logical inference.

### 1.2 Solving Real-World Transfer (The "Avicenna Gap")
*   **Goal:** Move Avicenna accuracy from 0% to >10%.
*   **Hypothesis:** The current "Supervised Warmup" uses synthetic logic (A->B), which creates attractors geometrically distinct from natural language NLI (Premise->Conclusion).
*   **Tasks:**
    - [ ] **Curriculum Warmup:** Create a "Bridge Dataset" that mixes formal logic symbols with natural language templates to smooth the geometric transition.
    - [ ] **Prompt Engineering:** Test more diverse prompt templates ("True/False", "Does it follow?", etc.) to find a better base-state for the model.
    - [ ] **Enriched Attractors:** Warmup the watcher with a small subset of general NLI data (e.g., MNLI or SNLI) before testing on Avicenna.

### 1.3 Geometric Interpretability
*   **Goal:** Prove *why* it works by visualizing the latent space.
*   **Tasks:**
    - [ ] **PCA/t-SNE Visualization:** Plot the trajectory of activations for "Correct" vs "Incorrect" inferences. Show how the Watcher "snaps" the trajectory.
    - [ ] **Attractor Decoding:** Use the `logit_lens` technique to project the Attractor Centroids back into vocabulary space. What words do the attractors represent? (e.g., do they map to "True", "Therefore", "Yes"?).

## 2. Engineering & Infrastructure

### 2.1 Configuration Management
*   **Current State:** Hyperparameters are hardcoded in scripts.
*   **Task:** Migrate to `argparse` or `Hydra` for easy configuration of:
    - Model Name
    - Layer Index
    - Watcher Specs (K, alpha, momentum)
    - Warmup Size

### 2.2 Testing & Stability
*   **Current State:** Basic smoke tests and validation suite.
*   **Task:**
    - [ ] Add unit tests for `EmergentWatcher` (clustering logic, buffer updates).
    - [ ] Add regression tests to ensure Pythia-70m performance remains >15% improvement.

## 3. Long-Term Vision (Scaling)

*   **Goal:** Verify EAS efficacy on larger models.
*   **Target Models:**
    - Pythia-160m, Pythia-410m (Scaling laws for EAS).
    - TinyLlama-1.1B (Introduction of GQA and modern architecture).
    - BERT (Encoder-only architecture test).

## 4. Paper/Publication Targets

*   **Narrative:** "Unsupervised Geometric Steering for Logical Reasoning in Frozen LLMs."
*   **Key Claims to Validate:**
    1.  Attractors represent "Logical Types" (Modus Ponens, etc.).
    2.  Intervention works across model families (if tuned).
    3.  Data-efficient adaptation (Warmup is cheaper than Fine-tuning).
