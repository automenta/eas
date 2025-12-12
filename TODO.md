# Next Steps: Maximizing Research Benefit

Based on the results from the `eas-proof-of-concept` validation, we have identified that while the mechanisms (EAS, MCRE) are functional, the current model scale (Pythia-70m) is the primary bottleneck. To maximize research benefit and publishability, we must transition to capable base models.

## Immediate Actions (High Priority)

1.  **Upgrade Base Models**:
    *   Transition from `EleutherAI/pythia-70m` to `microsoft/phi-2` (2.7B) or `TinyLlama/TinyLlama-1.1B`. These models fit in consumer memory but possess actual reasoning capabilities.
    *   Re-run "David vs Goliath" with Phi-2 (David) vs Llama-2-7B (Goliath).

2.  **Refine MCRE Thresholding**:
    *   Implement **Adaptive Thresholding**: Instead of a fixed scalar (e.g., 0.95), normalize entropy against the model's baseline entropy on a held-out "easy" dataset.
    *   Implement "Uncertainty calibration" phase: Run 50 examples to determine the distribution of entropy for correct vs incorrect answers, then set threshold at the intersection.

3.  **Targeted CoT Steering**:
    *   The current CoT injection ("First, let's consider...") is too generic.
    *   **Action**: Fine-tune the "elaboration phrases" based on the specific question type (e.g., for math: "Let's calculate..."; for logic: "Let's trace the premises...").

## Research Roadmap

### Phase 1: Capability Uplift
- [ ] Swap Pythia-70m for Phi-2 in `david_vs_goliath.py`.
- [ ] Run the validation suite and target >60% effective accuracy.

### Phase 2: EAS tuning
- [ ] Implement "Contrastive Warmup" for EAS: Instead of just warming up on correct answers, explicitly push *away* from incorrect answer activations.
- [ ] Run `reproduce_context_aligned_eas.py` on the full LogiQA dataset (not just 20 samples) to get statistical significance.

### Phase 3: Publication
- [ ] Generate clean plots of "Accuracy vs Abstention Rate".
- [ ] Document the "Steering Vector" geometry: Visualize (via PCA) how the intervention shifts the latent state.
