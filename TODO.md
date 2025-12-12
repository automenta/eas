# Next Steps: scaling to Production

With the completion of the proof-of-concept phase, we have a mixed set of results. While the model upgrade to `Pythia-410m` proved beneficial, the **MCRE hypothesis remains unproven** on the current dataset.

## Immediate Actions

1.  **Re-Evaluate MCRE on Harder Tasks**:
    *   **CRITICAL FAILURE**: MCRE did not trigger abstention on `logiqa`. The model was too confident (Z-score < 0.5).
    *   **Action**: Run `david_vs_goliath.py` on a dataset where `Pythia-410m` has <30% accuracy (e.g., GSM8K or hard synthetic logic). We need the model to be *wrong* and *uncertain* for MCRE to demonstrate value.

2.  **Fine-tune CoT Injection**:
    *   The repetition issue in PoC 3 ("John has 5 First...") suggests the injection needs to be context-aware (replace the previous token instead of inserting?) or the model needs a "reasoning fine-tuning" on the prompts.
    *   **Action**: Experiment with `Phi-2` (2.7B) if memory permits, as it has strong native CoT capabilities.

3.  **Generalize Adaptive MCRE**:
    *   The calibration phase (n=30) worked well technically but needs to be tuned to a sensitivity that actually catches errors.

## Roadmap

- [x] Validate Model Uplift (Success: Pythia-410m +12% over GPT-2 Large).
- [ ] **Validate MCRE Benefit** (FAILED/NULL: No improvement over baseline).
- [x] Validate EAS Steering (Success: 100% impact).
- [ ] Find "Goldilocks" dataset where Small Model fails but is uncertain.
