# Next Steps: Scaling to Production

With the successful validation of the "David vs Goliath" hypothesis using Pythia-410m, the research direction shifts from *validation* to *scaling*.

## Immediate Actions

1.  **Fine-tune CoT Injection**:
    *   The repetition issue in PoC 3 ("John has 5 First...") suggests the injection needs to be context-aware (replace the previous token instead of inserting?) or the model needs a "reasoning fine-tuning" on the prompts.
    *   **Action**: Experiment with `Phi-2` (2.7B) if memory permits, as it has strong native CoT capabilities.

2.  **Generalize Adaptive MCRE**:
    *   The calibration phase (n=30) worked well.
    *   **Action**: Implement an online calibration that updates $\mu, \sigma$ continuously during inference (sliding window).

3.  **Publish Findings**:
    *   The +12% accuracy gain (64% vs 52%) is a strong result.
    *   Prepare a blog post or paper draft focusing on "Efficiency > Scale".

## Roadmap

- [x] Validate MCRE on Pythia-410m (Success: +12% over GPT-2 Large).
- [x] Validate EAS Steering (Success: 100% impact).
- [ ] Implement Online Calibration for MCRE.
- [ ] Optimize CoT prompts to reduce repetition.
