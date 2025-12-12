# Next Steps: Productionizing EAS

With the validation phase complete and successful results in Efficiency and Safety, we move to production readiness.

## Validated Capabilities

- [x] **Efficiency**: `Pythia-410m` is the recommended base model (+12% acc vs GPT-2 Large).
- [x] **Safety**: Perplexity-based filtering detects common jailbreaks.
- [x] **Control**: EAS Steering allows intervention in latent space.

## Future Work

1.  **MCRE Stress Test**: Deploy MCRE on harder datasets (GSM8K) to validate abstention benefit.
2.  **Defense Hardening**: Integrate the Perplexity Defense into a production API middleware.
3.  **CoT Fine-Tuning**: Fine-tune `Pythia-410m` on a small dataset of "Reasoning Traces" to improve the coherence of injected CoT steps.
