# EAS Multi-Model Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.0267 | 0.0249 | - |
| Baseline | avicenna | 0.0000 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.2000 | 0.0283 | **+0.1733** |
| EAS_Standard | avicenna | 0.0000 | 0.0000 | **+0.0000** |
| EAS_Adversarial | complex_synthetic | 0.1400 | 0.0163 | **+0.1133** |

## Model: openai-community/gpt2

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.2867 | 0.0249 | - |
| Baseline | avicenna | 0.0000 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.1000 | 0.0327 | **-0.1867** |
| EAS_Standard | avicenna | 0.0000 | 0.0000 | **+0.0000** |
| EAS_Adversarial | complex_synthetic | 0.0933 | 0.0340 | **-0.1933** |

### Layer Sweep Analysis (GPT-2)

A comprehensive sweep of all 12 layers of GPT-2 was conducted to determine if the negative performance was due to incorrect intervention placement.

| Layer | Baseline Acc | EAS Acc | Improvement |
|---|---|---|---|
| 0 | 0.30 | 0.11 | -0.19 |
| 1 | 0.28 | 0.08 | -0.20 |
| 2 | 0.27 | 0.10 | -0.17 |
| 3 | 0.31 | 0.07 | -0.24 |
| 4 | 0.24 | 0.10 | -0.14 |
| 5 | 0.29 | 0.13 | -0.16 |
| 6 | 0.26 | 0.10 | -0.16 |
| 7 | 0.43 | 0.16 | -0.27 |
| 8 | 0.35 | 0.15 | -0.20 |
| 9 | 0.25 | 0.13 | **-0.12** |
| 10 | 0.30 | 0.17 | -0.13 |
| 11 | 0.32 | 0.13 | -0.19 |

**Conclusion:** The intervention consistently degrades performance across *all* layers. Layer 9 showed the least degradation (-0.12), but no layer showed positive transfer. This strongly suggests that the issue is not merely topological (where we intervene) but likely related to the intervention dynamics (how strong/what direction) or fundamental representational incompatibility.

### Multi-Model Architecture Sweep

To verify architecture sensitivity, we expanded validation to diverse model families.

| Model | Architecture | Baseline Acc | EAS Acc | Improvement | Status |
|---|---|---|---|---|---|
| **Pythia-70m** | GPT-NeoX (Parallel) | 0.0267 | 0.2000 | **+0.1733** | **SUCCESS** |
| **Pythia-160m** | GPT-NeoX (Parallel) | 0.0200 | 0.0900 | **+0.0700** | **Partial Success** |
| **GPT-2** | GPT-2 (Absolute) | 0.3000 | 0.1100 | **-0.1900** | Failure |
| **OPT-125m** | OPT (Learned Pos) | 0.5200 | 0.0200 | **-0.5000** | Catastrophic Failure |
| **Bloom-560m** | Bloom (ALiBi) | 0.3000 | 0.1200 | **-0.1800** | Failure |

**Key Findings:**
1.  **Scaling Law (Pythia):** Efficacy holds but diminishes with scale (70m -> 160m). This suggests the intervention might need to scale in complexity (K, dimension) with the model size.
2.  **Architecture Lock:** EAS currently *only* works on the GPT-NeoX architecture (Pythia).
    *   **OPT** (Open Pre-trained Transformer) suffered a massive 50% drop, effectively destroying its reasoning capabilities.
    *   **Bloom** (ALiBi embeddings) and **GPT-2** also degraded significantly.
3.  **Hypothesis:** The "Geometric Snapping" mechanism relies on a specific activation distribution found in NeoX models (perhaps related to their parallel attention/MLP block structure). Other architectures may encode logic in a way that is orthogonal or adversarial to the Watcher's simple Euclidean clustering.
