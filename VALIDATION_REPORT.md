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

