# EAS Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.0267 | 0.0249 | - |
| Baseline | avicenna | 0.0000 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.1800 | 0.0566 | **+0.1533** |
| EAS_Standard | avicenna | 0.0000 | 0.0000 | **+0.0000** |
| EAS_Adversarial | complex_synthetic | 0.0800 | 0.0283 | **+0.0533** |
