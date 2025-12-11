# EAS Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.3400 | 0.0748 | - |
| Baseline | avicenna | 0.1667 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.4667 | 0.0499 | **+0.1267** |
| EAS_Standard | avicenna | 0.0926 | 0.0693 | **-0.0741** |
| EAS_Adversarial | complex_synthetic | 0.4000 | 0.0864 | **+0.0600** |
