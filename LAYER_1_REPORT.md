# EAS Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.3400 | 0.0748 | - |
| Baseline | avicenna | 0.1667 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.5733 | 0.0573 | **+0.2333** |
| EAS_Standard | avicenna | 0.1481 | 0.0262 | **-0.0185** |
| EAS_Adversarial | complex_synthetic | 0.4200 | 0.0163 | **+0.0800** |
