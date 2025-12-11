# EAS Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.3400 | 0.0748 | - |
| Baseline | avicenna | 0.1667 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.2867 | 0.0340 | **-0.0533** |
| EAS_Standard | avicenna | 0.0185 | 0.0262 | **-0.1481** |
| EAS_Adversarial | complex_synthetic | 0.3467 | 0.0838 | **+0.0067** |
