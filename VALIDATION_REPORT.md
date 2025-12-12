# EAS Validation Report

## Model: EleutherAI/pythia-70m

| Scenario | Dataset | Mean Acc | Std Dev | Improvement |
|---|---|---|---|---|
| Baseline | complex_synthetic | 0.3400 | 0.0748 | - |
| Baseline | avicenna | 0.1667 | 0.0000 | - |
| EAS_Standard | complex_synthetic | 0.3667 | 0.0189 | **+0.0267** |
| EAS_Standard | avicenna | 0.1852 | 0.0524 | **+0.0185** |
| EAS_Adversarial | complex_synthetic | 0.2400 | 0.0589 | **-0.1000** |

