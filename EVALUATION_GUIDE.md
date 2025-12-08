# EAS Evaluation Guide

There are several scripts in this repository. This guide clarifies which one to use.

## **Recommended Evaluation**

### `evaluate.sh`
**Use this.** This is the unified entry point that runs the **Advanced Validation Framework**.
It executes `run_advanced_validation.py` which:
1.  Tests on Real Data (Avicenna dataset).
2.  Tests on Complex Synthetic Data (Modus Tollens, etc.).
3.  Generates `VALIDATION_REPORT.md` with an honest assessment of effectiveness.

## Legacy Scripts (Superseded)

The following scripts were part of previous validation attempts that suffered from result homogeneity or lacked real-world data coverage. They are kept for archival purposes but should not be used for primary evaluation.

*   `comprehensive_validation_suite.py`: Older suite focusing on synthetic data only.
*   `multi_dataset_validation_suite.py`: Attempted to vary datasets but lacked true logic diversity.
*   `robust_validation_suite.py`: Focused on noise resistance on simple tasks.
*   `comprehensive_smoke_test.py`: Quick check for crashes, not for scientific validity.
*   `run_experiment.sh`: Old runner for the basic experiment.

## How to Run

Simply execute:
```bash
./evaluate.sh
```
This will print the results and the final assessment report to your console.
