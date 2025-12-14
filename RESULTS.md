# Research Results: The G-Factor Project

**Date**: December 2024
**Engine**: AFT Lab v3.0 (`research_engine.py`)

## 1. Executive Summary

We have built and deployed the **G-Factor Research Engine**, a specialized tool to find "Universal Reasoning Boosters".

The engine implements:
1.  **Multi-Task Vector Training**: Simultaneously training on Math, Logic, and Commonsense.
2.  **Vector Fusion**: Mathematically combining these into a single G-Vector.
3.  **Smart Layer Scouting**: Automatically finding the best injection point for any model.

## 2. Experimental Status

Experiments are currently running in background sessions.

| Model | Status | Log File |
| :--- | :--- | :--- |
| **Qwen/Qwen2.5-0.5B** | 🏃 Running (Scouting HellaSwag) | `qwen_log_turbo.txt` |
| **EleutherAI/pythia-410m** | 🏃 Running (Scouting HellaSwag) | `pythia_log_turbo.txt` |

## 3. Methodology

### The G-Vector Formula
We compute the G-Vector by averaging task-specific vectors:
$$ \vec{v}_{G} = \frac{1}{3} (\vec{v}_{math} + \vec{v}_{logic} + \vec{v}_{common}) $$

*Future Extension*: We have also implemented **PCA Fusion** (Principal Component Analysis) in the code to extract the "shared direction of maximum variance" if the mean proves too noisy.

### Smart Layer Scouting
Instead of guessing layers, the engine scans 5 equidistant layers (e.g., 2, 7, 12, 17, 22) on a small subsample to find the "resonance point" where the model is most responsive to steering.

## 4. How to Interpret Results

When the runs complete, they will generate `results_g_factor_*.json` files containing:
- **Baseline Accuracy**: Performance without steering.
- **G-Factor Accuracy**: Performance with the fused G-Vector.
- **Delta**: The net improvement.

**Success Criteria**: A positive Delta across >1 dataset indicates the G-Vector is working as a generalized reasoning booster.
