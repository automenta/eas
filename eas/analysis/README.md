# EAS Analysis Module

This module contains analysis scripts and results from the EAS research project.

## Structure

```
analysis/
├── scripts/          # Runnable analysis scripts
├── results/          # JSON output from experiments
└── findings/         # Research findings documentation
```

## Quick Start

```bash
cd /home/me/eas

# Run layer-by-layer divergence analysis
python eas/analysis/scripts/run_layer_analysis.py

# Test on grammar/semantic/arithmetic tasks
python eas/analysis/scripts/run_clean_analysis.py

# Full geometric validation
python eas/analysis/scripts/run_geometric_validation.py
```

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `run_layer_analysis.py` | Per-layer cosine/distance analysis | `layer_analysis_results.json` |
| `run_magnitude_analysis.py` | Activation norm comparison | `magnitude_analysis_results.json` |
| `run_clean_analysis.py` | Multi-task (grammar, semantic, arithmetic) | `clean_analysis_results.json` |
| `run_geometric_validation.py` | Attractor alignment + snapping tests | `geometric_validation_results.json` |
| `run_breakthrough_test.py` | Full experiment with Pythia model | `experiment_results_*.json` |

## Key Finding

> Pythia-70m shows <0.5% cosine divergence between correct/incorrect text representations, limiting EAS effectiveness on small models.

See `findings/FINDINGS.md` for detailed results.
