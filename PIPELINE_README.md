# Complete EAS Experiment Pipeline

This turn-key script automates the entire Emergent Activation Snapping (EAS) experiment process from start to finish, with full transparency and observability.

## Overview

The `run_complete_experiment.py` script orchestrates the complete EAS experimental pipeline:

1. **Small Model Validation** - Pre-screening phase to assess basic efficacy
2. **Decision Logic** - Automated decision whether to proceed to standard model  
3. **Standard Model Experiment** - Full experiment if small model shows promise
4. **Comprehensive Analysis** - Statistical analysis and validation
5. **Automated Reporting** - Final report generation with recommendations

## Usage

### Basic Execution
```bash
cd /home/me/eas
python run_complete_experiment.py
```

### Custom Output Directory
```bash
python run_complete_experiment.py --output-dir my_custom_results
```

### Skip Small Model Validation (if already validated)
```bash
python run_complete_experiment.py --skip-small
```

## Features

### Transparency
- Real-time progress logging with timestamps
- Clear decision points and reasoning
- Detailed metrics collection at each stage
- Reproducible results with fixed random seeds

### Observability  
- Progress indicators at each experimental stage
- Performance metrics displayed as they're collected
- Intermediate results saved throughout
- Resource usage monitoring

### Automation
- Sequential experiment execution without manual intervention
- Pre-programmed decision trees based on success criteria
- Automatic result aggregation and analysis
- Standardized report generation

### Results Organization
All results are organized in a structured directory:

```
experiment_results/
├── results/           # Final experiment results in JSON format
├── logs/             # Detailed execution logs
├── plots/            # Visualizations (if matplotlib is available)
└── final_report_*.json  # Comprehensive final analysis
```

## Success Criteria

The pipeline evaluates success based on:

- **Accuracy Improvement**: ≥20% improvement over baseline within specified iterations
- **Latency Overhead**: <5% additional inference time
- **Attractor Stability**: Convergence to stable geometric structures
- **Robustness**: No mode collapse or system instability

## Output Files

- `complete_results_*.json` - Full experiment results with all metrics
- `final_report_*.json` - Executive summary with recommendations  
- Individual log files for each experimental stage
- Metric history for analysis and visualization

## Decision Logic

The pipeline implements the following decision rules:

1. If small model improvement ≥15%, proceed to standard model
2. If standard model shows ≥20% improvement, recommend extended experiments
3. If improvement is moderate (10-20%), recommend parameter optimization
4. If improvement is <10%, recommend alternative approaches

## Reproducibility

The script ensures reproducibility with:
- Fixed random seeds (42) for torch, numpy, and python random
- Complete logging of hyperparameters and system information
- Version tracking of key components
- Git commit hash logging (if available)

This turn-key solution provides the complete EAS experimental framework with full automation, transparency, and observability for reliable scientific evaluation.