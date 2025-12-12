#!/bin/bash
# Execute the complete EAS experiment pipeline

echo "Emergent Activation Snapping (EAS) - Complete Experiment Pipeline"
echo "=================================================================="

# Check if required dependencies are available
echo "Checking dependencies..."
python -c "import torch; import numpy; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages not found. Please install dependencies:"
    echo "pip install -r requirements.txt"
    exit 1
fi

# Create results directory
RESULTS_DIR="experiment_results_$(date +%Y%m%d_%H%M%S)"
echo "Creating results directory: $RESULTS_DIR"

# Run the complete experiment
echo ""
echo "Starting complete EAS experiment pipeline..."
echo "This may take 30-60 minutes depending on hardware..."
echo ""

python run_complete_experiment.py --output-dir "$RESULTS_DIR"

echo ""
echo "=================================================================="
echo "Experiment completed!"
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "To view the final report:"
echo "cat $RESULTS_DIR/results/final_report_*.json"
echo ""
echo "To view experiment logs:"
echo "cat $RESULTS_DIR/logs/*.log"
echo "=================================================================="