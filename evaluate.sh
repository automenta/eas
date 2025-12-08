#!/bin/bash
# Unified Evaluation Script for EAS
# This script runs the Advanced Validation Suite which addresses previous limitations
# regarding real data (Avicenna) and complex synthetic logic.

echo "========================================================"
echo "      EAS Advanced Validation & Evaluation Runner"
echo "========================================================"
echo ""
echo "This script runs the new validation framework designed to provide"
echo "an honest assessment of the EAS approach using both real and"
echo "complex synthetic datasets."
echo ""
echo "Target: eas/advanced_validation/"
echo "Reports: VALIDATION_REPORT.md"
echo ""

# Ensure we are in the root directory
if [ ! -f "run_advanced_validation.py" ]; then
    echo "Error: run_advanced_validation.py not found in current directory."
    echo "Please run this script from the project root."
    exit 1
fi

# Set PYTHONPATH to include the current directory
export PYTHONPATH=$PYTHONPATH:.

echo "Starting Validation Run..."
echo "--------------------------------------------------------"

# Run the python script
python run_advanced_validation.py

echo "--------------------------------------------------------"
echo "Evaluation Complete."
echo ""

if [ -f "VALIDATION_REPORT.md" ]; then
    echo "Displaying Report Summary:"
    echo "========================================================"
    cat VALIDATION_REPORT.md
    echo "========================================================"
    echo "Full report saved to VALIDATION_REPORT.md"
else
    echo "Error: Report generation failed."
fi
