#!/bin/bash
# Setup script for Emergent Activation Snapping (EAS) experiment

echo "Setting up EAS experiment environment..."

# Create logs directory
mkdir -p logs

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the small-scale EAS validation experiment, use:"
echo "python -m eas.src.main"