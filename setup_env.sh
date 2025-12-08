#!/bin/bash
# Setup script for coverletter-rag environment

echo "Setting up coverletter-rag environment..."
echo ""

# Activate conda environment
echo "Activating conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate coverletter-rag

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To use this environment in the future, run:"
echo "  conda activate coverletter-rag"
echo ""
echo "Then you can run:"
echo "  streamlit run app_streamlit.py"

