#!/bin/bash

# Vast.ai / Cloud GPU Setup Script for BrainDed/Proj2
# Usage: ./setup_vast.sh

echo "Setting up environment on Cloud GPU..."

# 1. Update System & Install Basics
apt-get update && apt-get install -y git python3-pip unzip libgl1-mesa-glx libglib2.0-0

# 2. Install PyTorch (CUDA 12.8 Nightly for RTX 5090 / Blackwell)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install Project Requirements
pip install -r requirements.txt
# proj2 requirements if separate
if [ -f "proj2/requirements.txt" ]; then
    pip install -r proj2/requirements.txt
fi
pip install kagglehub timm transformers pandas numpy tqdm scikit-learn

# 4. Set Config for Full Training (Disable Hackathon Limit)
echo "Configuring for Full Training..."
# We use sed to remove the MAX_SAMPLES line or set it to None
sed -i 's/MAX_SAMPLES = 500/MAX_SAMPLES = None/' proj2/config.py

echo "Setup Complete!"
echo "Run training with: python -m proj2.src.train"
