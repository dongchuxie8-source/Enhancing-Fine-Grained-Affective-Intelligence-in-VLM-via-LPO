#!/bin/bash
# Environment setup script for LPO-Emotion-VLM

set -e

echo "================================="
echo "  LPO-Emotion-VLM Environment Setup"
echo "================================="

# 1. Create virtual environment
echo "[1/4] Creating conda environment..."
conda create -n lpo-vlm python=3.10 -y
source activate lpo-vlm || conda activate lpo-vlm

# 2. Install PyTorch (CUDA 11.8)
echo "[2/4] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install project dependencies
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt

# 4. Create necessary directories
echo "[4/4] Creating project directories..."
mkdir -p data/images
mkdir -p outputs
mkdir -p logs

echo ""
echo "================================="
echo "  Setup complete!"
echo "================================="
echo ""
echo "Next steps:"
echo "  1. Run 'wandb login' to configure experiment tracking"
echo "  2. Prepare AffectNet data and run:"
echo "     python data/build_preference.py --input_csv <annotations.csv>"
echo "  3. Train SFT baseline:"
echo "     bash scripts/train_sft.sh"
echo "  4. Train LPO model:"
echo "     bash scripts/train_lpo.sh"
echo "  5. Evaluate:"
echo "     bash scripts/evaluate.sh"
