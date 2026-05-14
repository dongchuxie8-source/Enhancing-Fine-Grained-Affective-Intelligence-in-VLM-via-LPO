#!/bin/bash
# 环境配置脚本

set -e

echo "========================="
echo "LPO-Emotion-VLM 环境配置"
echo "=================="

# 1. 创建虚拟环境
echo "[1/5] 创建conda虚拟环境..."
conda create -n lpo-vlm python=3.10 -y
source activate lpo-vlm || conda activate lpo-vlm

# 2. 安装PyTorch (CUDA 11.8)
echo "[2/5] 安装PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. 安装项目依赖
echo "[3/5] 安装项目依赖..."
pip install -r requirements.txt

# 4. 克隆LLaVA仓库（可选，用于参考实现）
echo "[4/5] 克隆LLaVA仓库..."
if [ ! -d "LLaVA" ]; then
    git clone https://github.com/haotian-liu/LLaVA.git
    cd LLaVA && pip install -e . && cd ..
fi

# 5. 创建必要的目录
echo "[5/5] 创建项目目录..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

echo "======================================"
echo "环境配置完成！"
echo "======================"
echo ""
echo "下一步："
echo "1. 运行 'wandb login' 登录WandB"
echo "2. 运行 'python data/download_fer2013.py' 下载数据"
echo "3. 运行 'python data/build_preference.py' 构建偏好数据集"
echo "4. 运行 'python training/train_lpo.py' 开始训练"
