#!/bin/bash
# M2S-Guardrail Conda 환경 설정 스크립트
# RTX A5000 8-GPU 환경에서 QLoRA 훈련을 위한 환경 구성

set -e  # 에러 발생 시 스크립트 중단

echo "=== M2S-Guardrail Conda Environment Setup ==="

# 환경 변수 설정
ENV_NAME="m2s-guardrail"
PYTHON_VERSION="3.10"

# 기존 환경 제거 (선택사항)
read -p "Remove existing environment '$ENV_NAME' if it exists? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Removing existing environment..."
    conda remove -n $ENV_NAME --all -y || true
fi

echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing core ML packages..."
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install peft>=0.6.0
pip install bitsandbytes>=0.41.0

echo "Installing additional dependencies..."
pip install pandas>=1.5.0
pip install openpyxl>=3.1.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install psutil
pip install tqdm
pip install wandb  # 선택적 로깅
pip install flash-attn --no-build-isolation  # Flash Attention 2

echo "Installing development tools..."
pip install jupyter
pip install matplotlib
pip install seaborn

# CUDA 환경 테스트
echo "Testing CUDA installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Transformers 테스트
echo "Testing transformers installation..."
python -c "
from transformers import AutoTokenizer
from peft import LoraConfig
from bitsandbytes import functional as F
print('All packages imported successfully!')
"

echo "Environment setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "    conda activate $ENV_NAME"
echo ""
echo "To test the setup, run:"
echo "    python test_environment.py"