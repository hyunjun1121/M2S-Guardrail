#!/bin/bash
# GPU 서버용 M2S-Guardrail Conda 환경 설정
# /eic/data에 conda 및 캐시 디렉터리 설정

set -e

echo "=== M2S-Guardrail GPU Server Environment Setup ==="

# 사용자 확인
read -p "어느 GPU 디렉터리를 사용하시겠습니까? (hkim_gpu2/hkim_gpu3/slee_gpu2/slee_gpu3): " gpu_dir

# 유효성 검사
valid_dirs=("hkim_gpu2" "hkim_gpu3" "slee_gpu2" "slee_gpu3")
if [[ ! " ${valid_dirs[@]} " =~ " ${gpu_dir} " ]]; then
    echo "❌ Invalid directory. Please choose from: ${valid_dirs[*]}"
    exit 1
fi

# 설정 변수
EIC_DATA_PATH="/eic/data/${gpu_dir}"
CONDA_DIR="${EIC_DATA_PATH}/miniconda3"
ENV_NAME="m2s-guardrail"

echo "🔧 Setting up environment in: ${EIC_DATA_PATH}"

# 디렉터리 확인 및 생성
if [ ! -d "$EIC_DATA_PATH" ]; then
    echo "❌ Directory $EIC_DATA_PATH does not exist"
    exit 1
fi

echo "✅ Using directory: $EIC_DATA_PATH"

# Miniconda 설치 (존재하지 않는 경우)
if [ ! -d "$CONDA_DIR" ]; then
    echo "📥 Installing Miniconda to $CONDA_DIR..."
    
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $CONDA_DIR
    rm miniconda.sh
    
    echo "✅ Miniconda installed"
else
    echo "✅ Miniconda already exists at $CONDA_DIR"
fi

# Conda 초기화 및 PATH 설정
export PATH="$CONDA_DIR/bin:$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 환경 생성
echo "🔨 Creating conda environment: $ENV_NAME"
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists. Removing..."
    conda env remove -n $ENV_NAME -y
fi

conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# 캐시 디렉터리를 /eic/data로 설정
export HF_HOME="${EIC_DATA_PATH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${EIC_DATA_PATH}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${EIC_DATA_PATH}/.cache/huggingface/datasets"

# 캐시 디렉터리 생성
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE
mkdir -p $HF_DATASETS_CACHE

echo "📦 Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "📦 Installing ML packages..."
pip install transformers>=4.36.0
pip install datasets>=2.14.0
pip install accelerate>=0.24.0
pip install peft>=0.6.0
pip install bitsandbytes>=0.41.0
pip install scipy
pip install scikit-learn

# Excel 처리
pip install pandas openpyxl

# HuggingFace Hub
pip install huggingface-hub

# Flash Attention (성능 향상, 설치 실패해도 계속 진행)
echo "📦 Installing Flash Attention (optional)..."
pip install flash-attn --no-build-isolation || echo "⚠️ Flash Attention 설치 실패 (선택사항이므로 계속 진행)"

echo "✅ Package installation completed"

# 환경 설정 파일 생성
cat > "${EIC_DATA_PATH}/activate_env.sh" << EOF
#!/bin/bash
# M2S-Guardrail 환경 활성화 스크립트

export PATH="$CONDA_DIR/bin:\$PATH"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# 캐시 디렉터리 설정
export HF_HOME="${EIC_DATA_PATH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${EIC_DATA_PATH}/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="${EIC_DATA_PATH}/.cache/huggingface/datasets"

echo "✅ M2S-Guardrail 환경이 활성화되었습니다"
echo "Working directory: \$(pwd)"
echo "Python: \$(which python)"
echo "Cache directory: \$HF_HOME"
EOF

chmod +x "${EIC_DATA_PATH}/activate_env.sh"

echo "🎯 Environment Setup Complete!"
echo ""
echo "📝 Next steps:"
echo "1. 환경 활성화: source ${EIC_DATA_PATH}/activate_env.sh"
echo "2. 프로젝트 디렉터리로 이동"
echo "3. 모델 다운로드: bash download_models_gpu_server.sh"
echo "4. 환경 테스트: python test_environment.py"
echo ""
echo "💡 Tips:"
echo "- 모든 모델과 캐시가 ${EIC_DATA_PATH}에 저장됩니다"
echo "- 환경 활성화 스크립트: ${EIC_DATA_PATH}/activate_env.sh"