#!/bin/bash
# M2S-Guardrail 모델 다운로드 스크립트
# Llama Guard 3/4 모델을 로컬에 미리 다운로드

set -e

echo "=== M2S-Guardrail Model Download ==="

# HuggingFace Hub 설치 확인
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface-hub..."
    pip install huggingface-hub
fi

# 모델 저장 디렉터리 생성
MODELS_DIR="./models"
mkdir -p $MODELS_DIR

echo "Creating models directory: $MODELS_DIR"

# Llama Guard 3 (8B) 다운로드
echo ""
echo "📥 Downloading Llama Guard 3 (8B)..."
echo "This may take 10-20 minutes depending on your internet speed..."

huggingface-cli download meta-llama/Llama-Guard-3-8B \
    --local-dir $MODELS_DIR/Llama-Guard-3-8B \
    --local-dir-use-symlinks False

echo "✅ Llama Guard 3 (8B) downloaded to: $MODELS_DIR/Llama-Guard-3-8B"

# 다운로드 확인
if [ -d "$MODELS_DIR/Llama-Guard-3-8B" ]; then
    echo "📊 Model files:"
    ls -lah $MODELS_DIR/Llama-Guard-3-8B/
    
    # 모델 크기 확인
    echo ""
    echo "📈 Total size:"
    du -sh $MODELS_DIR/Llama-Guard-3-8B/
else
    echo "❌ Download failed"
    exit 1
fi

# Llama Guard 4 (12B) 다운로드 여부 확인
echo ""
read -p "Also download Llama Guard 4 (12B)? This will take additional 20-30 minutes [y/N]: " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading Llama Guard 4 (12B)..."
    
    huggingface-cli download meta-llama/Llama-Guard-4-12B \
        --local-dir $MODELS_DIR/Llama-Guard-4-12B \
        --local-dir-use-symlinks False
    
    echo "✅ Llama Guard 4 (12B) downloaded to: $MODELS_DIR/Llama-Guard-4-12B"
    
    if [ -d "$MODELS_DIR/Llama-Guard-4-12B" ]; then
        echo "📊 Model files:"
        ls -lah $MODELS_DIR/Llama-Guard-4-12B/
        
        echo ""
        echo "📈 Total size:"
        du -sh $MODELS_DIR/Llama-Guard-4-12B/
    fi
fi

# 전체 용량 확인
echo ""
echo "📋 Download Summary:"
echo "Models directory: $MODELS_DIR"
echo "Total size: $(du -sh $MODELS_DIR | cut -f1)"

echo ""
echo "✅ Model download completed!"
echo ""
echo "Next steps:"
echo "1. Set up conda environment: bash setup_conda_env.sh"
echo "2. Test environment: python test_environment.py"
echo "3. Run experiments: python train_experiments.py"