#!/bin/bash
# GPU 서버용 M2S-Guardrail 모델 다운로드 스크립트
# /eic/data 환경에 맞춰 모델을 다운로드

set -e

echo "=== M2S-Guardrail GPU Server Model Download ==="

# GPU 디렉터리 확인
if [ -z "$HF_HOME" ]; then
    echo "❌ HF_HOME 환경 변수가 설정되지 않았습니다."
    echo "먼저 환경을 활성화하세요: source /eic/data/[your_dir]/activate_env.sh"
    exit 1
fi

echo "✅ HuggingFace cache directory: $HF_HOME"

# 모델 저장 디렉터리 생성 (현재 프로젝트 디렉터리에)
MODELS_DIR="./models"
mkdir -p $MODELS_DIR

echo "Creating models directory: $MODELS_DIR"

# HuggingFace CLI 경로 확인
if command -v huggingface-cli &> /dev/null; then
    echo "✅ huggingface-cli found at: $(which huggingface-cli)"
else
    echo "❌ huggingface-cli not found"
    echo "설치를 다시 확인해주세요: pip install huggingface-hub"
    exit 1
fi

# Llama Guard 3 (8B) 다운로드
echo ""
echo "📥 Downloading Llama Guard 3 (8B)..."
echo "This may take 10-20 minutes depending on your internet speed..."
echo "Models will be cached in: $HF_HOME"

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
echo "Total size: $(du -sh $MODELS_DIR 2>/dev/null | cut -f1 || echo 'calculating...')"
echo "Cache directory: $HF_HOME"
echo "Cache size: $(du -sh $HF_HOME 2>/dev/null | cut -f1 || echo 'calculating...')"

echo ""
echo "✅ Model download completed!"
echo ""
echo "Next steps:"
echo "1. Test environment: python test_environment.py"
echo "2. Run experiments: python train_experiments.py"