#!/bin/bash
# 메모리 안전한 방식으로 모든 실험 순차 실행

set -e

echo "=== M2S-Guardrail Sequential Experiments ==="

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Environment setup:"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"

# 결과 디렉터리 생성
mkdir -p experiments
mkdir -p logs

# 실험 목록
experiments=(
    "1:original"
    "2:hyphenize" 
    "3:numberize"
    "4:pythonize"
    "5:combined"
)

# 성공한 실험 추적
successful_experiments=()
failed_experiments=()

echo ""
echo "🚀 Starting 5 sequential experiments..."
echo ""

for exp_info in "${experiments[@]}"; do
    IFS=':' read -r exp_id exp_name <<< "$exp_info"
    
    echo "[$exp_id/5] Running experiment: $exp_name"
    echo "----------------------------------------"
    
    # 로그 파일
    log_file="logs/exp_${exp_id}_${exp_name}.log"
    
    # 실험 실행 (실패해도 계속 진행)
    if python run_single_experiment.py --exp_id $exp_id --test_mode 2>&1 | tee "$log_file"; then
        echo "✅ Experiment $exp_id ($exp_name) completed successfully"
        successful_experiments+=("$exp_id:$exp_name")
    else
        echo "❌ Experiment $exp_id ($exp_name) failed"
        failed_experiments+=("$exp_id:$exp_name")
    fi
    
    echo ""
    
    # Python 프로세스 완전 정리 (메모리 누적 방지)
    sleep 5
    pkill -f python || true
    sleep 3
    
    # GPU 메모리 상태 확인
    echo "GPU Memory Status:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
    echo ""
done

# 최종 결과 요약
echo "============================================================"
echo "🎯 Final Results Summary"
echo "============================================================"

echo "✅ Successful experiments (${#successful_experiments[@]}/5):"
for exp in "${successful_experiments[@]}"; do
    IFS=':' read -r id name <<< "$exp"
    echo "   - Experiment $id: $name"
done

echo ""
echo "❌ Failed experiments (${#failed_experiments[@]}/5):"
for exp in "${failed_experiments[@]}"; do
    IFS=':' read -r id name <<< "$exp"
    echo "   - Experiment $id: $name"
done

echo ""
echo "📁 Results saved to:"
echo "   - experiments/ (model checkpoints)"
echo "   - logs/ (training logs)"

# 평가 실행 (성공한 실험이 있는 경우)
if [ ${#successful_experiments[@]} -gt 0 ]; then
    echo ""
    echo "🔍 Starting model evaluation..."
    if python evaluate_models.py 2>&1 | tee logs/evaluation.log; then
        echo "✅ Evaluation completed successfully"
    else
        echo "❌ Evaluation failed"
    fi
fi

echo ""
echo "🎉 All experiments completed!"