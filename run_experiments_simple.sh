#!/bin/bash
# 간단한 실험 실행 스크립트 - conda 환경 문제 해결

set -e

echo "=== M2S-Guardrail Simple Experiments ==="

# Conda 환경 설정
export PATH="/eic/data/hkim_gpu2/miniconda3/bin:$PATH"
source /eic/data/hkim_gpu2/miniconda3/etc/profile.d/conda.sh
conda activate m2s-guardrail

# GPU 환경 설정
export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

echo "Environment check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

echo ""
echo "🚀 Running experiments with memory-efficient trainer..."

# 실험 목록 - 더 간단한 방식
experiments=("original" "hyphenize" "numberize" "pythonize" "combined")
successful=0
failed=0

for i in "${!experiments[@]}"; do
    exp_name="${experiments[$i]}"
    exp_id=$((i + 1))
    
    echo ""
    echo "[$exp_id/5] Starting experiment: $exp_name"
    echo "=========================================="
    
    # Python 스크립트를 직접 실행하지 말고, 환경이 이미 로드된 상태에서 진행
    python -c "
import os
import sys
sys.path.append('.')

# 환경 변수 설정
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,5,6,7'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from train_experiments_fixed import MemoryEfficientExperimentRunner
from pathlib import Path

# 실험 설정
experiments_config = {
    'original': {
        'id': 'exp_01',
        'name': 'GuardI8_original', 
        'model': 'meta-llama/Llama-Guard-3-8B-INT8',
        'data_type': 'original',
        'data_files': ['training_data/train_original.xlsx'],
        'output_dir': Path('experiments/exp_01_GuardI8_original')
    },
    'hyphenize': {
        'id': 'exp_02',
        'name': 'GuardI8_hyphenize',
        'model': 'meta-llama/Llama-Guard-3-8B-INT8', 
        'data_type': 'hyphenize',
        'data_files': ['training_data/train_hyphenize.xlsx'],
        'output_dir': Path('experiments/exp_02_GuardI8_hyphenize')
    },
    'numberize': {
        'id': 'exp_03',
        'name': 'GuardI8_numberize',
        'model': 'meta-llama/Llama-Guard-3-8B-INT8',
        'data_type': 'numberize', 
        'data_files': ['training_data/train_numberize.xlsx'],
        'output_dir': Path('experiments/exp_03_GuardI8_numberize')
    },
    'pythonize': {
        'id': 'exp_04',
        'name': 'GuardI8_pythonize',
        'model': 'meta-llama/Llama-Guard-3-8B-INT8',
        'data_type': 'pythonize',
        'data_files': ['training_data/train_pythonize.xlsx'], 
        'output_dir': Path('experiments/exp_04_GuardI8_pythonize')
    },
    'combined': {
        'id': 'exp_05',
        'name': 'GuardI8_combined',
        'model': 'meta-llama/Llama-Guard-3-8B-INT8',
        'data_type': 'combined',
        'data_files': ['training_data/train_combined.xlsx'],
        'output_dir': Path('experiments/exp_05_GuardI8_combined') 
    }
}

# 실험 실행
config = experiments_config['$exp_name']
print(f'Running experiment: {config[\"name\"]}')

runner = MemoryEfficientExperimentRunner()
success = runner.run_single_experiment(config, test_mode=True)

if success:
    print(f'✅ Experiment $exp_name completed successfully!')
    exit(0)
else:
    print(f'❌ Experiment $exp_name failed!')  
    exit(1)
"
    
    # 결과 확인
    if [ $? -eq 0 ]; then
        echo "✅ Experiment $exp_id ($exp_name) succeeded"
        successful=$((successful + 1))
    else
        echo "❌ Experiment $exp_id ($exp_name) failed"
        failed=$((failed + 1))
    fi
    
    # 메모리 정리
    echo "Cleaning up memory..."
    sleep 3
    
done

echo ""
echo "=============================================="
echo "🎯 Final Results: $successful successful, $failed failed"
echo "=============================================="