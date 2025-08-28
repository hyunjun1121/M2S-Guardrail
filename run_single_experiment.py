#!/usr/bin/env python3
"""
메모리 효율적인 단일 실험 실행 스크립트
각 실험을 독립적으로 실행하여 메모리 누적 방지
"""

import os
import sys
import torch
import json
from pathlib import Path
import argparse

def setup_environment():
    """환경 변수 설정"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5,6,7"  # GPU 1 제외
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    print("Environment setup:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")

def run_single_experiment(exp_config, test_mode=True):
    """단일 실험 실행"""
    
    from train_experiments import ExperimentRunner
    
    print(f"\n{'='*60}")
    print(f"🚀 Running Single Experiment: {exp_config['name']}")
    print(f"{'='*60}")
    
    # ExperimentRunner 초기화
    runner = ExperimentRunner()
    
    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 단일 실험 실행
    success = runner.run_single_experiment(exp_config, test_mode)
    
    # 명시적 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return success

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, required=True, help="Experiment ID (1-5)")
    parser.add_argument("--test_mode", action="store_true", default=True, help="Run in test mode")
    args = parser.parse_args()
    
    setup_environment()
    
    # 실험 설정 정의 (INT8 모델로만)
    experiments = [
        {
            "id": "exp_01",
            "name": "GuardI8_original",
            "model": "meta-llama/Llama-Guard-3-8B-INT8",
            "data_type": "original",
            "data_files": ["training_data/train_original.xlsx"],
            "output_dir": Path("experiments/exp_01_GuardI8_original")
        },
        {
            "id": "exp_02", 
            "name": "GuardI8_hyphenize",
            "model": "meta-llama/Llama-Guard-3-8B-INT8",
            "data_type": "hyphenize",
            "data_files": ["training_data/train_hyphenize.xlsx"],
            "output_dir": Path("experiments/exp_02_GuardI8_hyphenize")
        },
        {
            "id": "exp_03",
            "name": "GuardI8_numberize", 
            "model": "meta-llama/Llama-Guard-3-8B-INT8",
            "data_type": "numberize",
            "data_files": ["training_data/train_numberize.xlsx"],
            "output_dir": Path("experiments/exp_03_GuardI8_numberize")
        },
        {
            "id": "exp_04",
            "name": "GuardI8_pythonize",
            "model": "meta-llama/Llama-Guard-3-8B-INT8", 
            "data_type": "pythonize",
            "data_files": ["training_data/train_pythonize.xlsx"],
            "output_dir": Path("experiments/exp_04_GuardI8_pythonize")
        },
        {
            "id": "exp_05",
            "name": "GuardI8_combined",
            "model": "meta-llama/Llama-Guard-3-8B-INT8",
            "data_type": "combined", 
            "data_files": ["training_data/train_combined.xlsx"],
            "output_dir": Path("experiments/exp_05_GuardI8_combined")
        }
    ]
    
    if args.exp_id < 1 or args.exp_id > len(experiments):
        print(f"❌ Invalid experiment ID. Must be 1-{len(experiments)}")
        sys.exit(1)
    
    exp_config = experiments[args.exp_id - 1]
    
    print(f"Running experiment {args.exp_id}/{len(experiments)}")
    print(f"Config: {exp_config}")
    
    success = run_single_experiment(exp_config, args.test_mode)
    
    if success:
        print(f"✅ Experiment {args.exp_id} completed successfully!")
    else:
        print(f"❌ Experiment {args.exp_id} failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()