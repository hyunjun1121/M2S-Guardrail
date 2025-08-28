#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import re
import time

def get_gpu_memory_usage():
    """GPU 메모리 사용량 확인"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(', ')
                gpu_id = int(parts[0])
                used_mb = int(parts[1])
                total_mb = int(parts[2])
                free_mb = total_mb - used_mb
                usage_percent = (used_mb / total_mb) * 100
                
                gpu_info.append({
                    'id': gpu_id,
                    'used_mb': used_mb,
                    'total_mb': total_mb,
                    'free_mb': free_mb,
                    'usage_percent': usage_percent
                })
        
        return gpu_info
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")
        return []

def find_available_gpu(min_free_gb=6):
    """사용 가능한 GPU 찾기 (최소 free_gb 이상 여유 메모리)"""
    gpu_info = get_gpu_memory_usage()
    
    if not gpu_info:
        return None
    
    print("=== GPU 메모리 사용 현황 ===")
    for gpu in gpu_info:
        print(f"GPU {gpu['id']}: {gpu['used_mb']:,} MB / {gpu['total_mb']:,} MB 사용 ({gpu['usage_percent']:.1f}%) - 여유: {gpu['free_mb']:,} MB")
    
    # 최소 요구 메모리 이상의 여유가 있는 GPU 찾기
    min_free_mb = min_free_gb * 1024
    available_gpus = [gpu for gpu in gpu_info if gpu['free_mb'] >= min_free_mb]
    
    if available_gpus:
        # 가장 여유 메모리가 많은 GPU 선택
        best_gpu = max(available_gpus, key=lambda x: x['free_mb'])
        print(f"\n✅ GPU {best_gpu['id']} 선택 (여유 메모리: {best_gpu['free_mb']:,} MB)")
        return best_gpu['id']
    else:
        print(f"\n❌ {min_free_gb}GB 이상 여유 메모리를 가진 GPU가 없습니다.")
        return None

def run_experiment_on_available_gpu():
    """사용 가능한 GPU에서 실험 실행"""
    
    print("🔍 사용 가능한 GPU 검색 중...")
    
    # 6GB 이상 여유 메모리가 있는 GPU 찾기
    available_gpu = find_available_gpu(min_free_gb=6)
    
    if available_gpu is None:
        print("대기 중... 5분 후 다시 확인합니다.")
        time.sleep(300)  # 5분 대기
        available_gpu = find_available_gpu(min_free_gb=6)
        
        if available_gpu is None:
            print("❌ 사용 가능한 GPU가 없습니다. CPU로 실행하시겠습니까? (y/n)")
            response = input().lower()
            if response == 'y':
                available_gpu = ""  # CPU 모드
            else:
                print("실험을 취소합니다.")
                return
    
    # 환경 변수 설정
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    if available_gpu == "":
        print("🖥️  CPU에서 실험 시작...")
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        print(f"🚀 GPU {available_gpu}에서 실험 시작...")
        os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu)
    
    # 실험 실행
    experiments = [
        {
            "name": "integrated_data",
            "train": "integrated_data_train.xlsx",
            "val": "integrated_data_val.xlsx"
        },
        {
            "name": "hyphenize",
            "train": "m2s_hyphenize_data_train.xlsx", 
            "val": "m2s_hyphenize_data_val.xlsx"
        },
        {
            "name": "numberize",
            "train": "m2s_numberize_data_train.xlsx",
            "val": "m2s_numberize_data_val.xlsx"  
        },
        {
            "name": "pythonize",
            "train": "m2s_pythonize_data_train.xlsx",
            "val": "m2s_pythonize_data_val.xlsx"
        },
        {
            "name": "combined_all", 
            "train": "m2s_combined_all_data_train.xlsx",
            "val": "m2s_combined_all_data_val.xlsx"
        }
    ]
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n🧪 실험 {i}/5: {exp['name']} 시작...")
        
        cmd = [
            "python", "run_guardrail_with_validation_fixed.py",
            "--model", "Llama-Prompt-Guard-2-86M",
            "--train", exp['train'],
            "--val", exp['val'], 
            "--name", exp['name']
        ]
        
        try:
            print(f"실행 명령어: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            print(f"✅ 실험 {exp['name']} 완료!")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ 실험 {exp['name']} 실패: {e}")
            print("다음 실험으로 계속 진행...")
            continue
        except KeyboardInterrupt:
            print("\n⏹️  사용자에 의해 실험이 중단되었습니다.")
            break
        
        # 실험 간 5분 대기 (메모리 정리)
        if i < len(experiments):
            print("💤 다음 실험을 위해 5분 대기 중...")
            time.sleep(300)
    
    print("\n🎉 모든 실험 완료!")

if __name__ == "__main__":
    run_experiment_on_available_gpu()