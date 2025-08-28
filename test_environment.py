#!/usr/bin/env python3
"""
M2S-Guardrail 환경 테스트 스크립트
모든 필수 패키지와 GPU 환경이 올바르게 설정되었는지 확인
"""

import sys
import torch
import subprocess
import importlib.util

def test_package_import(package_name, import_name=None):
    """패키지 import 테스트"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False

def test_gpu_environment():
    """GPU 환경 테스트"""
    print("\n=== GPU Environment Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ GPU count: {torch.cuda.device_count()}")
    
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        total_memory += memory_gb
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({memory_gb:.1f} GB)")
    
    print(f"✅ Total GPU memory: {total_memory:.1f} GB")
    
    # 간단한 GPU 연산 테스트
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation test passed")
        return True
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        return False

def test_memory_estimation():
    """메모리 사용량 추정 테스트"""
    print("\n=== Memory Estimation Test ===")
    
    try:
        from transformers import AutoTokenizer
        
        # 간단한 tokenizer 테스트
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        test_text = "Hello, this is a test message."
        tokens = tokenizer(test_text, return_tensors="pt")
        
        print(f"✅ Tokenizer test passed")
        print(f"   Test text: {test_text}")
        print(f"   Token count: {len(tokens['input_ids'][0])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory estimation test failed: {e}")
        return False

def check_available_gpu_memory():
    """사용 가능한 GPU 메모리 확인"""
    print("\n=== Available GPU Memory ===")
    
    if not torch.cuda.is_available():
        return False
    
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        
        # 현재 할당된 메모리
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        # 예약된 메모리  
        reserved = torch.cuda.memory_reserved(i) / (1024**3)
        # 전체 메모리
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        # 사용 가능한 메모리
        available = total - reserved
        
        print(f"GPU {i}:")
        print(f"   Total: {total:.1f} GB")
        print(f"   Allocated: {allocated:.1f} GB")
        print(f"   Reserved: {reserved:.1f} GB") 
        print(f"   Available: {available:.1f} GB")
        
        # QLoRA 실행 가능 여부 판단
        if available >= 4.0:
            print(f"   ✅ Suitable for QLoRA training")
        elif available >= 2.0:
            print(f"   ⚠️  Limited for QLoRA (need optimization)")
        else:
            print(f"   ❌ Insufficient for QLoRA training")
    
    return True

def main():
    """메인 테스트 함수"""
    print("=== M2S-Guardrail Environment Test ===")
    print(f"Python version: {sys.version}")
    print()
    
    # 패키지 import 테스트
    print("=== Package Import Test ===")
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"), 
        ("datasets", "datasets"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("sklearn", "sklearn"),
    ]
    
    failed_packages = []
    for package_name, import_name in packages:
        if not test_package_import(package_name, import_name):
            failed_packages.append(package_name)
    
    if failed_packages:
        print(f"\n❌ Failed packages: {', '.join(failed_packages)}")
        print("Please install missing packages and try again.")
        return False
    
    # GPU 환경 테스트
    if not test_gpu_environment():
        print("\n❌ GPU environment test failed")
        return False
    
    # 메모리 테스트
    if not test_memory_estimation():
        print("\n❌ Memory estimation test failed") 
        return False
    
    # 사용 가능한 GPU 메모리 확인
    check_available_gpu_memory()
    
    print("\n=== Summary ===")
    print("✅ All tests passed!")
    print("Environment is ready for M2S-Guardrail training.")
    print("\nNext steps:")
    print("1. Prepare training data: python m2s_preprocess_new.py")
    print("2. Start QLoRA training: python train_experiments.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)