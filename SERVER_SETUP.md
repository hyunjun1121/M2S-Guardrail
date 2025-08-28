# 🚀 Server Setup Guide

서버에서 M2S-Guardrail 프로젝트를 실행하기 위한 단계별 가이드입니다.

## 📋 **1단계: Repository Clone**

```bash
# 프로젝트 클론
cd ~
git clone https://github.com/hyunjun1121/M2S-Guardrail.git
cd M2S-Guardrail

# 최신 버전 확인
git log --oneline -3
```

## 🔍 **2단계: GPU 환경 분석**

```bash
# 현재 GPU 상태 확인
nvidia-smi

# 실행 중인 프로세스 분석
python3 check_gpu_process.py

# 간단 프로세스 확인
ps -ef | grep 697115
cat /proc/697115/cmdline | tr '\0' ' '
```

## 📦 **3단계: 환경 설정**

```bash
# Python 환경 확인
python3 --version
pip3 --version

# 필수 패키지 설치
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip3 install transformers>=4.36.0
pip3 install datasets>=2.14.0
pip3 install accelerate>=0.24.0
pip3 install peft>=0.6.0
pip3 install bitsandbytes>=0.41.0
pip3 install pandas openpyxl
pip3 install psutil

# CUDA 환경 테스트
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 📊 **4단계: 데이터 준비**

### **방법 A: HuggingFace에서 직접 다운로드**
```bash
# HuggingFace CLI 설치
pip3 install huggingface_hub

# 데이터셋 다운로드
huggingface-cli download marslabucla/XGuard-Train --repo-type=dataset --local-dir ./XGuard-Train
huggingface-cli download Anthropic/hh-rlhf --repo-type=dataset --local-dir ./hh-rlhf

# M2S 전처리 실행
python3 m2s_preprocess_new.py
```

### **방법 B: 로컬에서 전처리된 데이터 전송 (권장)**
```bash
# 로컬에서 서버로 전송 (Windows → Linux)
scp combined_all_*.xlsx hkim@eic-gt-gpu2:~/M2S-Guardrail/
```

## 🎯 **5단계: GPU 리소스 관리**

### **시나리오 A: 전체 GPU 사용 가능한 경우**
```bash
# 현재 프로세스 종료 (안전 확인 후)
# kill 697115

# GPU 메모리 정리 확인
nvidia-smi

# Full SFT 실행 가능
```

### **시나리오 B: 현재 상태 유지**
```bash
# QLoRA 모드로 실행
# 각 GPU당 ~6GB 사용 가능
```

## 🚀 **6단계: 훈련 시작**

### **테스트 실행**
```bash
# 간단한 환경 테스트
python3 -c "
import torch
from transformers import AutoTokenizer
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
"
```

### **실제 훈련 (다음 단계에서 구현)**
```bash
# 단계별 훈련 스크립트 실행 예정
# python3 train_llama_guard_binary.py --config guard3_hyphenize
```

## 🔧 **트러블슈팅**

### **CUDA 관련 이슈**
```bash
# CUDA 버전 확인
nvcc --version
nvidia-smi

# PyTorch CUDA 호환성 확인
python3 -c "import torch; print(torch.version.cuda)"
```

### **메모리 부족 시**
```bash
# GPU 메모리 정리
sudo fuser -v /dev/nvidia*
# 또는
sudo systemctl restart nvidia-persistenced
```

### **권한 이슈**
```bash
# GPU 접근 권한 확인
ls -la /dev/nvidia*
groups $USER
```

## 📝 **다음 단계**

1. ✅ Repository clone 완료
2. ✅ GPU 환경 분석 완료  
3. ⏳ 환경 설정 및 데이터 준비
4. ⏳ 훈련 스크립트 구현
5. ⏳ 실험 실행

---

**현재 진행 상황을 체크하며 단계별로 진행해주세요!**