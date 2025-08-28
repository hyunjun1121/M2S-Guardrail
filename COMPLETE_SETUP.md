# 🚀 Complete Setup Guide - Start to Finish

M2S-Guardrail 프로젝트를 처음부터 완전히 설정하는 단계별 가이드입니다.

## 📋 **전체 진행 순서**

```
1. 모델 다운로드 → 2. Conda 환경 → 3. 환경 테스트 → 4. 실험 실행
```

---

## 🔽 **1단계: 모델 다운로드**

### **HuggingFace CLI 설치 (필요시)**
```bash
pip install huggingface-hub
```

### **모델 다운로드 실행**
```bash
# 자동 다운로드 스크립트 실행
bash download_models.sh

# 또는 수동으로
hf download meta-llama/Llama-Guard-3-8B --local-dir ./models/Llama-Guard-3-8B
hf download meta-llama/Llama-Guard-4-12B --local-dir ./models/Llama-Guard-4-12B
```

### **예상 소요 시간 및 용량**
- **Llama Guard 3 (8B)**: ~15-20분, ~16GB
- **Llama Guard 4 (12B)**: ~25-30분, ~24GB
- **총 용량**: ~40GB

---

## 🐍 **2단계: Conda 환경 설정**

### **환경 생성 및 패키지 설치**
```bash
# 자동 설정 스크립트 실행
bash setup_conda_env.sh
```

### **환경 활성화**
```bash
conda activate m2s-guardrail
```

### **설치되는 주요 패키지**
```
- Python 3.10
- PyTorch 2.x (CUDA 12.1)
- transformers 4.36+
- datasets 2.14+  
- accelerate 0.24+
- peft 0.6+
- bitsandbytes 0.41+
- flash-attn (최적화용)
```

---

## 🧪 **3단계: 환경 테스트**

### **전체 환경 검증**
```bash
python test_environment.py
```

### **체크리스트**
- ✅ CUDA 환경 확인
- ✅ GPU 메모리 상태 확인
- ✅ 패키지 import 테스트
- ✅ 모델 로드 테스트

### **예상 출력 예시**
```
=== GPU Environment Test ===
✅ CUDA available: 12.1
✅ GPU count: 8
   GPU 0: NVIDIA RTX A5000 (23.0 GB)
   ...
✅ Total GPU memory: 184.0 GB
✅ GPU computation test passed
```

---

## 🚀 **4단계: 데이터 준비 (선택사항)**

### **신규 데이터 전처리 (필요시)**
```bash
# M2S 전처리 실행 (시간 소요: 30-60분)
python m2s_preprocess_new.py
```

### **기존 데이터 사용**
```bash
# 로컬에서 전송된 파일 사용
ls -la combined_all_*.xlsx
```

---

## 🎯 **5단계: 실험 실행**

### **테스트 모드 (권장 첫 실행)**
```bash
python train_experiments.py
# → Test mode 선택: Y
# → 10개 실험, 각 200 샘플, 1 epoch
# → 예상 시간: 2-4시간
```

### **전체 모드 (최종 실행)**
```bash  
python train_experiments.py
# → Test mode 선택: N
# → 10개 실험, 전체 데이터셋, 3 epochs
# → 예상 시간: 20-40시간
```

---

## 📊 **실험 매트릭스 (10개 실험)**

| ID | 모델 | 데이터 | 설명 |
|----|----|-----|-----|
| exp_01 | Guard-3-8B | hyphenize | 하이픈 리스트 형태 |
| exp_02 | Guard-3-8B | numberize | 번호 리스트 형태 |
| exp_03 | Guard-3-8B | pythonize | Python 코드 형태 |
| exp_04 | Guard-3-8B | **combined** | **3가지 통합** |
| exp_05 | Guard-3-8B | **original** | **원본 multi-turn** |
| exp_06 | Guard-4-12B | hyphenize | 하이픈 리스트 형태 |
| exp_07 | Guard-4-12B | numberize | 번호 리스트 형태 |
| exp_08 | Guard-4-12B | pythonize | Python 코드 형태 |
| exp_09 | Guard-4-12B | **combined** | **3가지 통합** |
| exp_10 | Guard-4-12B | **original** | **원본 multi-turn** |

---

## 🔧 **트러블슈팅**

### **모델 다운로드 실패**
```bash
# HuggingFace 토큰 설정 (필요시)
huggingface-cli login

# 수동 재시도
hf download meta-llama/Llama-Guard-3-8B --resume-download
```

### **CUDA 메모리 부족**
```bash
# 다른 프로세스 확인
nvidia-smi

# GPU 메모리 정리
python -c "import torch; torch.cuda.empty_cache()"
```

### **패키지 충돌**
```bash
# 환경 재생성
conda remove -n m2s-guardrail --all
bash setup_conda_env.sh
```

---

## 📁 **파일 구조 (완료 후)**

```
M2S-Guardrail/
├── models/                          # 다운로드된 모델들
│   ├── Llama-Guard-3-8B/
│   └── Llama-Guard-4-12B/
├── experiments/                     # 실험 결과
│   ├── exp_01_Guard38B_hyphenize/
│   ├── exp_02_Guard38B_numberize/
│   └── ...
├── combined_all_*.xlsx              # M2S 변환 데이터
├── train_experiments.py             # 실험 실행기
└── README.md                        # 프로젝트 문서
```

---

## ⏰ **전체 소요 시간 예상**

| 단계 | 테스트 모드 | 전체 모드 |
|------|-------------|-----------|
| 모델 다운로드 | 30-45분 | 30-45분 |
| 환경 설정 | 10-15분 | 10-15분 |
| 실험 실행 | 2-4시간 | 20-40시간 |
| **총합** | **3-5시간** | **21-41시간** |

---

## 🎯 **실행 요약 (Copy & Paste)**

```bash
# 전체 과정 한번에 실행
cd ~/M2S-Guardrail
git pull origin master

# 1. 모델 다운로드
bash download_models.sh

# 2. Conda 환경 설정
bash setup_conda_env.sh
conda activate m2s-guardrail

# 3. 환경 테스트
python test_environment.py

# 4. 실험 실행 (테스트 모드)
python train_experiments.py
```

**이 가이드를 따라 단계별로 진행하시면 됩니다! 🚀**