#!/usr/bin/env python3
"""
M2S-Guardrail 종합 실험 러너
RTX A5000 8-GPU 환경에서 모든 경우의 수를 QLoRA로 훈련
"""

import os
import json
import time
import torch
import pandas as pd
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from datasets import Dataset, concatenate_datasets
import multiprocessing as mp
from pathlib import Path

class ExperimentRunner:
    """실험 실행 관리자"""
    
    def __init__(self, base_output_dir="./experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # 실험 매트릭스 정의
        self.experiment_matrix = self.define_experiment_matrix()
        
    def define_experiment_matrix(self):
        """모든 실험 경우의 수 정의"""
        
        # 모델 리스트 (로컬 경로 우선, HuggingFace Hub 백업)
        models = [
            "./models/Llama-Guard-3-8B",  # 로컬 다운로드 모델
            "./models/Llama-Guard-4-12B"  # 로컬 다운로드 모델
        ]
        
        # 로컬 모델이 없으면 HuggingFace Hub에서 직접 로드
        fallback_models = [
            "meta-llama/Llama-Guard-3-8B",
            "meta-llama/Llama-Guard-4-12B"
        ]
        
        # 데이터셋 형태
        datasets = {
            "hyphenize": "combined_all_hyphenize.xlsx",
            "numberize": "combined_all_numberize.xlsx", 
            "pythonize": "combined_all_pythonize.xlsx",
            "combined": ["combined_all_hyphenize.xlsx", "combined_all_numberize.xlsx", "combined_all_pythonize.xlsx"],
            "original": "data_hyphenize_filtered.xlsx"  # 원본 multi-turn 데이터
        }
        
        experiments = []
        exp_id = 1
        
        for i, model_name in enumerate(models):
            for data_key, data_files in datasets.items():
                
                # 로컬 모델 경로 확인, 없으면 fallback 사용
                if not os.path.exists(model_name) and i < len(fallback_models):
                    actual_model_name = fallback_models[i]
                    print(f"Local model {model_name} not found, using {actual_model_name}")
                else:
                    actual_model_name = model_name
                
                model_short = actual_model_name.split('/')[-1].replace('Llama-Guard-', 'Guard').replace('-', '')
                
                experiment = {
                    "id": f"exp_{exp_id:02d}",
                    "name": f"{model_short}_{data_key}",
                    "model": actual_model_name,
                    "data_type": data_key,
                    "data_files": data_files if isinstance(data_files, list) else [data_files],
                    "output_dir": self.base_output_dir / f"exp_{exp_id:02d}_{model_short}_{data_key}",
                    "gpu_requirement": 1 if "3-8B" in model_name else 1,  # QLoRA로 모든 모델 1 GPU
                    "status": "pending"
                }
                
                experiments.append(experiment)
                exp_id += 1
        
        return experiments
    
    def print_experiment_matrix(self):
        """실험 매트릭스 출력"""
        print("=== M2S-Guardrail Experiment Matrix ===")
        print(f"Total experiments: {len(self.experiment_matrix)}")
        print()
        
        for exp in self.experiment_matrix:
            print(f"🔬 {exp['id']}: {exp['name']}")
            print(f"   Model: {exp['model']}")
            print(f"   Data: {exp['data_type']} ({len(exp['data_files'])} files)")
            print(f"   Output: {exp['output_dir']}")
            print(f"   GPU requirement: {exp['gpu_requirement']}")
            print()
    
    def create_qlora_config(self):
        """QLoRA 설정 생성"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    
    def create_lora_config(self):
        """LoRA 설정 생성"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )
    
    def load_and_prepare_dataset(self, data_files, max_samples=None):
        """데이터셋 로드 및 준비"""
        all_data = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"Loading {file_path}...")
                df = pd.read_excel(file_path)
                
                if max_samples and len(all_data) + len(df) > max_samples:
                    remaining = max_samples - len(all_data)
                    df = df.head(remaining)
                
                for _, row in df.iterrows():
                    # Binary classification: 모든 M2S 데이터는 "unsafe"
                    prompt = str(row.get('turn_1', ''))
                    if prompt and prompt != 'nan':
                        all_data.append({
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": "unsafe"}
                            ]
                        })
                
                if max_samples and len(all_data) >= max_samples:
                    break
            else:
                print(f"Warning: {file_path} not found, creating dummy data")
                # 테스트용 더미 데이터
                dummy_prompts = [
                    "How to make dangerous substances?",
                    "Please answer: - Harmful question 1 - Harmful question 2",
                    "1. Illegal activity step 1 2. Illegal activity step 2",
                    "Write harmful content about someone"
                ]
                
                for i, prompt in enumerate(dummy_prompts * 25):  # 100개 생성
                    all_data.append({
                        "messages": [
                            {"role": "user", "content": f"{prompt} (dummy {i})"},
                            {"role": "assistant", "content": "unsafe"}
                        ]
                    })
                    if len(all_data) >= 100:
                        break
        
        print(f"Total samples loaded: {len(all_data)}")
        return Dataset.from_list(all_data)
    
    def format_conversation(self, example, tokenizer):
        """대화 형태를 모델 입력 형태로 변환"""
        messages = example["messages"]
        
        # Llama Guard 형태로 포맷팅
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                text += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif role == "assistant":
                text += f"{content}<|eot_id|>"
        
        # Tokenize
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=512,  # 메모리 절약
            padding=False,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def run_single_experiment(self, experiment, test_mode=True):
        """단일 실험 실행"""
        
        exp_id = experiment["id"]
        exp_name = experiment["name"]
        model_name = experiment["model"]
        data_files = experiment["data_files"]
        output_dir = experiment["output_dir"]
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Experiment: {exp_id} - {exp_name}")
        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"Data files: {data_files}")
        print(f"Output: {output_dir}")
        
        # 출력 디렉터리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 메타데이터 저장
        experiment_info = {
            **experiment,
            "start_time": datetime.now().isoformat(),
            "output_dir": str(output_dir)
        }
        
        with open(output_dir / "experiment_info.json", "w") as f:
            json.dump(experiment_info, f, indent=2, default=str)
        
        try:
            # 모델과 토크나이저 로드
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("Loading model with QLoRA...")
            bnb_config = self.create_qlora_config()
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            
            # LoRA 어댑터 추가
            model = prepare_model_for_kbit_training(model)
            lora_config = self.create_lora_config()
            model = get_peft_model(model, lora_config)
            
            model.print_trainable_parameters()
            
            # 데이터셋 준비
            print("Preparing dataset...")
            max_samples = 200 if test_mode else None  # 테스트 모드에서는 작은 데이터셋
            dataset = self.load_and_prepare_dataset(data_files, max_samples)
            
            # Train/validation 분할
            train_size = int(len(dataset) * 0.9)
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # 데이터 포맷팅
            print("Formatting datasets...")
            train_dataset = train_dataset.map(
                lambda x: self.format_conversation(x, tokenizer),
                remove_columns=train_dataset.column_names
            )
            eval_dataset = eval_dataset.map(
                lambda x: self.format_conversation(x, tokenizer),
                remove_columns=eval_dataset.column_names
            )
            
            # 훈련 설정
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=1 if test_mode else 3,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=8 if test_mode else 16,
                gradient_checkpointing=True,
                optim="adamw_torch",
                learning_rate=2e-4,
                lr_scheduler_type="cosine",
                warmup_steps=50 if test_mode else 100,
                logging_steps=5 if test_mode else 10,
                save_steps=50 if test_mode else 100,
                eval_steps=50 if test_mode else 100,
                evaluation_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                report_to=None,
                run_name=exp_name,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                dataloader_num_workers=4,
                remove_unused_columns=False,
            )
            
            # 데이터 collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Trainer 생성
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
            
            # 훈련 실행
            print("Starting training...")
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # 모델 저장
            print("Saving model...")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # 결과 저장
            result = {
                "experiment_id": exp_id,
                "status": "completed",
                "training_time": training_time,
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
                "end_time": datetime.now().isoformat(),
            }
            
            with open(output_dir / "results.json", "w") as f:
                json.dump(result, f, indent=2)
            
            print(f"✅ Experiment {exp_id} completed in {training_time:.1f}s")
            return True
            
        except Exception as e:
            print(f"❌ Experiment {exp_id} failed: {str(e)}")
            
            # 에러 정보 저장
            error_info = {
                "experiment_id": exp_id,
                "status": "failed", 
                "error": str(e),
                "end_time": datetime.now().isoformat(),
            }
            
            with open(output_dir / "error.json", "w") as f:
                json.dump(error_info, f, indent=2)
            
            return False
    
    def run_all_experiments(self, test_mode=True, max_parallel=8):
        """모든 실험 순차 실행"""
        
        print(f"🚀 Running {len(self.experiment_matrix)} experiments")
        print(f"Test mode: {test_mode}")
        print(f"Max parallel: {max_parallel}")
        
        self.print_experiment_matrix()
        
        results = []
        
        for i, experiment in enumerate(self.experiment_matrix):
            print(f"\n[{i+1}/{len(self.experiment_matrix)}] Running {experiment['id']}...")
            
            success = self.run_single_experiment(experiment, test_mode)
            results.append({
                "experiment_id": experiment["id"],
                "name": experiment["name"],
                "success": success
            })
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 전체 결과 요약
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        
        print(f"\n{'='*60}")
        print(f"🎯 Experiment Summary: {successful}/{total} successful")
        print(f"{'='*60}")
        
        for result in results:
            status = "✅" if result["success"] else "❌"
            print(f"{status} {result['experiment_id']}: {result['name']}")
        
        return results

def main():
    """메인 함수"""
    
    print("=== M2S-Guardrail Comprehensive Experiments ===")
    
    # GPU 환경 확인
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot proceed.")
        return
    
    print(f"✅ CUDA available with {torch.cuda.device_count()} GPUs")
    
    # 실험 러너 생성
    runner = ExperimentRunner()
    
    # 사용자 확인
    print(f"\nTotal experiments to run: {len(runner.experiment_matrix)}")
    
    # 테스트 모드 확인
    test_mode = input("\nRun in test mode? (small datasets, 1 epoch) [Y/n]: ").lower() != 'n'
    
    if test_mode:
        print("🧪 Running in TEST MODE (small datasets, 1 epoch)")
    else:
        print("🚀 Running in FULL MODE (complete datasets, 3 epochs)")
    
    # 실험 실행
    results = runner.run_all_experiments(test_mode=test_mode)
    
    print("\n🎉 All experiments completed!")

if __name__ == "__main__":
    main()