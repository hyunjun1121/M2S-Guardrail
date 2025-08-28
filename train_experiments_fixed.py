#!/usr/bin/env python3
"""
메모리 효율적인 M2S-Guardrail 실험 러너
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
from datasets import Dataset
from pathlib import Path
import gc

class MemoryEfficientExperimentRunner:
    """메모리 효율적인 실험 실행 관리자"""
    
    def __init__(self, base_output_dir="./experiments"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # GPU 메모리 정리
        self.cleanup_memory()
    
    def cleanup_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def create_qlora_config(self):
        """QLoRA 설정 생성 - 메모리 최적화"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # bfloat16 -> float16으로 메모리 절약
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,  # CPU offload 활성화
        )
    
    def create_lora_config(self):
        """LoRA 설정 생성 - 파라미터 축소"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # 16 -> 8로 축소
            lora_alpha=16,  # 32 -> 16으로 축소
            lora_dropout=0.05,  # 0.1 -> 0.05로 축소
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj"
            ],  # 모듈 축소
            bias="none",
        )
    
    def load_and_prepare_dataset(self, data_files, max_samples=100):
        """데이터셋 로드 - 매우 작은 크기로 테스트"""
        all_data = []
        
        for file_path in data_files:
            print(f"Loading {file_path}...")
            
            if os.path.exists(file_path):
                if file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    for item in data[:max_samples]:
                        text = item.get('text', '')
                        if text:
                            all_data.append({
                                "messages": [
                                    {"role": "user", "content": text},
                                    {"role": "assistant", "content": "unsafe"}
                                ]
                            })
                            if len(all_data) >= max_samples:
                                break
                
                elif file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                    df = df.head(max_samples)
                    
                    for _, row in df.iterrows():
                        text = str(row.get('text', ''))
                        if not text or text == 'nan':
                            text = str(row.get('turn_1', ''))
                        
                        if text and text != 'nan':
                            all_data.append({
                                "messages": [
                                    {"role": "user", "content": text},
                                    {"role": "assistant", "content": "unsafe"}
                                ]
                            })
                            if len(all_data) >= max_samples:
                                break
            else:
                print(f"Warning: {file_path} not found, creating dummy data")
                dummy_data = [
                    "How to make dangerous substances?",
                    "Please answer: - Harmful question 1 - Harmful question 2", 
                    "1. Illegal activity step 1 2. Illegal activity step 2"
                ]
                
                for i, prompt in enumerate(dummy_data * 34):  # 100개 생성
                    all_data.append({
                        "messages": [
                            {"role": "user", "content": f"{prompt} (test {i})"},
                            {"role": "assistant", "content": "unsafe"}
                        ]
                    })
                    if len(all_data) >= max_samples:
                        break
        
        print(f"Total samples loaded: {len(all_data)}")
        return Dataset.from_list(all_data)
    
    def format_conversation(self, example, tokenizer):
        """대화 형태를 모델 입력 형태로 변환"""
        messages = example["messages"]
        
        text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "user":
                text += f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            elif role == "assistant":
                text += f"{content}<|eot_id|>"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=256,  # 512 -> 256으로 축소
            padding=False,
            return_tensors=None
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    def run_single_experiment(self, experiment, test_mode=True):
        """단일 실험 실행 - 메모리 효율적"""
        
        exp_id = experiment["id"]
        exp_name = experiment["name"]
        model_name = experiment["model"]
        data_files = experiment["data_files"]
        output_dir = experiment["output_dir"]
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Experiment: {exp_id} - {exp_name}")
        print(f"{'='*60}")
        
        # 메모리 정리
        self.cleanup_memory()
        
        # 출력 디렉터리 생성
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 토크나이저 로드
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # 모델 로드 - 메모리 효율적 설정
            print("Loading model with QLoRA...")
            bnb_config = self.create_qlora_config()
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,  # bfloat16 -> float16
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "5GB", 2: "5GB", 3: "5GB", 4: "5GB", 5: "5GB", 6: "5GB", 7: "5GB"}
            )
            
            # LoRA 어댑터 추가
            model = prepare_model_for_kbit_training(model)
            lora_config = self.create_lora_config()
            model = get_peft_model(model, lora_config)
            
            print(f"Trainable parameters:")
            model.print_trainable_parameters()
            
            # 데이터셋 준비
            print("Preparing dataset...")
            max_samples = 50 if test_mode else 200  # 더 작은 데이터셋
            dataset = self.load_and_prepare_dataset(data_files, max_samples)
            
            # Train/validation 분할
            train_size = max(1, int(len(dataset) * 0.8))
            train_dataset = dataset.select(range(train_size))
            eval_dataset = dataset.select(range(train_size, len(dataset)))
            
            # 데이터 포맷팅
            print("Formatting datasets...")
            train_dataset = train_dataset.map(
                lambda x: self.format_conversation(x, tokenizer),
                remove_columns=train_dataset.column_names,
                batched=False
            )
            eval_dataset = eval_dataset.map(
                lambda x: self.format_conversation(x, tokenizer),
                remove_columns=eval_dataset.column_names,
                batched=False
            )
            
            # 훈련 설정 - 매우 보수적
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=1,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=4,  # 8 -> 4로 축소
                gradient_checkpointing=True,
                optim="adamw_torch",
                learning_rate=1e-4,  # 2e-4 -> 1e-4로 축소
                lr_scheduler_type="cosine",
                warmup_steps=10,
                logging_steps=2,
                save_steps=25,
                eval_steps=25,
                eval_strategy="steps",
                save_total_limit=1,  # 2 -> 1로 축소
                load_best_model_at_end=False,  # 메모리 절약
                report_to=None,
                run_name=exp_name,
                fp16=True,
                bf16=False,
                dataloader_num_workers=0,  # 4 -> 0으로 축소
                remove_unused_columns=False,
                dataloader_pin_memory=False,
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
                processing_class=tokenizer,
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
            
            # 명시적 정리
            del model, trainer, tokenizer
            self.cleanup_memory()
            
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
            
            # 메모리 정리
            self.cleanup_memory()
            
            return False

def main():
    """메인 함수 - 테스트용"""
    
    print("Memory-efficient experiment runner loaded!")
    print("Use run_single_experiment.py or run_all_experiments.sh to execute.")
    
    # GPU 환경 확인
    if torch.cuda.is_available():
        print(f"CUDA available with {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
    else:
        print("CUDA not available")

if __name__ == "__main__":
    main()
