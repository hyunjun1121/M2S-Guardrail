#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForLanguageModeling, DataCollatorWithPadding
)
from datasets import Dataset
import argparse
import json

def format_prompt_guard_data(turns_list):
    """Prompt-Guard 형식으로 데이터 포맷팅 (injection 탐지용)"""
    prompts_and_labels = []
    
    # 각 턴을 prompt injection으로 분류
    for turn in turns_list:
        if turn and str(turn).strip():
            prompts_and_labels.append({
                'text': str(turn).strip(),
                'label': 1  # 1 = injection/jailbreak
            })
    
    return prompts_and_labels

def process_harmful_dataset(data_file, model_type):
    """Harmful 데이터셋을 Guardrail 형식으로 변환"""
    print(f"데이터 로드 중: {data_file}")
    df = pd.read_excel(data_file)
    
    all_training_data = []
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(df)} rows...")
        
        turns = []
        
        # M2S 압축된 데이터인 경우
        if 'prompt' in row and pd.notna(row['prompt']):
            # M2S 전체 프롬프트를 harmful로 처리
            if model_type == "prompt_guard":
                all_training_data.append({
                    'text': str(row['prompt']).strip(),
                    'label': 1
                })
        
        # 원본 턴 데이터인 경우
        else:
            # turn_1부터 turn_12까지 추출
            for i in range(1, 13):
                turn_col = f'turn_{i}'
                if turn_col in row and pd.notna(row[turn_col]):
                    turns.append(str(row[turn_col]).strip())
            
            if turns and model_type == "prompt_guard":
                formatted_prompts = format_prompt_guard_data(turns)
                all_training_data.extend(formatted_prompts)
    
    print(f"총 훈련 샘플 수: {len(all_training_data)}")
    return all_training_data

def create_prompt_guard_dataset(training_data, tokenizer, max_length=512):
    """Prompt-Guard용 데이터셋 생성 (성공한 방식 사용)"""
    texts = [item['text'] for item in training_data]
    labels = [item['label'] for item in training_data]
    
    print(f"첫 5개 텍스트 샘플:")
    for i, text in enumerate(texts[:5]):
        print(f"  {i+1}. {text[:100]}... (label: {labels[i]})")
    
    def tokenize_function(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            padding=True,  # 성공한 설정
            max_length=max_length,
            return_tensors=None  # Dataset에서는 None 사용
        )
        return result
    
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    print(f"데이터셋 크기: {len(dataset)}")
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("토큰화 완료")
    
    return tokenized_dataset

def run_guardrail_final(model_name, train_file, val_file, experiment_name):
    """최종 Guardrail Fine-tuning (성공한 방식 기반)"""
    
    print(f"🚀 Guardrail Fine-tuning 시작 (최종 버전)")
    print(f"모델: {model_name}")
    print(f"Train 데이터: {train_file}")
    print(f"Validation 데이터: {val_file}")
    print(f"실험: {experiment_name}")
    
    # 현재는 Prompt-Guard만 지원
    if "Prompt-Guard" not in model_name:
        print("❌ 현재는 Llama-Prompt-Guard-2-86M만 지원됩니다.")
        return None
    
    # 모델 경로 설정
    model_path = "./models/Llama-Prompt-Guard-2-86M"
    model_type = "prompt_guard"
    max_length = 512
    
    print(f"모델 타입: {model_type}")
    
    # 토크나이저 로드
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드 (성공한 설정 사용)
    print("모델 로드 중...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 성공한 설정
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"모델 파라미터: {model.num_parameters():,}")
    
    # Train 데이터 처리
    print("Train 데이터셋 처리 중...")
    train_data = process_harmful_dataset(train_file, model_type)
    
    # Validation 데이터 처리
    val_data = None
    if val_file and os.path.exists(val_file):
        print("Validation 데이터셋 처리 중...")
        val_data = process_harmful_dataset(val_file, model_type)
    
    # 샘플링 (메모리 절약)
    train_sample_size = min(2000, len(train_data))  # 더 작게 설정
    if len(train_data) > train_sample_size:
        import random
        train_data = random.sample(train_data, train_sample_size)
        print(f"Train 샘플링: {train_sample_size}개 사용")
    
    if val_data:
        val_sample_size = min(500, len(val_data))  # 더 작게 설정
        if len(val_data) > val_sample_size:
            import random
            val_data = random.sample(val_data, val_sample_size)
            print(f"Validation 샘플링: {val_sample_size}개 사용")
    
    # 데이터셋 생성 (성공한 방식 사용)
    train_dataset = create_prompt_guard_dataset(train_data, tokenizer, max_length)
    val_dataset = create_prompt_guard_dataset(val_data, tokenizer, max_length) if val_data else None
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 출력 디렉터리
    output_dir = f"./fine_tuned_models/{model_name.replace('/', '_')}_{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 훈련 설정 (성공한 설정 사용)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=2,  # 짧게 설정
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_steps=200,
        eval_steps=200,
        save_strategy="steps",
        eval_strategy="steps" if val_dataset else "no",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        fp16=False,  # 성공한 설정
        bf16=False,  # 성공한 설정
        gradient_checkpointing=False,  # 간소화
        dataloader_pin_memory=False,
        remove_unused_columns=True,
        report_to=None,
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 훈련 실행
    print("🔥 Fine-tuning 시작...")
    trainer.train()
    
    # 모델 저장
    print("💾 모델 저장 중...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # 훈련 통계 저장
    stats = {
        "model_name": model_name,
        "model_type": model_type,
        "experiment_name": experiment_name,
        "train_file": train_file,
        "val_file": val_file,
        "train_samples": len(train_data),
        "val_samples": len(val_data) if val_data else 0,
        "output_dir": output_dir,
        "max_length": max_length,
        "model_parameters": model.num_parameters()
    }
    
    with open(os.path.join(output_dir, "training_stats.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Fine-tuning 완료!")
    print(f"📁 모델 저장 위치: {output_dir}")
    print(f"📊 Train 샘플: {len(train_data)}")
    print(f"📊 Validation 샘플: {len(val_data) if val_data else 0}")
    
    # 메모리 정리
    del trainer, model
    torch.cuda.empty_cache()
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Final Guardrail Model Fine-tuning')
    parser.add_argument('--model', required=True, 
                       choices=['Llama-Guard-4-12B', 'Llama-Prompt-Guard-2-86M'],
                       help='Guardrail model to fine-tune')
    parser.add_argument('--train', required=True, 
                       help='Path to training dataset (Excel file)')
    parser.add_argument('--val', required=False,
                       help='Path to validation dataset (Excel file)')
    parser.add_argument('--name', required=True,
                       help='Experiment name')
    
    args = parser.parse_args()
    
    try:
        result_dir = run_guardrail_final(args.model, args.train, args.val, args.name)
        print(f"\n🎉 실험 성공! 결과: {result_dir}")
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()