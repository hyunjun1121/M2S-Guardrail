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

def format_llama_guard_data(turns_list):
    """Llama-Guard 형식으로 대화 데이터 포맷팅"""
    conversations = []
    
    # 각 턴을 개별 대화로 처리
    for i, turn in enumerate(turns_list):
        if turn and str(turn).strip():
            # Llama-Guard 표준 형식
            formatted_conversation = f"""<s>[INST] {str(turn).strip()} [/INST] unsafe

Category: O1
Category explanation: The message contains potentially harmful content that could lead to dangerous or illegal activities.</s>"""
            conversations.append(formatted_conversation)
    
    # 누적 컨텍스트 버전도 추가 (multi-turn 학습용)
    if len(turns_list) > 1:
        cumulative_context = ""
        for i, turn in enumerate(turns_list):
            if turn and str(turn).strip():
                if cumulative_context:
                    cumulative_context += f"\n\nUser: {str(turn).strip()}"
                else:
                    cumulative_context = f"User: {str(turn).strip()}"
                
                # 누적된 컨텍스트로 안전성 판단
                formatted_cumulative = f"""<s>[INST] {cumulative_context} [/INST] unsafe

Category: O1  
Category explanation: This multi-turn conversation contains escalating harmful content designed to bypass safety measures.</s>"""
                conversations.append(formatted_cumulative)
    
    return conversations

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
            if model_type == "llama_guard":
                formatted = f"""<s>[INST] {str(row['prompt'])} [/INST] unsafe

Category: O1
Category explanation: This is a multi-turn jailbreak attempt using structured prompting to bypass safety measures.</s>"""
                all_training_data.append(formatted)
            else:  # prompt_guard
                all_training_data.append({
                    'text': str(row['prompt']),
                    'label': 1
                })
        
        # 원본 턴 데이터인 경우
        else:
            # turn_1부터 turn_12까지 추출
            for i in range(1, 13):
                turn_col = f'turn_{i}'
                if turn_col in row and pd.notna(row[turn_col]):
                    turns.append(str(row[turn_col]).strip())
            
            if turns:
                if model_type == "llama_guard":
                    formatted_conversations = format_llama_guard_data(turns)
                    all_training_data.extend(formatted_conversations)
                else:  # prompt_guard
                    formatted_prompts = format_prompt_guard_data(turns)
                    all_training_data.extend(formatted_prompts)
    
    print(f"총 훈련 샘플 수: {len(all_training_data)}")
    return all_training_data

def create_llama_guard_dataset(training_data, tokenizer, max_length=1024):
    """Llama-Guard용 데이터셋 생성"""
    def tokenize_function(examples):
        # 전체 텍스트를 토큰화 (input + output 포함)
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
        # 언어 모델링: 입력과 출력이 동일
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized
    
    dataset = Dataset.from_dict({"text": training_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def create_prompt_guard_dataset(training_data, tokenizer, max_length=512):
    """Prompt-Guard용 데이터셋 생성"""
    texts = [item['text'] for item in training_data]
    labels = [item['label'] for item in training_data]
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None
        )
    
    dataset = Dataset.from_dict({"text": texts, "labels": labels})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def run_guardrail_with_validation(model_name, train_file, val_file, experiment_name):
    """Train/Validation으로 분할된 데이터로 Guardrail Fine-tuning"""
    
    print(f"🚀 Guardrail Fine-tuning 시작 (with Validation)")
    print(f"모델: {model_name}")
    print(f"Train 데이터: {train_file}")
    print(f"Validation 데이터: {val_file}")
    print(f"실험: {experiment_name}")
    
    # 모델 경로 설정
    model_paths = {
        "Llama-Guard-4-12B": "./models/Llama-Guard-4-12B",
        "Llama-Prompt-Guard-2-86M": "./models/Llama-Prompt-Guard-2-86M"
    }
    
    model_path = model_paths[model_name]
    
    # 모델 타입 결정
    if "Prompt-Guard" in model_name:
        model_type = "prompt_guard"
        max_length = 512
    else:
        model_type = "llama_guard"  
        max_length = 1024
    
    print(f"모델 타입: {model_type}")
    
    # 토크나이저 로드
    print("토크나이저 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 로드
    print("모델 로드 중...")
    if model_type == "prompt_guard":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    else:  # llama_guard
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
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
    train_sample_size = min(5000, len(train_data))
    if len(train_data) > train_sample_size:
        import random
        train_data = random.sample(train_data, train_sample_size)
        print(f"Train 샘플링: {train_sample_size}개 사용")
    
    if val_data:
        val_sample_size = min(1000, len(val_data))
        if len(val_data) > val_sample_size:
            import random
            val_data = random.sample(val_data, val_sample_size)
            print(f"Validation 샘플링: {val_sample_size}개 사용")
    
    # 데이터셋 생성
    if model_type == "prompt_guard":
        train_dataset = create_prompt_guard_dataset(train_data, tokenizer, max_length)
        val_dataset = create_prompt_guard_dataset(val_data, tokenizer, max_length) if val_data else None
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    else:
        train_dataset = create_llama_guard_dataset(train_data, tokenizer, max_length)
        val_dataset = create_llama_guard_dataset(val_data, tokenizer, max_length) if val_data else None
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 출력 디렉터리
    output_dir = f"./fine_tuned_models/{model_name.replace('/', '_')}_{experiment_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 훈련 설정
    if "86M" in model_name:  # 작은 모델
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            eval_steps=500,
            save_strategy="steps",
            eval_strategy="steps" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=None,
        )
    else:  # 큰 모델 (12B)
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=2,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-5,
            weight_decay=0.01,
            logging_steps=25,
            save_steps=250,
            eval_steps=250,
            save_strategy="steps",
            eval_strategy="steps" if val_dataset else "no",
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            fp16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
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
    parser = argparse.ArgumentParser(description='Guardrail Model Fine-tuning with Train/Validation Split')
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
        result_dir = run_guardrail_with_validation(args.model, args.train, args.val, args.name)
        print(f"\n🎉 실험 성공! 결과: {result_dir}")
    except Exception as e:
        print(f"\n❌ 실험 실패: {e}")
        import traceback
        traceback.print_exc()