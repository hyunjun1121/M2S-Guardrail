#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, 
    AutoModelForCausalLM, TrainingArguments, Trainer
)
from datasets import Dataset
import argparse

def load_guard_model(model_path):
    """Guard 모델에 맞는 로더"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prompt Guard는 SequenceClassification 모델
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model_type = "classification"
    except:
        # Guard 4는 CausalLM 시도
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto", 
            trust_remote_code=True
        )
        model_type = "causal"
    
    return tokenizer, model, model_type

def run_guard_experiment(model_name, data_file, experiment_name):
    """Guard 모델용 실험"""
    
    print(f"실험 시작: {experiment_name}")
    print(f"모델: {model_name}")
    print(f"데이터: {data_file}")
    
    model_paths = {
        "Llama-Guard-4-12B": "./models/Llama-Guard-4-12B",
        "Llama-Prompt-Guard-2-86M": "./models/Llama-Prompt-Guard-2-86M"
    }
    
    model_path = model_paths[model_name]
    
    # 모델 로드
    print("모델 로드 중...")
    tokenizer, model, model_type = load_guard_model(model_path)
    
    print(f"모델 타입: {model_type}")
    print(f"모델 파라미터: {model.num_parameters():,}")
    
    # 데이터 로드
    print("데이터 로드 중...")
    df = pd.read_excel(data_file)
    
    # 텍스트 준비
    texts = []
    if 'prompt' in df.columns:
        texts = df['prompt'].dropna().tolist()[:100]  # 작은 샘플로 테스트
    else:
        for _, row in df.head(100).iterrows():
            turns = []
            for i in range(1, 13):
                turn_col = f'turn_{i}'
                if turn_col in row and pd.notna(row[turn_col]):
                    turns.append(str(row[turn_col]))
            if turns:
                texts.append(" ".join(turns))
    
    print(f"샘플 수: {len(texts)}")
    
    # 데이터셋 생성 (classification 모델용으로 단순화)
    if model_type == "classification":
        # 간단한 분류 태스크로 변환
        labels = [0] * len(texts)  # 모든 샘플을 safe로 라벨링
        
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = Dataset.from_dict({"text": texts, "labels": labels})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
    else:
        # Causal LM의 경우
        def tokenize_function(examples):
            tokenized = tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors="pt"
            )
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments (매우 가벼운 설정)
    output_dir = f"./fine_tuned_models/{model_name.replace('/', '_')}_{experiment_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        save_steps=50,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None,
        max_steps=10  # 매우 짧은 테스트
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # 훈련 실행
    print("훈련 시작...")
    trainer.train()
    
    # 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 실험 완료: {experiment_name}")
    print(f"모델 저장: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, 
                       choices=['Llama-Guard-4-12B', 'Llama-Prompt-Guard-2-86M'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--name', required=True)
    
    args = parser.parse_args()
    
    run_guard_experiment(args.model, args.data, args.name)