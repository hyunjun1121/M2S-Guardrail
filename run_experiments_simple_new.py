#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import argparse

def run_simple_experiment(model_name, data_file, experiment_name):
    """간단한 fine-tuning 실험"""
    
    print(f"실험 시작: {experiment_name}")
    print(f"모델: {model_name}")
    print(f"데이터: {data_file}")
    
    # 모델 경로 설정
    model_paths = {
        "Llama-Guard-4-12B": "./models/Llama-Guard-4-12B",
        "Llama-Prompt-Guard-2-86M": "./models/Llama-Prompt-Guard-2-86M"
    }
    
    model_path = model_paths[model_name]
    
    # 토크나이저와 모델 로드
    print("모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 데이터 로드 및 전처리
    print("데이터 로드 중...")
    df = pd.read_excel(data_file)
    
    # 텍스트 데이터 준비
    texts = []
    if 'prompt' in df.columns:
        # M2S 데이터
        texts = df['prompt'].dropna().tolist()[:1000]  # 1000개 샘플
    else:
        # 원본 데이터
        for _, row in df.head(1000).iterrows():
            turns = []
            for i in range(1, 13):
                turn_col = f'turn_{i}'
                if turn_col in row and pd.notna(row[turn_col]):
                    turns.append(f"Turn {i}: {row[turn_col]}")
            if turns:
                texts.append("\n".join(turns))
    
    # 토큰화
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
    
    # 데이터셋 생성
    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    output_dir = f"./fine_tuned_models/{model_name.replace('/', '_')}_{experiment_name}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # 빠른 테스트를 위해 1 epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        save_steps=500,
        logging_steps=100,
        fp16=True,
        gradient_checkpointing=True,
        report_to=None,
    )
    
    # Trainer 설정
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Fine-tuning 실행
    print("Fine-tuning 시작...")
    trainer.train()
    
    # 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ 실험 완료: {experiment_name}")
    print(f"모델 저장 위치: {output_dir}")
    
    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['Llama-Guard-4-12B', 'Llama-Prompt-Guard-2-86M'])
    parser.add_argument('--data', required=True, help='Data file path')
    parser.add_argument('--name', required=True, help='Experiment name')
    
    args = parser.parse_args()
    
    run_simple_experiment(args.model, args.data, args.name)