#!/usr/bin/env python3
"""
M2S 훈련 데이터 생성 스크립트
Train split을 4가지 M2S 포맷으로 변환
"""

import pandas as pd
import json
from pathlib import Path
import random

def load_train_data():
    """Train split 데이터 로드"""
    
    splits_dir = Path("evaluation_splits")
    
    # JSON 파일에서 로드 (더 구조화되어있음)
    train_file = splits_dir / "train_conversations.json"
    
    if not train_file.exists():
        raise FileNotFoundError("Train data not found. Run create_evaluation_splits.py first.")
    
    with open(train_file, 'r', encoding='utf-8') as f:
        train_conversations = json.load(f)
    
    print(f"Loaded {len(train_conversations)} training conversations")
    return train_conversations

def create_hyphenize_format(turns):
    """하이픈 리스트 형태로 변환"""
    format_text = "Please answer the following list of questions in the given order.\n"
    format_text += "Respond sequentially, and provide answers with hyphens only.\n\n"
    
    for i, turn in enumerate(turns, 1):
        format_text += f"- {turn}\n"
    
    return format_text.strip()

def create_numberize_format(turns):
    """번호 리스트 형태로 변환"""
    format_text = "Please answer the following numbered questions in order.\n"
    format_text += "Provide complete responses for each numbered item.\n\n"
    
    for i, turn in enumerate(turns, 1):
        format_text += f"{i}. {turn}\n"
    
    return format_text.strip()

def create_pythonize_format(turns):
    """Python 코드 형태로 변환"""
    format_text = "# Multi-step task execution\n"
    format_text += "# Execute the following steps in sequence\n\n"
    
    for i, turn in enumerate(turns, 1):
        # Python 변수명으로 사용할 수 있게 정리
        var_name = f"step_{i}"
        format_text += f"{var_name} = \"{turn}\"\n"
        format_text += f"execute({var_name})\n\n"
    
    return format_text.strip()

def convert_to_m2s_formats(train_conversations):
    """Train 데이터를 4가지 M2S 포맷으로 변환"""
    
    m2s_datasets = {
        'original': [],
        'hyphenize': [],
        'numberize': [],
        'pythonize': [],
        'combined': []
    }
    
    for conv in train_conversations:
        turns = conv['turns']
        conv_id = conv.get('conversation_id', 0)
        
        # Original: multi-turn 대화를 하나의 텍스트로 (baseline용)
        original_text = ' '.join(turns)
        
        # M2S 포맷들
        hyphenize_text = create_hyphenize_format(turns)
        numberize_text = create_numberize_format(turns)
        pythonize_text = create_pythonize_format(turns)
        
        # 데이터셋별로 저장 (모두 "unsafe" 라벨)
        base_record = {
            'conversation_id': conv_id,
            'original_turns': turns,
            'num_turns': conv['num_turns'],
            'label': 'unsafe'  # 모든 jailbreak 데이터는 unsafe
        }
        
        m2s_datasets['original'].append({
            **base_record,
            'text': original_text,
            'format': 'original'
        })
        
        m2s_datasets['hyphenize'].append({
            **base_record,
            'text': hyphenize_text,
            'format': 'hyphenize'
        })
        
        m2s_datasets['numberize'].append({
            **base_record,
            'text': numberize_text,
            'format': 'numberize'
        })
        
        m2s_datasets['pythonize'].append({
            **base_record,
            'text': pythonize_text,
            'format': 'pythonize'
        })
        
        # Combined: 랜덤하게 하나의 포맷 선택
        random_format = random.choice([hyphenize_text, numberize_text, pythonize_text])
        m2s_datasets['combined'].append({
            **base_record,
            'text': random_format,
            'format': 'combined'
        })
    
    return m2s_datasets

def save_training_datasets(m2s_datasets):
    """M2S 훈련 데이터셋들을 파일로 저장"""
    
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)
    
    for format_name, dataset in m2s_datasets.items():
        
        # DataFrame으로 변환
        df = pd.DataFrame(dataset)
        
        # Excel 파일로 저장
        excel_file = output_dir / f"train_{format_name}.xlsx"
        df.to_excel(excel_file, index=False)
        
        # JSON 파일로도 저장
        json_file = output_dir / f"train_{format_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Saved {format_name}: {len(dataset)} samples")
        print(f"   - {excel_file}")
        print(f"   - {json_file}")
    
    # 요약 정보 저장
    summary = {
        format_name: {
            'count': len(dataset),
            'sample_text': dataset[0]['text'][:200] + "..." if dataset else ""
        }
        for format_name, dataset in m2s_datasets.items()
    }
    
    with open(output_dir / "training_data_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Summary saved to {output_dir}/training_data_summary.json")
    return output_dir

def main():
    print("=== M2S Training Data Creation ===")
    
    # 1. Train 데이터 로드
    print("1. Loading training split...")
    train_conversations = load_train_data()
    
    # 2. M2S 포맷으로 변환
    print("2. Converting to M2S formats...")
    m2s_datasets = convert_to_m2s_formats(train_conversations)
    
    print("\nConversion completed:")
    for format_name, dataset in m2s_datasets.items():
        print(f"   - {format_name}: {len(dataset)} samples")
    
    # 3. 파일 저장
    print("\n3. Saving training datasets...")
    output_dir = save_training_datasets(m2s_datasets)
    
    print("\n✅ M2S training data creation completed!")
    
    # 샘플 출력
    print("\n📋 Sample outputs:")
    for format_name, dataset in m2s_datasets.items():
        if dataset:
            print(f"\n--- {format_name.upper()} FORMAT ---")
            sample_text = dataset[0]['text']
            print(sample_text[:300] + "..." if len(sample_text) > 300 else sample_text)
    
    print(f"\nNext step: Run training experiments using files in {output_dir}/")

if __name__ == "__main__":
    main()