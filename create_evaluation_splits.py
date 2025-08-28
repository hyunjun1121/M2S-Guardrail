#!/usr/bin/env python3
"""
M2S-Guardrail 평가용 데이터 분할 스크립트
5개 소스 데이터를 균등하게 train/validation으로 분할
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
import os

def load_source_data():
    """5개 소스 데이터 로드"""
    
    # 원본 데이터 파일 경로들
    data_sources = {
        "MHJ_local": None,  # 파일 경로 확인 필요
        "SafeMTData_1K": None,
        "SafeMTData_Attack600": None,
        "XGuard_Train": None,
        "hh_rlhf": None
    }
    
    # data.xlsx에서 소스별로 분리 (실제 파일 구조에 따라 조정 필요)
    if os.path.exists("data.xlsx"):
        print("Loading original data.xlsx...")
        df_original = pd.read_excel("data.xlsx")
        
        # Source 컬럼이 있다면 활용
        if 'source' in df_original.columns:
            sources = df_original['source'].unique()
            print(f"Found sources: {sources}")
            
            for source in sources:
                source_data = df_original[df_original['source'] == source]
                if 'MHJ' in source or 'local' in source:
                    data_sources["MHJ_local"] = source_data
                elif '1K' in source:
                    data_sources["SafeMTData_1K"] = source_data
                elif 'Attack600' in source or '600' in source:
                    data_sources["SafeMTData_Attack600"] = source_data
        else:
            # source 컬럼이 없다면 전체를 하나로 처리
            print("No source column found, treating as single dataset")
            data_sources["SafeMTData_Original"] = df_original
    
    # 추가 데이터 파일들 확인
    additional_files = [
        ("XGuard_Train", "xguard_train.xlsx"),
        ("hh_rlhf", "hh_rlhf.xlsx")
    ]
    
    for source_name, filename in additional_files:
        if os.path.exists(filename):
            print(f"Loading {filename}...")
            data_sources[source_name] = pd.read_excel(filename)
    
    return data_sources

def extract_multi_turn_conversations(df):
    """DataFrame에서 multi-turn 대화 추출"""
    conversations = []
    
    for _, row in df.iterrows():
        # turn_1부터 turn_12까지 확인
        turns = []
        for i in range(1, 13):
            turn_col = f'turn_{i}'
            if turn_col in row and pd.notna(row[turn_col]) and str(row[turn_col]).strip():
                turns.append(str(row[turn_col]).strip())
        
        if len(turns) >= 2:  # 최소 2턴 이상인 대화만 선택
            conversations.append({
                'turns': turns,
                'num_turns': len(turns),
                'original_index': row.name,
                'conversation_text': ' [TURN] '.join(turns)
            })
    
    return conversations

def stratified_split_by_source(data_sources, train_ratio=0.8, random_state=42):
    """소스별로 균등하게 train/validation 분할"""
    
    train_data = []
    validation_data = []
    split_info = {}
    
    for source_name, df in data_sources.items():
        if df is None or df.empty:
            print(f"Warning: {source_name} is empty, skipping...")
            continue
        
        print(f"\nProcessing {source_name}: {len(df)} samples")
        
        # Multi-turn 대화 추출
        conversations = extract_multi_turn_conversations(df)
        
        if not conversations:
            print(f"Warning: No multi-turn conversations found in {source_name}")
            continue
        
        print(f"Extracted {len(conversations)} multi-turn conversations")
        
        # Train/validation 분할 (turn length별로 stratify)
        turn_lengths = [conv['num_turns'] for conv in conversations]
        
        try:
            train_convs, val_convs = train_test_split(
                conversations,
                train_size=train_ratio,
                stratify=turn_lengths,
                random_state=random_state
            )
        except ValueError:
            # Stratify가 불가능한 경우 (클래스가 적을 때)
            train_convs, val_convs = train_test_split(
                conversations,
                train_size=train_ratio,
                random_state=random_state
            )
        
        train_data.extend(train_convs)
        validation_data.extend(val_convs)
        
        split_info[source_name] = {
            'total': len(conversations),
            'train': len(train_convs),
            'validation': len(val_convs),
            'train_ratio': len(train_convs) / len(conversations)
        }
    
    return train_data, validation_data, split_info

def save_splits(train_data, validation_data, split_info):
    """분할된 데이터를 파일로 저장"""
    
    # 출력 디렉터리 생성
    output_dir = Path("evaluation_splits")
    output_dir.mkdir(exist_ok=True)
    
    # Train 데이터 저장
    train_df = pd.DataFrame([
        {
            'conversation_id': i,
            'turns': conv['turns'],
            'num_turns': conv['num_turns'],
            'conversation_text': conv['conversation_text'],
            'original_index': conv['original_index']
        }
        for i, conv in enumerate(train_data)
    ])
    
    # Validation 데이터 저장
    val_df = pd.DataFrame([
        {
            'conversation_id': i,
            'turns': conv['turns'],
            'num_turns': conv['num_turns'],
            'conversation_text': conv['conversation_text'],
            'original_index': conv['original_index']
        }
        for i, conv in enumerate(validation_data)
    ])
    
    # Excel 파일로 저장
    train_df.to_excel(output_dir / "train_original_multiturn.xlsx", index=False)
    val_df.to_excel(output_dir / "validation_original_multiturn.xlsx", index=False)
    
    # JSON으로도 저장 (처리하기 쉽게)
    with open(output_dir / "train_conversations.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / "validation_conversations.json", 'w', encoding='utf-8') as f:
        json.dump(validation_data, f, indent=2, ensure_ascii=False)
    
    # Split 정보 저장
    with open(output_dir / "split_info.json", 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\n📁 Files saved to {output_dir}/")
    print(f"   - train_original_multiturn.xlsx ({len(train_data)} conversations)")
    print(f"   - validation_original_multiturn.xlsx ({len(validation_data)} conversations)")
    print(f"   - train_conversations.json")
    print(f"   - validation_conversations.json")
    print(f"   - split_info.json")
    
    return output_dir

def main():
    print("=== M2S-Guardrail Evaluation Data Preparation ===")
    
    # 1. 데이터 로드
    print("1. Loading source datasets...")
    data_sources = load_source_data()
    
    print(f"Loaded {len([k for k, v in data_sources.items() if v is not None])} datasets:")
    for name, data in data_sources.items():
        if data is not None:
            print(f"   - {name}: {len(data)} samples")
    
    # 2. Train/Validation 분할
    print("\n2. Creating train/validation splits...")
    train_data, validation_data, split_info = stratified_split_by_source(data_sources)
    
    # 3. 분할 정보 출력
    print("\n3. Split Summary:")
    print(f"   Total Train: {len(train_data)} conversations")
    print(f"   Total Validation: {len(validation_data)} conversations")
    
    for source, info in split_info.items():
        print(f"\n   {source}:")
        print(f"     Train: {info['train']}")
        print(f"     Validation: {info['validation']}")
        print(f"     Ratio: {info['train_ratio']:.2f}")
    
    # 4. 파일 저장
    print("\n4. Saving splits...")
    output_dir = save_splits(train_data, validation_data, split_info)
    
    print("\n✅ Data preparation completed!")
    print("\nNext steps:")
    print("1. Create M2S training datasets: python create_m2s_training_data.py")
    print("2. Run training experiments: python train_experiments.py")
    print("3. Evaluate models: python evaluate_models.py")

if __name__ == "__main__":
    main()