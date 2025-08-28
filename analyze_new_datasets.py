import json
import pandas as pd
from pathlib import Path
import os

def analyze_xguard_train():
    """XGuard-Train 데이터셋 구조 분석"""
    print("=== Analyzing XGuard-Train Dataset ===")
    
    xguard_file = "XGuard-Train/xguard-train.json"
    
    if not os.path.exists(xguard_file):
        print(f"✗ XGuard-Train file not found: {xguard_file}")
        return
    
    try:
        with open(xguard_file, 'r', encoding='utf-8') as f:
            # JSON Lines 형태인지 확인
            first_line = f.readline()
            f.seek(0)
            
            if first_line.startswith('['):
                # JSON array 형태
                data = json.load(f)
            else:
                # JSON Lines 형태
                data = []
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        print(f"Total samples: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"First sample:")
            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        # 데이터 유형 분석
        if len(data) >= 5:
            print(f"\nAnalyzing first 5 samples:")
            for i in range(5):
                sample = data[i]
                print(f"Sample {i+1}:")
                for key, value in sample.items():
                    if isinstance(value, list):
                        print(f"  {key}: List with {len(value)} items")
                        if len(value) > 0:
                            print(f"    First item: {value[0]}")
                    elif isinstance(value, str) and len(value) > 50:
                        print(f"  {key}: {value[:50]}...")
                    else:
                        print(f"  {key}: {value}")
                print()
        
        return data
        
    except Exception as e:
        print(f"✗ Error analyzing XGuard-Train: {e}")
        return None

def analyze_hh_rlhf():
    """hh-rlhf 데이터셋 구조 분석"""
    print("=== Analyzing hh-rlhf Dataset ===")
    
    hh_rlhf_dir = "hh-rlhf"
    subdirs = [d for d in os.listdir(hh_rlhf_dir) if os.path.isdir(os.path.join(hh_rlhf_dir, d))]
    
    print(f"Subdirectories: {subdirs}")
    
    dataset_info = {}
    
    for subdir in subdirs:
        print(f"\n--- Analyzing {subdir} ---")
        subdir_path = os.path.join(hh_rlhf_dir, subdir)
        files = os.listdir(subdir_path)
        print(f"Files: {files}")
        
        # JSON 파일들 분석
        json_files = [f for f in files if f.endswith('.json')]
        
        subdir_data = []
        for json_file in json_files:
            file_path = os.path.join(subdir_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # JSON Lines 형태인지 확인
                    first_line = f.readline()
                    f.seek(0)
                    
                    if first_line.startswith('['):
                        # JSON array 형태
                        file_data = json.load(f)
                    else:
                        # JSON Lines 형태
                        file_data = []
                        for line in f:
                            if line.strip():
                                file_data.append(json.loads(line))
                
                print(f"  {json_file}: {len(file_data)} samples")
                subdir_data.extend(file_data)
                
                if len(file_data) > 0:
                    sample = file_data[0]
                    print(f"    Sample keys: {list(sample.keys())}")
                    
            except Exception as e:
                print(f"    ✗ Error reading {json_file}: {e}")
        
        dataset_info[subdir] = subdir_data
        
        # 첫 번째 샘플 상세 분석
        if len(subdir_data) > 0:
            sample = subdir_data[0]
            print(f"  First sample structure:")
            for key, value in sample.items():
                if isinstance(value, list):
                    print(f"    {key}: List with {len(value)} items")
                    if len(value) > 0 and isinstance(value[0], dict):
                        print(f"      First item keys: {list(value[0].keys())}")
                    elif len(value) > 0:
                        print(f"      First item: {value[0][:100] if isinstance(value[0], str) and len(value[0]) > 100 else value[0]}")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"    {key}: {value[:100]}...")
                else:
                    print(f"    {key}: {value}")
    
    return dataset_info

def analyze_multi_turn_potential(data, dataset_name):
    """Multi-turn conversation 가능성 분석"""
    print(f"\n=== Multi-Turn Analysis for {dataset_name} ===")
    
    if not data:
        print("No data to analyze")
        return
    
    multi_turn_count = 0
    conversation_structures = {}
    
    # 샘플들을 분석하여 multi-turn 패턴 찾기
    sample_size = min(100, len(data))
    
    for i in range(sample_size):
        sample = data[i] if isinstance(data, list) else data
        
        # 다양한 multi-turn 패턴 확인
        has_conversation = False
        turn_count = 0
        
        for key, value in sample.items():
            if isinstance(value, list):
                # List 형태의 대화
                if any(isinstance(item, dict) and any(k in item for k in ['human', 'assistant', 'user', 'role']) for item in value):
                    has_conversation = True
                    turn_count = len(value)
                    pattern = f"list_of_dicts_with_roles_{turn_count}turns"
                    conversation_structures[pattern] = conversation_structures.get(pattern, 0) + 1
            
            elif isinstance(value, str):
                # 문자열 내 대화 패턴
                if any(marker in value.lower() for marker in ['human:', 'assistant:', 'user:', '\n\n']):
                    # 대화 턴 수 추정
                    turn_markers = ['human:', 'assistant:', 'user:']
                    turn_count = sum(value.lower().count(marker) for marker in turn_markers)
                    if turn_count >= 2:
                        has_conversation = True
                        pattern = f"string_with_markers_{turn_count}turns"
                        conversation_structures[pattern] = conversation_structures.get(pattern, 0) + 1
        
        if has_conversation:
            multi_turn_count += 1
    
    print(f"Multi-turn samples found: {multi_turn_count}/{sample_size}")
    print(f"Conversation structures:")
    for structure, count in conversation_structures.items():
        print(f"  {structure}: {count}")
    
    return multi_turn_count, conversation_structures

if __name__ == "__main__":
    print("=== Dataset Structure Analysis ===")
    
    # XGuard-Train 분석
    xguard_data = analyze_xguard_train()
    if xguard_data:
        analyze_multi_turn_potential(xguard_data, "XGuard-Train")
    
    # hh-rlhf 분석
    hh_rlhf_data = analyze_hh_rlhf()
    for subdir, data in hh_rlhf_data.items():
        if data:
            analyze_multi_turn_potential(data, f"hh-rlhf/{subdir}")
    
    print("\n=== Analysis Complete ===")