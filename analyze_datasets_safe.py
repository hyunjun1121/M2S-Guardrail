import json
import pandas as pd
import os

def analyze_xguard_train():
    """XGuard-Train 데이터셋 구조 분석"""
    print("=== Analyzing XGuard-Train Dataset ===")
    
    xguard_file = "XGuard-Train/xguard-train.json"
    
    if not os.path.exists(xguard_file):
        print("XGuard-Train file not found")
        return None
    
    try:
        with open(xguard_file, 'r', encoding='utf-8') as f:
            # 첫 줄만 읽어서 구조 확인
            first_line = f.readline()
            f.seek(0)
            
            if first_line.startswith('['):
                data = json.load(f)
            else:
                # JSON Lines 형태 - 처음 몇 개만 읽기
                data = []
                for i, line in enumerate(f):
                    if i >= 5:  # 처음 5개만
                        break
                    if line.strip():
                        data.append(json.loads(line))
        
        print(f"Total samples analyzed: {len(data)}")
        
        if len(data) > 0:
            sample = data[0]
            print(f"Sample keys: {list(sample.keys())}")
            
            # conversations 구조 분석
            if 'conversations' in sample:
                conversations = sample['conversations']
                print(f"Conversations type: {type(conversations)}")
                print(f"Number of turns: {len(conversations)}")
                
                if len(conversations) > 0:
                    first_turn = conversations[0]
                    print(f"First turn keys: {list(first_turn.keys())}")
                    print(f"First turn 'from': {first_turn.get('from', 'N/A')}")
                    
                    # 'value' 필드의 길이만 출력
                    if 'value' in first_turn:
                        value_len = len(first_turn['value'])
                        print(f"First turn 'value' length: {value_len} characters")
        
        return data
        
    except Exception as e:
        print(f"Error analyzing XGuard-Train: {str(e)}")
        return None

def analyze_hh_rlhf():
    """hh-rlhf 데이터셋 구조 분석"""
    print("\n=== Analyzing hh-rlhf Dataset ===")
    
    hh_rlhf_dir = "hh-rlhf"
    if not os.path.exists(hh_rlhf_dir):
        print("hh-rlhf directory not found")
        return {}
    
    subdirs = [d for d in os.listdir(hh_rlhf_dir) if os.path.isdir(os.path.join(hh_rlhf_dir, d))]
    print(f"Subdirectories: {subdirs}")
    
    dataset_info = {}
    
    for subdir in subdirs[:2]:  # 처음 2개만 분석
        print(f"\n--- Analyzing {subdir} ---")
        subdir_path = os.path.join(hh_rlhf_dir, subdir)
        files = os.listdir(subdir_path)
        json_files = [f for f in files if f.endswith('.json')][:1]  # 첫 번째 파일만
        
        subdir_data = []
        for json_file in json_files:
            file_path = os.path.join(subdir_path, json_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    f.seek(0)
                    
                    if first_line.startswith('['):
                        file_data = json.load(f)
                        file_data = file_data[:3]  # 처음 3개만
                    else:
                        file_data = []
                        for i, line in enumerate(f):
                            if i >= 3:  # 처음 3개만
                                break
                            if line.strip():
                                file_data.append(json.loads(line))
                
                print(f"  {json_file}: {len(file_data)} samples analyzed")
                subdir_data.extend(file_data)
                
                if len(file_data) > 0:
                    sample = file_data[0]
                    print(f"    Sample keys: {list(sample.keys())}")
                    
                    # 각 키의 데이터 타입과 길이 정보
                    for key, value in sample.items():
                        if isinstance(value, list):
                            print(f"    {key}: List with {len(value)} items")
                        elif isinstance(value, str):
                            print(f"    {key}: String with {len(value)} characters")
                        else:
                            print(f"    {key}: {type(value).__name__}")
                    
            except Exception as e:
                print(f"    Error reading {json_file}: {str(e)}")
        
        dataset_info[subdir] = subdir_data
    
    return dataset_info

def count_conversation_turns(data, dataset_name):
    """Multi-turn conversation 수 분석"""
    print(f"\n=== Multi-Turn Analysis for {dataset_name} ===")
    
    if not data:
        print("No data to analyze")
        return
    
    multi_turn_patterns = {}
    
    sample_data = data[:10] if isinstance(data, list) else [data]  # 최대 10개 샘플만
    
    for sample in sample_data:
        if 'conversations' in sample:
            conv_len = len(sample['conversations'])
            pattern = f"{conv_len}_turns"
            multi_turn_patterns[pattern] = multi_turn_patterns.get(pattern, 0) + 1
        
        # hh-rlhf 패턴 확인
        elif 'chosen' in sample and 'rejected' in sample:
            # chosen/rejected 문자열에서 턴 수 추정
            chosen_turns = sample['chosen'].count('\n\nHuman:') + sample['chosen'].count('\n\nAssistant:')
            pattern = f"hh_rlhf_{chosen_turns}_markers"
            multi_turn_patterns[pattern] = multi_turn_patterns.get(pattern, 0) + 1
    
    print(f"Conversation patterns found:")
    for pattern, count in multi_turn_patterns.items():
        print(f"  {pattern}: {count}")
    
    return multi_turn_patterns

if __name__ == "__main__":
    print("=== Dataset Structure Analysis (Safe Mode) ===")
    
    # XGuard-Train 분석
    xguard_data = analyze_xguard_train()
    if xguard_data:
        count_conversation_turns(xguard_data, "XGuard-Train")
    
    # hh-rlhf 분석  
    hh_rlhf_data = analyze_hh_rlhf()
    for subdir, data in hh_rlhf_data.items():
        if data:
            count_conversation_turns(data, f"hh-rlhf/{subdir}")
    
    print("\n=== Analysis Complete ===")