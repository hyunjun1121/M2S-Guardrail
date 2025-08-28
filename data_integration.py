#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
import os
from pathlib import Path

def extract_human_turns_from_hh_rlhf(text):
    """hh-rlhf 텍스트에서 Human: 발언만 추출"""
    human_pattern = re.compile(r'Human: (.*?)(?=\n\nAssistant:|\n\nH:|\n\nHuman:|$)', re.DOTALL)
    human_turns = human_pattern.findall(text)
    # 각 turn의 공백 정리
    return [turn.strip() for turn in human_turns]

def process_hh_rlhf_data():
    """hh-rlhf 데이터 처리"""
    hh_rlhf_path = r'E:\Project\guardrail\hh-rlhf'
    all_records = []
    
    # 각 폴더별로 처리
    folders = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    
    for folder in folders:
        folder_path = os.path.join(hh_rlhf_path, folder)
        if os.path.exists(folder_path):
            # train.jsonl과 test.jsonl 처리
            for file_type in ['train.jsonl', 'test.jsonl']:
                file_path = os.path.join(folder_path, file_type)
                if os.path.exists(file_path):
                    print(f"Processing {folder}/{file_type}...")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            try:
                                data = json.loads(line.strip())
                                
                                # chosen과 rejected 각각 처리
                                for response_type in ['chosen', 'rejected']:
                                    if response_type in data:
                                        human_turns = extract_human_turns_from_hh_rlhf(data[response_type])
                                        
                                        if human_turns:  # Human 발언이 있는 경우만
                                            record = {
                                                'source': f'hh-rlhf_{folder}_{response_type}',
                                                'base_prompt': human_turns[0] if human_turns else '',
                                                'jailbreak_turns': json.dumps({f'turn_{i+1}': turn for i, turn in enumerate(human_turns)}, ensure_ascii=False),
                                                'turn_type': 'multi' if len(human_turns) > 1 else 'single',
                                                'num_turns': len(human_turns),
                                                'meta': json.dumps({'file': f'{folder}/{file_type}', 'line': line_num, 'response_type': response_type}, ensure_ascii=False)
                                            }
                                            
                                            # turn_1부터 turn_12까지 채우기
                                            for i in range(1, 13):
                                                if i <= len(human_turns):
                                                    record[f'turn_{i}'] = human_turns[i-1]
                                                else:
                                                    record[f'turn_{i}'] = None
                                            
                                            all_records.append(record)
                            except Exception as e:
                                print(f"Error processing line {line_num} in {folder}/{file_type}: {e}")
    
    # red-team-attempts 별도 처리 (일단 스킵 - JSON 형식 문제)
    print("Skipping red-team-attempts due to JSON format issues...")
    
    return all_records

def process_xguard_data():
    """XGuard-Train 데이터 처리"""
    xguard_path = r'E:\Project\guardrail\XGuard-Train\xguard-train.json'
    all_records = []
    
    print("Processing XGuard-Train data...")
    
    with open(xguard_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    for idx, item in enumerate(data):
        try:
            conversations = item.get('conversations', [])
            
            # from="human"인 대화만 추출
            human_turns = []
            for conv in conversations:
                if conv.get('from') == 'human':
                    human_turns.append(conv.get('value', ''))
            
            if human_turns:  # Human 발언이 있는 경우만
                record = {
                    'source': 'XGuard-Train',
                    'base_prompt': human_turns[0] if human_turns else '',
                    'jailbreak_turns': json.dumps({f'turn_{i+1}': turn for i, turn in enumerate(human_turns)}, ensure_ascii=False),
                    'turn_type': 'multi' if len(human_turns) > 1 else 'single',
                    'num_turns': len(human_turns),
                    'meta': json.dumps({'record_index': idx}, ensure_ascii=False)
                }
                
                # turn_1부터 turn_12까지 채우기
                for i in range(1, 13):
                    if i <= len(human_turns):
                        record[f'turn_{i}'] = human_turns[i-1]
                    else:
                        record[f'turn_{i}'] = None
                
                all_records.append(record)
        except Exception as e:
            print(f"Error processing XGuard record {idx}: {e}")
    
    return all_records

def integrate_all_data():
    """모든 데이터 통합"""
    print("=== 데이터 통합 시작 ===")
    
    # 기존 data.xlsx 읽기
    print("Loading existing data.xlsx...")
    existing_df = pd.read_excel(r'E:\Project\guardrail\data.xlsx')
    print(f"기존 데이터: {len(existing_df)} records")
    
    # hh-rlhf 데이터 처리
    print("\n=== hh-rlhf 데이터 처리 ===")
    hh_rlhf_records = process_hh_rlhf_data()
    print(f"hh-rlhf 데이터: {len(hh_rlhf_records)} records")
    
    # XGuard-Train 데이터 처리
    print("\n=== XGuard-Train 데이터 처리 ===")
    xguard_records = process_xguard_data()
    print(f"XGuard-Train 데이터: {len(xguard_records)} records")
    
    # 모든 데이터 합치기
    all_new_records = hh_rlhf_records + xguard_records
    new_df = pd.DataFrame(all_new_records)
    
    # 기존 데이터와 합치기
    final_df = pd.concat([existing_df, new_df], ignore_index=True)
    
    print(f"\n=== 통합 완료 ===")
    print(f"최종 데이터: {len(final_df)} records")
    print(f"- 기존 data.xlsx: {len(existing_df)} records")
    print(f"- hh-rlhf: {len(hh_rlhf_records)} records")
    print(f"- XGuard-Train: {len(xguard_records)} records")
    
    # 결과 저장
    output_path = r'E:\Project\guardrail\integrated_data.xlsx'
    final_df.to_excel(output_path, index=False)
    print(f"\n통합된 데이터가 {output_path}에 저장되었습니다.")
    
    # 통계 출력
    print("\n=== 데이터 소스별 통계 ===")
    print(final_df['source'].value_counts())
    
    return final_df

if __name__ == "__main__":
    try:
        integrated_df = integrate_all_data()
        print("\n데이터 통합이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"Error during integration: {e}")
        import traceback
        traceback.print_exc()