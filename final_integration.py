#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import json
import re
import os
from tqdm import tqdm

def extract_human_turns_from_hh_rlhf(text):
    """hh-rlhf 텍스트에서 Human: 발언만 추출"""
    human_pattern = re.compile(r'Human: (.*?)(?=\n\nAssistant:|\n\nH:|\n\nHuman:|$)', re.DOTALL)
    human_turns = human_pattern.findall(text)
    return [turn.strip() for turn in human_turns]

def process_hh_rlhf_data():
    """hh-rlhf 데이터 처리 (개선된 버전)"""
    hh_rlhf_path = r'E:\Project\guardrail\hh-rlhf'
    all_records = []
    
    folders = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
    
    for folder in folders:
        folder_path = os.path.join(hh_rlhf_path, folder)
        if os.path.exists(folder_path):
            for file_type in ['train.jsonl', 'test.jsonl']:
                file_path = os.path.join(folder_path, file_type)
                if os.path.exists(file_path):
                    print(f"Processing {folder}/{file_type}...")
                    
                    # 파일 라인 수 계산
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines = sum(1 for _ in f)
                    
                    # 실제 처리
                    with open(file_path, 'r', encoding='utf-8') as f:
                        with tqdm(total=total_lines, desc=f"{folder}/{file_type}") as pbar:
                            for line_num, line in enumerate(f):
                                try:
                                    data = json.loads(line.strip())
                                    
                                    for response_type in ['chosen', 'rejected']:
                                        if response_type in data:
                                            human_turns = extract_human_turns_from_hh_rlhf(data[response_type])
                                            
                                            if human_turns:
                                                record = {
                                                    'source': f'hh-rlhf_{folder}_{response_type}',
                                                    'base_prompt': human_turns[0] if human_turns else '',
                                                    'jailbreak_turns': json.dumps({f'turn_{i+1}': turn for i, turn in enumerate(human_turns)}, ensure_ascii=False),
                                                    'turn_type': 'multi' if len(human_turns) > 1 else 'single',
                                                    'num_turns': len(human_turns),
                                                    'meta': json.dumps({'file': f'{folder}/{file_type}', 'line': line_num, 'response_type': response_type}, ensure_ascii=False)
                                                }
                                                
                                                for i in range(1, 13):
                                                    if i <= len(human_turns):
                                                        record[f'turn_{i}'] = human_turns[i-1]
                                                    else:
                                                        record[f'turn_{i}'] = None
                                                
                                                all_records.append(record)
                                except Exception as e:
                                    if line_num % 1000 == 0:  # 1000개마다 에러 출력
                                        print(f"Error at line {line_num}: {e}")
                                
                                pbar.update(1)
    
    return all_records

def process_xguard_data():
    """XGuard-Train 데이터 처리 (개선된 버전)"""
    xguard_path = r'E:\Project\guardrail\XGuard-Train\xguard-train.json'
    all_records = []
    
    print("Processing XGuard-Train data...")
    
    with open(xguard_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with tqdm(total=len(data), desc="XGuard-Train") as pbar:
        for idx, item in enumerate(data):
            try:
                conversations = item.get('conversations', [])
                
                human_turns = []
                for conv in conversations:
                    if conv.get('from') == 'human':
                        human_turns.append(conv.get('value', ''))
                
                if human_turns:
                    record = {
                        'source': 'XGuard-Train',
                        'base_prompt': human_turns[0] if human_turns else '',
                        'jailbreak_turns': json.dumps({f'turn_{i+1}': turn for i, turn in enumerate(human_turns)}, ensure_ascii=False),
                        'turn_type': 'multi' if len(human_turns) > 1 else 'single',
                        'num_turns': len(human_turns),
                        'meta': json.dumps({'record_index': idx}, ensure_ascii=False)
                    }
                    
                    for i in range(1, 13):
                        if i <= len(human_turns):
                            record[f'turn_{i}'] = human_turns[i-1]
                        else:
                            record[f'turn_{i}'] = None
                    
                    all_records.append(record)
            except Exception as e:
                if idx % 1000 == 0:
                    print(f"Error at XGuard record {idx}: {e}")
            
            pbar.update(1)
    
    return all_records

def integrate_all_data():
    """모든 데이터 통합 (개선된 버전)"""
    print("=== 전체 데이터 통합 시작 ===")
    
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
    print("\n=== 데이터 결합 및 저장 ===")
    all_new_records = hh_rlhf_records + xguard_records
    print(f"새로 추가되는 레코드: {len(all_new_records)} records")
    
    if all_new_records:
        new_df = pd.DataFrame(all_new_records)
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        
        print(f"\n=== 통합 완료 ===")
        print(f"최종 데이터: {len(final_df)} records")
        print(f"- 기존 data.xlsx: {len(existing_df)} records")
        print(f"- hh-rlhf: {len(hh_rlhf_records)} records")
        print(f"- XGuard-Train: {len(xguard_records)} records")
        
        # 결과 저장
        output_path = r'E:\Project\guardrail\integrated_data.xlsx'
        print(f"Saving to {output_path}...")
        final_df.to_excel(output_path, index=False)
        print(f"통합된 데이터가 저장되었습니다.")
        
        # 통계 출력
        print("\n=== 데이터 소스별 통계 ===")
        source_counts = final_df['source'].value_counts()
        for source, count in source_counts.head(10).items():
            print(f"{source}: {count} records")
        
        if len(source_counts) > 10:
            print(f"... and {len(source_counts) - 10} more sources")
        
        # 기본 통계
        print(f"\n=== 기본 통계 ===")
        print(f"평균 턴 수: {final_df['num_turns'].mean():.2f}")
        print(f"최대 턴 수: {final_df['num_turns'].max()}")
        print(f"단일 턴: {len(final_df[final_df['num_turns'] == 1])} records")
        print(f"다중 턴: {len(final_df[final_df['num_turns'] > 1])} records")
        
        return final_df
    else:
        print("새로 추가할 레코드가 없습니다.")
        return existing_df

if __name__ == "__main__":
    try:
        integrated_df = integrate_all_data()
        print("\n✅ 데이터 통합이 성공적으로 완료되었습니다!")
    except Exception as e:
        print(f"❌ Error during integration: {e}")
        import traceback
        traceback.print_exc()