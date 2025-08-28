#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_dataset_by_source(data_file, test_size=0.2, random_state=42):
    """각 source별로 비율을 맞춰서 train/validation 분할"""
    
    print(f"데이터 로드: {data_file}")
    df = pd.read_excel(data_file)
    
    print(f"전체 데이터: {len(df)} rows")
    
    # Source별 분포 확인
    print("\n=== Source별 분포 ===")
    source_counts = df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"{source}: {count} ({count/len(df)*100:.1f}%)")
    
    # 각 source별로 train/val 분할
    train_dfs = []
    val_dfs = []
    
    for source in source_counts.index:
        source_df = df[df['source'] == source]
        
        if len(source_df) >= 2:  # 최소 2개 이상인 경우만 분할
            train_source, val_source = train_test_split(
                source_df, 
                test_size=test_size, 
                random_state=random_state,
                shuffle=True
            )
            train_dfs.append(train_source)
            val_dfs.append(val_source)
            print(f"{source}: train={len(train_source)}, val={len(val_source)}")
        else:
            # 데이터가 1개뿐이면 train에만 포함
            train_dfs.append(source_df)
            print(f"{source}: train={len(source_df)}, val=0 (too few samples)")
    
    # 데이터프레임 결합
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    
    # 셔플
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    if len(val_df) > 0:
        val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"\n=== 분할 결과 ===")
    print(f"Train: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    
    # 파일명 생성
    base_name = data_file.replace('.xlsx', '')
    train_file = f"{base_name}_train.xlsx"
    val_file = f"{base_name}_val.xlsx"
    
    # 저장
    train_df.to_excel(train_file, index=False)
    print(f"Train 데이터 저장: {train_file}")
    
    if len(val_df) > 0:
        val_df.to_excel(val_file, index=False)
        print(f"Validation 데이터 저장: {val_file}")
    
    # Validation에서도 source 분포 확인
    if len(val_df) > 0:
        print(f"\n=== Validation Source 분포 ===")
        val_source_counts = val_df['source'].value_counts()
        for source, count in val_source_counts.items():
            print(f"{source}: {count} ({count/len(val_df)*100:.1f}%)")
    
    return train_file, val_file

def main():
    """모든 데이터셋에 대해 train/val 분할 수행"""
    
    datasets = [
        "integrated_data.xlsx",
        "m2s_hyphenize_data.xlsx", 
        "m2s_numberize_data.xlsx",
        "m2s_pythonize_data.xlsx",
        "m2s_combined_all_data.xlsx"
    ]
    
    results = {}
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\n{'='*60}")
            print(f"처리 중: {dataset}")
            print(f"{'='*60}")
            
            try:
                train_file, val_file = split_dataset_by_source(dataset)
                results[dataset] = {
                    'train': train_file,
                    'validation': val_file
                }
            except Exception as e:
                print(f"❌ {dataset} 처리 실패: {e}")
        else:
            print(f"⚠️  파일 없음: {dataset}")
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📋 분할 완료 요약")
    print(f"{'='*60}")
    
    for dataset, files in results.items():
        print(f"\n📁 {dataset}")
        print(f"  ├── Train: {files['train']}")
        print(f"  └── Validation: {files['validation']}")

if __name__ == "__main__":
    main()