#!/usr/bin/env python3
"""
3개 데이터 소스 통합 및 M2S 변환 스크립트 (구조 파악 후 수정)
- data.xlsx: SafeMTData 원본 multi-turn 대화
- XGuard-Train/xguard-train.json: conversations 형태 
- hh-rlhf/: 압축된 JSONL 파일들 (chosen/rejected)
"""

import pandas as pd
import json
import gzip
import os
from pathlib import Path
import re
from sklearn.model_selection import train_test_split

class DatasetIntegrator:
    def __init__(self):
        self.all_conversations = []
    
    def load_original_data(self, file_path="data.xlsx"):
        """원본 data.xlsx 로드 - SafeMTData"""
        print(f"Loading {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return []
        
        df = pd.read_excel(file_path)
        conversations = []
        
        for _, row in df.iterrows():
            turns = []
            for i in range(1, 13):  # turn_1 to turn_12
                turn_col = f'turn_{i}'
                if turn_col in row and pd.notna(row[turn_col]) and str(row[turn_col]).strip():
                    turns.append(str(row[turn_col]).strip())
            
            if len(turns) >= 2:  # 최소 2턴 이상
                source = row.get('source', 'SafeMTData')  # SafeMTData_Attack600, SafeMTData_1K, MHJ_local
                conversations.append({
                    'turns': turns,
                    'source': source,
                    'num_turns': len(turns)
                })
        
        print(f"Loaded {len(conversations)} conversations from data.xlsx")
        return conversations
    
    def load_xguard_data(self, file_path="XGuard-Train/xguard-train.json"):
        """XGuard-Train JSON 파일 로드 - conversations 구조"""
        print(f"Loading {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found")
            return []
        
        conversations = []
        
        try:
            print("Reading large XGuard file... (may take a moment)")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"Processing {len(data)} XGuard items...")
            
            for i, item in enumerate(data):
                if i % 5000 == 0:  # 진행상황 표시
                    print(f"  Processed {i}/{len(data)} items...")
                    
                if 'conversations' in item:
                    # conversations 리스트에서 turns 추출
                    turns = []
                    for conv in item['conversations']:
                        if 'value' in conv and conv.get('from') == 'human':
                            # human의 발언만 turn으로 처리 (jailbreak 질문)
                            turns.append(str(conv['value']))
                    
                    if turns:
                        conversations.append({
                            'turns': turns,
                            'source': 'XGuard-Train',
                            'num_turns': len(turns)
                        })
                    
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
        
        print(f"Loaded {len(conversations)} conversations from XGuard-Train")
        return conversations
    
    def load_hh_rlhf_data(self, base_dir="hh-rlhf"):
        """hh-rlhf 압축 JSONL 파일들 로드"""
        print(f"Loading from {base_dir}...")
        
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            return []
        
        conversations = []
        subdirs = ['harmless-base', 'helpful-base', 'helpful-online', 'helpful-rejection-sampled']
        
        for subdir in subdirs:
            subdir_path = Path(base_dir) / subdir
            if not subdir_path.exists():
                continue
                
            print(f"Processing {subdir_path}...")
            
            # train.jsonl.gz와 test.jsonl.gz 처리
            for jsonl_file in subdir_path.glob("*.jsonl.gz"):
                try:
                    print(f"  Reading {jsonl_file}...")
                    
                    with gzip.open(jsonl_file, 'rt', encoding='utf-8') as f:
                        line_count = 0
                        for line in f:
                            data = json.loads(line)
                            line_count += 1
                            
                            # chosen과 rejected 모두 처리
                            for key in ['chosen', 'rejected']:
                                if key in data:
                                    text = str(data[key])
                                    turns = self.parse_conversation_text(text)
                                    if turns:
                                        conversations.append({
                                            'turns': turns,
                                            'source': f'hh-rlhf-{subdir}-{key}',
                                            'num_turns': len(turns)
                                        })
                    
                    print(f"    Processed {line_count} lines from {jsonl_file.name}")
                                    
                except Exception as e:
                    print(f"Error processing {jsonl_file}: {e}")
                    continue
        
        print(f"Loaded {len(conversations)} conversations from hh-rlhf")
        return conversations
    
    def parse_conversation_text(self, text):
        """Human: Assistant: 패턴 텍스트 파싱"""
        turns = []
        
        # Human:과 Assistant: 패턴으로 분할
        # 정규식으로 더 정확하게 분할
        parts = re.split(r'\n\n(Human:|Assistant:)', text)
        
        current_speaker = None
        current_content = ""
        
        for part in parts:
            part = part.strip()
            if part in ['Human:', 'Assistant:']:
                # 이전 내용 저장
                if current_speaker == 'Human:' and current_content.strip():
                    turns.append(current_content.strip())
                
                current_speaker = part
                current_content = ""
            else:
                current_content += part
        
        # 마지막 내용 처리
        if current_speaker == 'Human:' and current_content.strip():
            turns.append(current_content.strip())
        
        return [turn for turn in turns if turn]
    
    def create_m2s_formats(self, conversation):
        """단일 대화를 4가지 M2S 포맷으로 변환"""
        turns = conversation['turns']
        
        # Original (multi-turn을 하나로 합침)
        original_text = ' '.join(turns)
        
        # Hyphenize
        hyphenize_text = "Please answer the following list of questions in the given order.\\n"
        hyphenize_text += "Respond sequentially, and provide answers with hyphens only.\\n\\n"
        for turn in turns:
            hyphenize_text += f"- {turn}\\n"
        
        # Numberize  
        numberize_text = "Please answer the following numbered questions in order.\\n"
        numberize_text += "Provide complete responses for each numbered item.\\n\\n"
        for i, turn in enumerate(turns, 1):
            numberize_text += f"{i}. {turn}\\n"
        
        # Pythonize
        pythonize_text = "# Multi-step task execution\\n"
        pythonize_text += "# Execute the following steps in sequence\\n\\n"
        for i, turn in enumerate(turns, 1):
            var_name = f"step_{i}"
            # 따옴표 이스케이프 처리
            escaped_turn = turn.replace('"', '\\\\"').replace('\\n', '\\\\n')
            pythonize_text += f'{var_name} = "{escaped_turn}"\\n'
            pythonize_text += f"execute({var_name})\\n\\n"
        
        return {
            'original': original_text.strip(),
            'hyphenize': hyphenize_text.strip(),
            'numberize': numberize_text.strip(), 
            'pythonize': pythonize_text.strip()
        }
    
    def integrate_all_datasets(self):
        """모든 데이터셋 통합"""
        print("=== Integrating All Datasets ===")
        
        # 1. 각 데이터 소스 로드
        print("\\n1. Loading SafeMTData (data.xlsx)...")
        original_data = self.load_original_data()
        
        print("\\n2. Loading XGuard-Train...")  
        xguard_data = self.load_xguard_data()
        
        print("\\n3. Loading hh-rlhf...")
        hh_rlhf_data = self.load_hh_rlhf_data()
        
        # 2. 모든 대화 통합
        self.all_conversations = original_data + xguard_data + hh_rlhf_data
        
        print(f"\\n=== Integration Complete ===")
        print(f"Total conversations: {len(self.all_conversations)}")
        
        if not self.all_conversations:
            print("No conversations loaded. Exiting.")
            return [], []
        
        # 소스별 통계
        print("\\nSource breakdown:")
        sources = {}
        for conv in self.all_conversations:
            src = conv['source']
            sources[src] = sources.get(src, 0) + 1
        
        for src, count in sources.items():
            print(f"  - {src}: {count:,}")
        
        # 턴 수 통계
        turn_stats = {}
        for conv in self.all_conversations:
            turns = conv['num_turns']
            turn_stats[turns] = turn_stats.get(turns, 0) + 1
        
        print("\\nTurn distribution:")
        for turns in sorted(turn_stats.keys()):
            print(f"  - {turns} turns: {turn_stats[turns]:,}")
        
        # 3. Train/Validation 분할 (소스별 stratified)
        print("\\n4. Creating train/validation split...")
        try:
            # 소스가 너무 많으면 단순 분할
            source_counts = list(sources.values())
            min_source_count = min(source_counts)
            
            if min_source_count < 2:
                # stratify 불가능한 경우 단순 분할
                split_idx = int(len(self.all_conversations) * 0.8)
                train_convs = self.all_conversations[:split_idx]
                val_convs = self.all_conversations[split_idx:]
                print("Using simple split (stratify not possible)")
            else:
                train_convs, val_convs = train_test_split(
                    self.all_conversations,
                    test_size=0.2,
                    random_state=42,
                    stratify=[conv['source'] for conv in self.all_conversations]
                )
                print("Using stratified split")
        except:
            # 실패시 단순 분할
            split_idx = int(len(self.all_conversations) * 0.8)
            train_convs = self.all_conversations[:split_idx]
            val_convs = self.all_conversations[split_idx:]
            print("Using simple split (fallback)")
        
        print(f"Split: {len(train_convs):,} train, {len(val_convs):,} validation")
        
        # 4. M2S 포맷 생성 및 저장
        print("\\n5. Creating training datasets...")
        self.create_training_datasets(train_convs)
        
        print("\\n6. Saving validation dataset...")
        self.save_validation_dataset(val_convs)
        
        return train_convs, val_convs
    
    def create_training_datasets(self, train_conversations):
        """훈련 데이터셋 생성"""
        print("Creating M2S training datasets...")
        
        # 출력 디렉터리 생성
        output_dir = Path("training_data")
        output_dir.mkdir(exist_ok=True)
        
        # 5가지 포맷별 데이터셋
        datasets = {
            'original': [],
            'hyphenize': [],
            'numberize': [],
            'pythonize': [],
            'combined': []
        }
        
        import random
        random.seed(42)
        
        print(f"Processing {len(train_conversations):,} training conversations...")
        
        for i, conv in enumerate(train_conversations):
            if i % 5000 == 0:
                print(f"  Processed {i:,}/{len(train_conversations):,}...")
                
            # M2S 변환
            m2s_formats = self.create_m2s_formats(conv)
            
            # 각 포맷별로 저장
            for format_name, text in m2s_formats.items():
                datasets[format_name].append({
                    'text': text,
                    'label': 'unsafe',  # 모든 jailbreak 데이터는 unsafe
                    'source': conv['source'],
                    'num_turns': conv['num_turns']
                })
            
            # Combined: 랜덤하게 하나의 M2S 포맷 선택
            random_format = random.choice(['hyphenize', 'numberize', 'pythonize'])
            datasets['combined'].append({
                'text': m2s_formats[random_format],
                'label': 'unsafe',
                'source': conv['source'],
                'num_turns': conv['num_turns'],
                'format_used': random_format
            })
        
        # 파일로 저장
        print("\\nSaving datasets...")
        for format_name, data in datasets.items():
            print(f"  Saving {format_name}: {len(data):,} samples...")
            
            df = pd.DataFrame(data)
            
            excel_file = output_dir / f"train_{format_name}.xlsx"
            df.to_excel(excel_file, index=False)
            
            json_file = output_dir / f"train_{format_name}.json" 
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        print("All training datasets saved!")
    
    def save_validation_dataset(self, val_conversations):
        """검증 데이터셋 저장 (원본 multi-turn 형태 유지)"""
        print("Saving validation dataset...")
        
        output_dir = Path("evaluation_splits")
        output_dir.mkdir(exist_ok=True)
        
        # 검증 데이터는 원본 multi-turn 형태로 유지
        val_data = []
        for i, conv in enumerate(val_conversations):
            val_data.append({
                'conversation_id': i,
                'turns': conv['turns'],
                'num_turns': conv['num_turns'],
                'source': conv['source'],
                'conversation_text': ' [TURN] '.join(conv['turns'])
            })
        
        # 저장
        df_val = pd.DataFrame(val_data)
        df_val.to_excel(output_dir / "validation_original_multiturn.xlsx", index=False)
        
        with open(output_dir / "validation_conversations.json", 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"Validation dataset saved: {len(val_data):,} conversations")

def main():
    """메인 함수"""
    
    print("🚀 M2S-Guardrail Dataset Integration")
    print("=" * 50)
    
    integrator = DatasetIntegrator()
    train_convs, val_convs = integrator.integrate_all_datasets()
    
    print("\\n" + "=" * 50)
    print("✅ Dataset integration completed!")
    print("\\nGenerated files:")
    print("📁 training_data/")
    print("   - train_original.xlsx (original multi-turn format)")
    print("   - train_hyphenize.xlsx (M2S hyphen format)")
    print("   - train_numberize.xlsx (M2S number format)")  
    print("   - train_pythonize.xlsx (M2S python format)")
    print("   - train_combined.xlsx (random M2S formats)")
    print("📁 evaluation_splits/")
    print("   - validation_original_multiturn.xlsx (original format for evaluation)")
    print("\\nNext step: Run experiments with bash run_experiments_simple.sh")

if __name__ == "__main__":
    main()