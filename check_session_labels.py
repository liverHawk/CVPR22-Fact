#!/usr/bin/env python3
"""セッションファイルのデータインデックスが正しいクラスを指しているか確認"""
import polars as pl
from pathlib import Path
import numpy as np

# データを読み込む
train_dir = 'data/CICIDS2017_flow_improved/train'
csv_files = sorted(Path(train_dir).glob('*.csv'))
dfs = [pl.read_csv(csv_file) for csv_file in csv_files]
train_df = pl.concat(dfs)

# ラベル列を確認
label_column = 'Label'
all_labels = sorted(train_df[label_column].unique().to_list())
print(f'All labels: {all_labels}')

# ラベルマッピングを作成
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
print(f'\nLabel to index:')
for label, idx in label_to_idx.items():
    print(f'  {idx}: {label}')

# ベースセッションのクラス
base_labels = ['BENIGN', 'Bot', 'DDoS', 'DoS', 'FTP-Patator']
base_class_indices = [label_to_idx[label] for label in base_labels]
print(f'\nBase classes (indices {base_class_indices}): {base_labels}')

# 新規クラス
new_labels = [label for label in all_labels if label not in base_labels]
new_class_indices = [label_to_idx[label] for label in new_labels]
print(f'\nNew classes (indices {new_class_indices}): {new_labels}')

# セッションファイルのインデックスを確認
for session in [2, 3, 4, 5, 6]:
    session_file = f'data/index_list/CICIDS2017_flow_improved/session_{session}.txt'
    with open(session_file, 'r') as f:
        indices = [int(line.strip()) for line in f if line.strip()]
    print(f'\n{"="*60}')
    print(f'Session {session}:')
    print(f'  Data indices: {indices}')
    # これらのインデックスのラベルを確認
    labels = train_df[indices][label_column].to_list()
    unique_labels = sorted(set(labels))
    print(f'  Labels: {labels}')
    print(f'  Unique labels: {unique_labels}')
    # クラスインデックスに変換
    class_indices = [label_to_idx[label] for label in labels]
    unique_class_indices = sorted(set(class_indices))
    print(f'  Unique class indices: {unique_class_indices}')
    
    # ベースクラスが含まれているか確認
    base_class_overlap = [idx for idx in unique_class_indices if idx in base_class_indices]
    if base_class_overlap:
        print(f'  ⚠️  WARNING: Base class indices found: {base_class_overlap}')
    else:
        print(f'  ✓ OK: Only new class indices')
