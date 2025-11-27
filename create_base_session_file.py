#!/usr/bin/env python3
"""
ベースセッション（session 0）のファイルのみを作成するスクリプト
"""

import os
import sys

# create_session_files.pyの関数を再利用
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_session_files import _fast_process, _column_adjustment
import pandas as pd
import numpy as np
from collections import defaultdict


def create_base_session_file(
    train_csv="data/train.csv",
    output_dir="data/index_list/CICIDS2017_improved",
    base_class=15,
):
    """
    ベースセッション（session 0）のファイルのみを作成
    
    Args:
        train_csv: 訓練データのCSVファイルパス
        output_dir: セッションファイルの出力ディレクトリ
        base_class: ベースクラス数
    """
    print(f"Loading training data from {train_csv}...")
    df = pd.read_csv(train_csv)

    print(f"Total rows before preprocessing: {len(df)}")

    # 前処理を適用
    df = _fast_process(df)
    df = _column_adjustment(df)
    df = df.reset_index(drop=True)

    print(f"Total rows after preprocessing: {len(df)}")

    # ラベル列を確認
    if "Label" in df.columns:
        label_col = "Label"
    else:
        label_col = df.columns[-1]

    print(f"Label column: {label_col}")

    # ラベルをエンコード
    unique_labels = sorted(df[label_col].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df["label_idx"] = df[label_col].map(label_to_idx)

    print(f"Total classes: {len(unique_labels)}")
    print(f"Base classes: {base_class}")

    # クラスごとにデータインデックスをグループ化
    class_indices = defaultdict(list)
    for idx, label_idx in enumerate(df["label_idx"]):
        class_indices[label_idx].append(idx)

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # session_0.txt（ベースセッション）を作成
    print(f"\nCreating session_0.txt (base session, classes 0-{base_class - 1})...")
    base_indices = []
    for class_idx in range(base_class):
        base_indices.extend(class_indices[class_idx])
        count = len(class_indices[class_idx])
        print(f"  Class {class_idx} ({unique_labels[class_idx]}): {count} samples")

    session_0_path = os.path.join(output_dir, "session_0.txt")
    with open(session_0_path, "w") as f:
        for idx in base_indices:
            f.write(f"{idx}\n")
    
    print(f"\nSaved {len(base_indices)} indices to {session_0_path}")
    print("Done!")


if __name__ == "__main__":
    import argparse
    from utils import load_params_yaml

    # params.yamlからデフォルト値を読み込む
    try:
        params = load_params_yaml("params.yaml")
        session_params = params.get("create_sessions", {})
        
        default_train_csv = session_params.get("train_csv", "data/train.csv")
        default_output_dir = session_params.get("output_dir", "data/index_list/CICIDS2017_improved")
        default_base_class = session_params.get("base_class", 15)
    except Exception as e:
        print(f"Warning: Failed to load params.yaml: {e}")
        default_train_csv = "data/train.csv"
        default_output_dir = "data/index_list/CICIDS2017_improved"
        default_base_class = 15

    parser = argparse.ArgumentParser(description="Create base session file")
    parser.add_argument("--train_csv", type=str, default=default_train_csv)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--base_class", type=int, default=default_base_class)

    args = parser.parse_args()

    create_base_session_file(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        base_class=args.base_class,
    )
