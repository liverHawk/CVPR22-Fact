#!/usr/bin/env python3
"""
CICIDS2017_improvedデータセットのセッションファイルを作成するスクリプト
各セッションで新しいクラスのfew-shotサンプルを選択
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import json


def create_session_files(
    train_csv="data/CICIDS2017_improved/train.csv",
    output_dir="data/index_list/CICIDS2017_improved",
    base_class=15,
    num_classes=27,
    way=2,
    shot=5,
    random_state=42,
):
    """
    セッションファイルを作成

    Args:
        train_csv: 訓練データのCSVファイルパス
        output_dir: セッションファイルの出力ディレクトリ
        base_class: ベースクラス数
        num_classes: 総クラス数
        way: セッションごとの新しいクラス数
        shot: クラスごとのサンプル数（few-shot）
        random_state: ランダムシード
    """
    print(f"Loading training data from {train_csv}...")
    df = pd.read_csv(train_csv)

    print(f"Total rows: {len(df)}")

    # ラベル列を確認
    if "Label" in df.columns:
        label_col = "Label"
    else:
        label_col = df.columns[-1]

    print(f"Label column: {label_col}")

    # ラベルの統合（前処理に合わせる）
    labels = df[label_col].unique()
    for label in labels:
        if "Attempted" in label:
            df.loc[df[label_col] == label, label_col] = "BENIGN"
        if "Web Attack" in label:
            df.loc[df[label_col] == label, label_col] = "Web Attack"
        if "Infiltration" in label:
            df.loc[df[label_col] == label, label_col] = "Infiltration"
        if "DoS" in label and label != "DDoS":
            df.loc[df[label_col] == label, label_col] = "DoS"

    # ラベルをエンコード（0から始まる連番）
    unique_labels = sorted(df[label_col].unique())
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    df["label_idx"] = df[label_col].map(label_to_idx)

    print(f"Total classes: {len(unique_labels)}")
    print(f"Classes: {unique_labels}")

    # クラスごとにデータインデックスをグループ化
    class_indices = defaultdict(list)
    for idx, label_idx in enumerate(df["label_idx"]):
        class_indices[label_idx].append(idx)

    print("\nClass distribution:")
    for label_idx in range(len(unique_labels)):
        count = len(class_indices[label_idx])
        print(f"  Class {label_idx} ({unique_labels[label_idx]}): {count} samples")

    with open("data/class_indices.json", "w") as f:
        json.dump(class_indices, f, indent=4)

    with open("data/column_names.txt", "w") as f:
        columns = df.columns.tolist()
        for column in columns:
            f.write(f"{column}\n")

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # セッション0（base session）のファイルを作成
    # session_1.txtにはベースクラスのすべてのサンプルインデックスを含む
    print(f"\nCreating session_0.txt (base session, classes 0-{base_class - 1})...")
    base_indices = []
    for class_idx in range(base_class):
        base_indices.extend(class_indices[class_idx])

    session_0_path = os.path.join(output_dir, "session_0.txt")
    with open(session_0_path, "w") as f:
        for idx in base_indices:
            f.write(f"{idx}\n")
    print(f"  Saved {len(base_indices)} indices to {session_0_path}")

    # 各セッションのファイルを作成
    num_sessions = (num_classes - base_class) // way
    print(f"\nCreating session files for {num_sessions} incremental sessions...")

    np.random.seed(random_state)

    for session in range(1, num_sessions + 1):
        session_num = session  # session_1.txt, session_2.txt, ...
        start_class = base_class + (session - 1) * way
        end_class = min(start_class + way, num_classes)

        print(
            f"\nCreating session_{session_num}.txt (session {session}, classes {start_class}-{end_class - 1})..."
        )

        session_indices = []
        for class_idx in range(start_class, end_class):
            class_samples = class_indices[class_idx]
            if len(class_samples) < shot:
                print(
                    f"  Warning: Class {class_idx} ({unique_labels[class_idx]}) has only {len(class_samples)} samples, using all"
                )
                selected = class_samples
            else:
                # ランダムにshot個のサンプルを選択
                selected = np.random.choice(
                    class_samples, size=shot, replace=False
                ).tolist()

            session_indices.extend(selected)
            print(
                f"  Class {class_idx} ({unique_labels[class_idx]}): selected {len(selected)} samples"
            )

        session_path = os.path.join(output_dir, f"session_{session_num}.txt")
        with open(session_path, "w") as f:
            for idx in session_indices:
                f.write(f"{idx}\n")
        print(f"  Saved {len(session_indices)} indices to {session_path}")

    print("\nDone!")
    print(f"\nSession files created in {output_dir}:")
    for session_num in range(1, num_sessions + 2):
        session_file = os.path.join(output_dir, f"session_{session_num}.txt")
        if os.path.exists(session_file):
            with open(session_file, "r") as f:
                num_lines = len(f.readlines())
            print(f"  session_{session_num}.txt: {num_lines} indices")


if __name__ == "__main__":
    import argparse
    import sys
    import os

    # utilsモジュールをインポート（パスを追加）
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from utils import load_params_yaml
    except ImportError:

        def load_params_yaml(yaml_path="params.yaml"):
            return {}

    # params.yamlからデフォルト値を読み込む
    try:
        params = load_params_yaml("params.yaml")
        session_params = params.get("create_sessions", {})
        common_params = params.get("common", {})

        default_train_csv = session_params.get(
            "train_csv", "data/CICIDS2017_improved/train.csv"
        )
        default_output_dir = session_params.get(
            "output_dir", "data/index_list/CICIDS2017_improved"
        )
        default_base_class = session_params.get("base_class", 15)
        default_num_classes = session_params.get("num_classes", 27)
        default_way = session_params.get("way", 2)
        default_shot = session_params.get("shot", 5)
        default_random_state = session_params.get(
            "random_state", common_params.get("seed", 42)
        )
    except Exception as e:
        print(f"Warning: Failed to load params.yaml: {e}, using hardcoded defaults")
        default_train_csv = "data/CICIDS2017_improved/train.csv"
        default_output_dir = "data/index_list/CICIDS2017_improved"
        default_base_class = 15
        default_num_classes = 27
        default_way = 2
        default_shot = 5
        default_random_state = 42

    parser = argparse.ArgumentParser(
        description="Create session files for CICIDS2017_improved"
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default=default_train_csv,
        help="Training CSV file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Output directory for session files",
    )
    parser.add_argument(
        "--base_class",
        type=int,
        default=default_base_class,
        help="Number of base classes",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=default_num_classes,
        help="Total number of classes",
    )
    parser.add_argument(
        "--way", type=int, default=default_way, help="Number of new classes per session"
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=default_shot,
        help="Number of samples per class (few-shot)",
    )
    parser.add_argument(
        "--random_state", type=int, default=default_random_state, help="Random seed"
    )

    args = parser.parse_args()

    create_session_files(
        train_csv=args.train_csv,
        output_dir=args.output_dir,
        base_class=args.base_class,
        num_classes=args.num_classes,
        way=args.way,
        shot=args.shot,
        random_state=args.random_state,
    )
