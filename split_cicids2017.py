#!/usr/bin/env python3
"""
CICIDS2017_improvedデータセットをtrain/testに分割するスクリプト
指定されたGitHubリポジトリのブランチ（cicids2017）の前処理パターンを参照
"""

import os
import pandas as pd
# import numpy as np
from sklearn.model_selection import train_test_split


def load_and_split_cicids2017(
    data_dir="data/CICIDS2017_improved",
    output_dir="data/CICIDS2017_improved",
    test_size=0.2,
    random_state=42,
    stratify_by_label=True,
):
    """
    CICIDS2017_improvedデータセットを読み込んでtrain/testに分割

    Args:
        data_dir: 入力データディレクトリ
        output_dir: 出力ディレクトリ
        test_size: テストセットの割合（デフォルト: 0.2 = 20%）
        random_state: ランダムシード
        stratify_by_label: ラベルで層化サンプリングを行うかどうか
    """
    print(f"Loading data from {data_dir}...")

    # すべてのCSVファイルを読み込む
    csv_files = [
        "monday.csv",
        "tuesday.csv",
        "wednesday.csv",
        "thursday.csv",
        "friday.csv",
    ]
    dataframes = []

    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        if os.path.exists(file_path):
            print(f"Loading {csv_file}...")
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} rows from {csv_file}")
            dataframes.append(df)
        else:
            print(f"Warning: {csv_file} not found, skipping...")

    if not dataframes:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # すべてのデータフレームを結合
    print("\nCombining all dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows: {len(combined_df)}")

    # ラベル列を確認
    if "Label" in combined_df.columns:
        label_col = "Label"
    else:
        # 最後の列をラベルとして使用
        label_col = combined_df.columns[-1]
        print(f"Warning: 'Label' column not found, using '{label_col}' as label column")
    
        # ラベルの統合（前処理に合わせる）
    labels = combined_df[label_col].unique()
    for label in labels:
        if "Attempted" in label:
            combined_df.loc[combined_df[label_col] == label, label_col] = "BENIGN"
        if "Web Attack" in label:
            combined_df.loc[combined_df[label_col] == label, label_col] = "Web Attack"
        if "Infiltration" in label:
            combined_df.loc[combined_df[label_col] == label, label_col] = "Infiltration"
        if "DoS" in label and label != "DDoS":
            combined_df.loc[combined_df[label_col] == label, label_col] = "DoS"

    print(f"\nLabel column: {label_col}")
    print(f"Unique labels: {combined_df[label_col].unique()}")
    print(f"Label distribution:\n{combined_df[label_col].value_counts()}")

    # 特徴量とラベルを分離
    feature_cols = [col for col in combined_df.columns if col != label_col]
    X = combined_df[feature_cols]
    y = combined_df[label_col]

    print(f"\nFeature columns: {len(feature_cols)}")
    print(f"Sample feature columns: {feature_cols[:10]}...")

    # train/testに分割
    print(
        f"\nSplitting data into train ({1 - test_size:.0%}) and test ({test_size:.0%})..."
    )

    if stratify_by_label:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            print("Stratified split completed")
        except ValueError as e:
            print(f"Warning: Stratified split failed ({e}), using random split instead")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    # train/testデータフレームを作成
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"\nTrain set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    print(f"\nTrain label distribution:\n{train_df[label_col].value_counts()}")
    print(f"\nTest label distribution:\n{test_df[label_col].value_counts()}")

    # 保存
    output_dir = os.path.dirname(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print(f"\nSaving train data to {train_path}...")
    train_df.to_csv(train_path, index=False)
    print(f"Saved {len(train_df)} rows")

    print(f"Saving test data to {test_path}...")
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(test_df)} rows")

    print("\nDone!")
    return train_df, test_df


if __name__ == "__main__":
    import argparse
    import sys
    import os

    root_dir = "/Users/toshi_pro/Documents/school/cvpr22-fact"

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
        split_params = params.get("split_cicids", {})
        common_params = params.get("common", {})

        default_data_dir = split_params.get("data_dir", "data/CICIDS2017_improved")
        default_output_dir = split_params.get("output_dir", "data/CICIDS2017_improved")
        default_test_size = split_params.get("test_size", 0.2)
        default_random_state = split_params.get(
            "random_state", common_params.get("seed", 42)
        )
        default_stratify = split_params.get("stratify", True)
    except Exception as e:
        print(f"Warning: Failed to load params.yaml: {e}, using hardcoded defaults")
        default_data_dir = "data/CICIDS2017_improved"
        default_output_dir = "data/CICIDS2017_improved"
        default_test_size = 0.2
        default_random_state = 42
        default_stratify = True

    parser = argparse.ArgumentParser(
        description="Split CICIDS2017_improved dataset into train/test"
    )
    parser.add_argument(
        "--data_dir", type=str, default=default_data_dir, help="Input data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=default_output_dir, help="Output directory"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=default_test_size,
        help="Test set size ratio (default: 0.2)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=default_random_state,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no_stratify", action="store_true", help="Disable stratified sampling"
    )

    args = parser.parse_args()

    args.data_dir = os.path.join(root_dir, args.data_dir)
    args.output_dir = os.path.join(root_dir, args.output_dir)

    # YAMLからstratify設定を読み込む（コマンドライン引数で上書き可能）
    stratify_by_label = default_stratify if not args.no_stratify else False

    train_df, test_df = load_and_split_cicids2017(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_by_label=stratify_by_label,
    )
