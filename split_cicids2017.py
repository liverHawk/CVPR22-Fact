#!/usr/bin/env python3
"""
CICIDS2017_improvedデータセットをtrain/testに分割するスクリプト
指定されたGitHubリポジトリのブランチ（cicids2017）の前処理パターンを参照
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_split_cicids2017(data_dir='data/CICIDS2017_improved', 
                               output_dir='data/CICIDS2017_improved',
                               test_size=0.2,
                               random_state=42,
                               stratify_by_label=True):
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
    csv_files = ['monday.csv', 'tuesday.csv', 'wednesday.csv', 'thursday.csv', 'friday.csv']
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
    if 'Label' in combined_df.columns:
        label_col = 'Label'
    else:
        # 最後の列をラベルとして使用
        label_col = combined_df.columns[-1]
        print(f"Warning: 'Label' column not found, using '{label_col}' as label column")
    
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
    print(f"\nSplitting data into train ({1-test_size:.0%}) and test ({test_size:.0%})...")
    
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
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    print(f"\nSaving train data to {train_path}...")
    train_df.to_csv(train_path, index=False)
    print(f"Saved {len(train_df)} rows")
    
    print(f"Saving test data to {test_path}...")
    test_df.to_csv(test_path, index=False)
    print(f"Saved {len(test_df)} rows")
    
    print("\nDone!")
    return train_df, test_df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Split CICIDS2017_improved dataset into train/test')
    parser.add_argument('--data_dir', type=str, default='data/CICIDS2017_improved',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='data/CICIDS2017_improved',
                        help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no_stratify', action='store_true',
                        help='Disable stratified sampling')
    
    args = parser.parse_args()
    
    train_df, test_df = load_and_split_cicids2017(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_by_label=not args.no_stratify
    )

