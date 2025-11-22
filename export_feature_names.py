#!/usr/bin/env python3
"""
CICIDS2017_improvedデータセットの特徴量名をファイルに出力するスクリプト
"""

import os
import argparse
import json
from dataloader.cicids2017.cicids2017 import CICIDS2017_improved


def export_feature_names(data_dir='data/', output_file='data/CICIDS2017_improved/feature_names.json', output_txt='data/CICIDS2017_improved/feature_names.txt'):
    """
    特徴量名を取得してファイルに出力
    
    Args:
        data_dir: データディレクトリ
        output_file: JSON形式の出力ファイルパス
        output_txt: テキスト形式の出力ファイルパス
    """
    print(f"Loading dataset from {data_dir}...")
    
    # データセットを初期化（特徴量名を取得するため）
    # train=Trueで訓練データを読み込む
    dataset = CICIDS2017_improved(root=data_dir, train=True)
    
    # 特徴量名を取得
    # データローダー内で前処理された後の特徴量名を取得する必要がある
    # 実際のCSVファイルから読み込んで特徴量名を取得
    train_file = os.path.join(data_dir, 'CICIDS2017_improved', 'train.csv')
    
    if not os.path.exists(train_file):
        print(f"Warning: {train_file} not found. Using processed feature count.")
        # データセットから特徴量数を取得
        num_features = dataset.data.shape[1] if hasattr(dataset, 'data') else None
        if num_features:
            feature_names = [f"feature_{i}" for i in range(num_features)]
        else:
            raise ValueError("Could not determine feature names")
    else:
        import pandas as pd
        print(f"Reading feature names from {train_file}...")
        df = pd.read_csv(train_file, nrows=1)  # ヘッダーのみ読み込み
        
        # ラベル列を除外
        if 'Label' in df.columns:
            label_col = 'Label'
        else:
            label_col = df.columns[-1]
        
        feature_cols = [col for col in df.columns if col != label_col]
        
        # データローダーと同じ前処理を適用
        delete_cols = dataset._get_delete_columns()
        existing_delete_cols = [col for col in delete_cols if col in feature_cols]
        feature_cols = [col for col in feature_cols if col not in existing_delete_cols]
        
        # 'Attempted Category'列も削除（データローダーで削除される）
        if 'Attempted Category' in feature_cols:
            feature_cols.remove('Attempted Category')
        
        # カラム名のリネームを適用
        rename_dict = dataset._get_column_rename_dict()
        feature_names = [rename_dict.get(col, col) for col in feature_cols]
    
    print(f"\nFound {len(feature_names)} features")
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    
    # JSON形式で保存
    feature_info = {
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "feature_indices": {name: idx for idx, name in enumerate(feature_names)}
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, indent=2, ensure_ascii=False)
    
    print(f"Saved feature names to {output_file}")
    
    # テキスト形式で保存（1行に1つの特徴量名）
    with open(output_txt, 'w', encoding='utf-8') as f:
        for name in feature_names:
            f.write(f"{name}\n")
    
    print(f"Saved feature names to {output_txt}")
    
    # コンソールにも表示
    print("\nFeature names:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:3d}. {name}")
    
    return feature_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export CICIDS2017_improved feature names to file')
    parser.add_argument('--data_dir', type=str, default='data/',
                        help='Data directory (default: data/)')
    parser.add_argument('--output_json', type=str, default='data/CICIDS2017_improved/feature_names.json',
                        help='Output JSON file path (default: data/CICIDS2017_improved/feature_names.json)')
    parser.add_argument('--output_txt', type=str, default='data/CICIDS2017_improved/feature_names.txt',
                        help='Output text file path (default: data/CICIDS2017_improved/feature_names.txt)')
    
    args = parser.parse_args()
    
    export_feature_names(
        data_dir=args.data_dir,
        output_file=args.output_json,
        output_txt=args.output_txt
    )

