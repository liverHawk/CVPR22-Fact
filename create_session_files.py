#!/usr/bin/env python3
"""
セッションファイルを自動生成するスクリプト

Few-Shot Class-Incremental Learning用のセッションファイルを生成します。
ベースセッションと新規セッションにデータを分割します。
"""

import argparse
import os
import random
import yaml
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def set_seed(seed: int):
    """乱数シードを設定"""
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(csv_path: str, label_column: str = 'Label') -> Tuple[pl.DataFrame, List]:
    """
    データセットを読み込む
    
    Args:
        csv_path: CSVファイルのパス
        label_column: ラベル列の名前
    
    Returns:
        (データフレーム, クラスリスト)
    """
    print(f"データセットを読み込み中: {csv_path}")
    df = pl.read_csv(csv_path)
    
    if label_column not in df.columns:
        raise ValueError(f"ラベル列 '{label_column}' が見つかりません。利用可能な列: {df.columns}")
    
    # クラスを取得
    unique_classes = sorted(df[label_column].unique().to_list())
    print(f"総クラス数: {len(unique_classes)}")
    print(f"クラス: {unique_classes}")
    
    return df, unique_classes


def split_classes(
    classes: List,
    base_class: int,
    num_classes: int,
    way: int,
    base_labels: Optional[List] = None
) -> Tuple[List, List[List]]:
    """
    クラスをベースクラスと新規クラスに分割
    
    Args:
        classes: 全クラスのリスト
        base_class: ベースクラス数（base_labelsが指定されていない場合に使用）
        num_classes: 総クラス数
        way: 各新規セッションのクラス数
        base_labels: ベースセッションに使用するラベルのリスト（指定時はこのリストを使用）
    
    Returns:
        (ベースクラスリスト, [新規セッション1のクラスリスト, 新規セッション2のクラスリスト, ...])
    """
    if base_labels is not None:
        # ベースラベルが指定されている場合
        base_classes = sorted(base_labels)
        # ベースクラス以外のクラスを新規クラスとして使用
        remaining_classes = [c for c in classes if c not in base_classes]
        new_classes = remaining_classes[:num_classes - len(base_classes)]
    else:
        # 自動分割
        if len(classes) < num_classes:
            raise ValueError(f"クラス数が不足しています: {len(classes)} < {num_classes}")
        
        # ベースクラス
        base_classes = classes[:base_class]
        
        # 新規クラス
        new_classes = classes[base_class:num_classes]
    
    # 新規クラスをセッションごとに分割
    new_sessions = []
    for i in range(0, len(new_classes), way):
        session_classes = new_classes[i:i+way]
        if len(session_classes) == way:
            new_sessions.append(session_classes)
    
    print(f"ベースクラス数: {len(base_classes)}")
    print(f"ベースクラス: {base_classes}")
    print(f"新規セッション数: {len(new_sessions)}")
    for i, session_classes in enumerate(new_sessions, 1):
        print(f"  セッション {i}: {session_classes}")
    
    return base_classes, new_sessions


def create_base_session_file(
    df: pl.DataFrame,
    base_classes: List,
    label_column: str,
    output_path: str,
    use_data_index: bool = True
):
    """
    ベースセッションファイルを作成（CIFAR100形式: session_1.txt）
    
    Args:
        df: データフレーム
        base_classes: ベースクラスのリスト
        label_column: ラベル列の名前
        output_path: 出力ファイルパス
        use_data_index: Trueの場合はデータインデックス、Falseの場合はクラスインデックス
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if use_data_index:
        # データインデックスを保存（CIFAR100形式）
        # polarsではインデックスがないので、with_row_index()で行番号を追加
        base_data = df.filter(pl.col(label_column).is_in(base_classes)).with_row_index("row_index")
        indices = base_data["row_index"].to_list()
        with open(output_path, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")
        print(f"ベースセッションファイルを作成: {output_path} (データインデックス: {len(indices)}件)")
    else:
        # クラスインデックスを保存
        with open(output_path, 'w') as f:
            for cls in base_classes:
                f.write(f"{cls}\n")
        print(f"ベースセッションファイルを作成: {output_path} (クラスインデックス: {len(base_classes)}件)")


def create_new_session_file(
    df: pl.DataFrame,
    session_classes: List,
    label_column: str,
    output_path: str,
    shot: int,
    seed: int = 42
):
    """
    新規セッションファイルを作成（Few-Shot用）
    
    Args:
        df: データフレーム
        session_classes: このセッションのクラスリスト
        label_column: ラベル列の名前
        output_path: 出力ファイルパス
        shot: 各クラスのショット数
        seed: 乱数シード
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 行インデックスを追加
    df_with_index = df.with_row_index("row_index")
    
    # 各クラスからshot個のサンプルをランダムに選択
    selected_indices = []
    rng = np.random.RandomState(seed)
    
    for cls in session_classes:
        class_data = df_with_index.filter(pl.col(label_column) == cls)
        class_indices = class_data["row_index"].to_list()
        
        if len(class_indices) < shot:
            print(f"警告: クラス {cls} のサンプル数 ({len(class_indices)}) が shot ({shot}) より少ないです。")
            selected = class_indices
        else:
            selected = rng.choice(class_indices, size=shot, replace=False).tolist()
        selected_indices.extend(selected)
    
    # ファイルに保存
    with open(output_path, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    
    print(f"新規セッションファイルを作成: {output_path} (データインデックス: {len(selected_indices)}件)")


def load_params_yaml(yaml_path: str = 'params.yaml') -> dict:
    """params.yamlから設定を読み込む"""
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found. Using command-line arguments only.")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params.get('create_sessions', {}) if params else {}
    except Exception as e:
        print(f"Warning: Failed to load {yaml_path}: {e}. Using command-line arguments only.")
        return {}


def main():
    # params.yamlからデフォルト値を読み込む
    yaml_params = load_params_yaml()
    
    parser = argparse.ArgumentParser(
        description='Few-Shot Class-Incremental Learning用のセッションファイルを生成',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train-csv',
        type=str,
        default=yaml_params.get('train_csv', None),
        help='訓練データのCSVファイルパス'
    )
    parser.add_argument(
        '--label-column',
        type=str,
        default=yaml_params.get('label_column', 'Label'),
        help='ラベル列の名前'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=yaml_params.get('output_dir', 'data/index_list'),
        help='セッションファイルの出力ディレクトリ'
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default=yaml_params.get('dataset_name', None),
        help='データセット名（例: CICIDS2017_improved）'
    )
    parser.add_argument(
        '--base-class',
        type=int,
        default=yaml_params.get('base_class', None),
        help='ベースクラス数（base-labelsが指定されていない場合に使用）'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=yaml_params.get('num_classes', None),
        help='総クラス数'
    )
    parser.add_argument(
        '--way',
        type=int,
        default=yaml_params.get('way', None),
        help='各新規セッションのクラス数'
    )
    parser.add_argument(
        '--shot',
        type=int,
        default=yaml_params.get('shot', 5),
        help='各クラスのショット数（Few-Shot用）'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=yaml_params.get('seed', 42),
        help='乱数シード'
    )
    parser.add_argument(
        '--base-use-class-index',
        action='store_true',
        default=yaml_params.get('base_use_class_index', False),
        help='ベースセッションファイルにクラスインデックスを使用（デフォルト: データインデックス）'
    )
    # base_labelsの処理（YAMLのnullはNoneになる）
    base_labels_yaml = yaml_params.get('base_labels')
    base_labels_default = base_labels_yaml if base_labels_yaml is not None else None
    
    parser.add_argument(
        '--base-labels',
        nargs='+',
        default=base_labels_default,
        help='ベースセッションに使用するラベルのリスト（指定時はこのラベルのみ使用、文字列または数値）'
    )
    parser.add_argument(
        '--params-yaml',
        type=str,
        default='params.yaml',
        help='params.yamlファイルのパス'
    )
    
    args = parser.parse_args()
    
    # 必須パラメータのチェック
    if args.train_csv is None:
        raise ValueError("--train-csv または params.yaml の create_sessions.train_csv を指定してください")
    if args.dataset_name is None:
        raise ValueError("--dataset-name または params.yaml の create_sessions.dataset_name を指定してください")
    if args.base_labels is None and args.base_class is None:
        raise ValueError("--base-class または --base-labels を指定してください（params.yamlでも可）")
    if args.num_classes is None:
        raise ValueError("--num-classes または params.yaml の create_sessions.num_classes を指定してください")
    if args.way is None:
        raise ValueError("--way または params.yaml の create_sessions.way を指定してください")
    
    # 乱数シードを設定
    set_seed(args.seed)
    
    # データセットを読み込む
    df, classes = load_dataset(args.train_csv, args.label_column)
    
    # クラスを分割
    base_classes, new_sessions = split_classes(
        classes, args.base_class, args.num_classes, args.way, args.base_labels
    )
    
    # 出力ディレクトリ（CIFAR100形式: すべて同じディレクトリに配置）
    output_base_dir = Path(args.output_dir) / args.dataset_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースセッションファイルを作成（session_1.txt = ベースセッション）
    # data_utils.pyでは session_0 + 1 = session_1.txt を読み込む
    base_session_path = output_base_dir / 'session_1.txt'
    create_base_session_file(
        df, base_classes, args.label_column,
        str(base_session_path), use_data_index=not args.base_use_class_index
    )
    
    # 新規セッションファイルを作成（session_2.txt, session_3.txt, ...）
    # data_utils.pyでは session + 1 を読み込む（session=1のとき session_2.txt）
    for i, session_classes in enumerate(new_sessions, 1):
        session_path = output_base_dir / f'session_{i + 1}.txt'
        create_new_session_file(
            df, session_classes, args.label_column,
            str(session_path), args.shot, args.seed + i
        )
    
    print(f"\nセッションファイルの生成が完了しました: {output_base_dir}")
    print(f"  ベースセッション: session_1.txt")
    print(f"  新規セッション: session_2.txt ~ session_{len(new_sessions) + 1}.txt ({len(new_sessions)}セッション)")


if __name__ == '__main__':
    main()
