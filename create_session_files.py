#!/usr/bin/env python3
"""
セッションファイルを自動生成するスクリプト

Few-Shot Class-Incremental Learning用のセッションファイルを生成します。
ベースセッションと新規セッションにデータを分割します。
"""

import argparse
import gc
import logging
import os
import random

import coloredlogs
import numpy as np
import polars as pl
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional


logger = logging.getLogger(__name__)
coloredlogs.install(level="INFO", logger=logger)


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
    logger.info(f"データセットを読み込み中: {csv_path}")
    if os.path.isdir(csv_path):
        csv_files = sorted(Path(csv_path).glob("*.csv"))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        dfs = []
        for csv_file in csv_files:
            dfs.append(pl.read_csv(csv_file))
        df = pl.concat(dfs)
        del dfs
        gc.collect()
    else:
        df = pl.read_csv(csv_path)
    
    if label_column not in df.columns:
        raise ValueError(f"ラベル列 '{label_column}' が見つかりません。利用可能な列: {df.columns}")
    
    # クラスを取得
    unique_classes = sorted(df[label_column].unique().to_list())
    logger.info(f"総クラス数: {len(unique_classes)}")
    # logger.info(f"クラス: {unique_classes}")
    
    return df, unique_classes


def sample_per_label_cap(
    df: pl.DataFrame,
    label_column: str,
    max_per_label: int,
    seed: int = 42
) -> pl.DataFrame:
    """
    処理前に、ラベルごとのデータ数が上限を超えている場合はその上限でサンプリングする。
    
    Args:
        df: データフレーム
        label_column: ラベル列の名前
        max_per_label: ラベルあたりの最大サンプル数（超えた分はランダムに捨てる）
        seed: 乱数シード
    
    Returns:
        サンプリング後のデータフレーム
    """
    labels = df[label_column].unique().sort().to_list()
    dfs = []
    for i, label in enumerate(labels):
        subset = df.filter(pl.col(label_column) == label)
        n = len(subset)
        if n <= max_per_label:
            dfs.append(subset)
            continue
        # 上限を超えているので max_per_label 件にランダムサンプリング（ラベルごとに別シードで再現性を確保）
        sampled = subset.sample(n=max_per_label, seed=seed + i)
        dfs.append(sampled)
        logger.info(f"ラベル {label}: {n}件 → {max_per_label}件にサンプリング")
    out = pl.concat(dfs)
    del dfs
    gc.collect()
    logger.info(f"ラベル別上限サンプリング後: 総件数 {len(df)} → {len(out)}")
    return out


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
        new_classes = [c for c in classes if c not in base_classes]
        logger.info(f"new classes: {new_classes}")
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
    
    logger.info(f"ベースクラス数: {len(base_classes)}")
    logger.info(f"ベースクラス: {base_classes}")
    logger.info(f"新規セッション数: {len(new_sessions)}")
    for i, session_classes in enumerate(new_sessions, 1):
        logger.info(f"  セッション {i:2d}: {session_classes}")
    
    return base_classes, new_sessions


def build_label_to_indices(
    df: pl.DataFrame,
    label_column: str,
    index_col: str = "_original_row_index",
) -> Dict[str, List[int]]:
    """
    ラベル→インデックスリストのマッピングを1回のスキャンで構築する。
    
    DataFrame の繰り返しフィルタを避け、セッションファイル生成を高速化するために使用。
    
    Args:
        df: データフレーム（index_col列を含むこと）
        label_column: ラベル列の名前
        index_col: インデックス列の名前
    
    Returns:
        {ラベル: [インデックス, ...], ...} の辞書
    """
    logger.info("ラベル→インデックスのマッピングを構築中...")
    label_to_indices: Dict[str, List[int]] = {}
    for group_df in df.partition_by(label_column):
        label = group_df[label_column][0]
        label_to_indices[label] = group_df[index_col].to_list()
    logger.info("マッピング構築完了: %d ラベル", len(label_to_indices))
    return label_to_indices


def create_base_session_file(
    label_to_indices: Dict[str, List[int]],
    base_classes: List,
    output_path: str,
    use_data_index: bool = True
):
    """
    ベースセッションファイルを作成（CIFAR100形式: session_1.txt）
    
    Args:
        label_to_indices: ラベル→インデックスリストのマッピング
        base_classes: ベースクラスのリスト
        output_path: 出力ファイルパス
        use_data_index: Trueの場合はデータインデックス、Falseの場合はクラスインデックス
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if use_data_index:
        # データインデックスを保存（CIFAR100形式）
        indices = []
        for cls in base_classes:
            if cls in label_to_indices:
                indices.extend(label_to_indices[cls])
            else:
                logger.warning("ベースクラス '%s' のデータが見つかりません", cls)
        with open(output_path, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")
        logger.info(
            "ベースセッションファイルを作成: %s (データインデックス: %d件)",
            output_path,
            len(indices),
        )
    else:
        # クラスインデックスを保存
        with open(output_path, 'w') as f:
            for cls in base_classes:
                f.write(f"{cls}\n")
        logger.info(
            "ベースセッションファイルを作成: %s (クラスインデックス: %d件)",
            output_path,
            len(base_classes),
        )


def create_new_session_file(
    label_to_indices: Dict[str, List[int]],
    session_classes: List,
    output_path: str,
    shot: int,
    seed: int = 42
):
    """
    新規セッションファイルを作成（Few-Shot用）
    
    Args:
        label_to_indices: ラベル→インデックスリストのマッピング
        session_classes: このセッションのクラスリスト
        output_path: 出力ファイルパス
        shot: 各クラスのショット数
        seed: 乱数シード
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 各クラスからshot個のサンプルをランダムに選択
    selected_indices = []
    selected_labels = []
    rng = np.random.RandomState(seed)
    
    for cls in session_classes:
        class_indices = label_to_indices.get(cls, [])
        
        if len(class_indices) < shot:
            logger.warning(f"クラス {cls} のサンプル数 ({len(class_indices)}) が shot ({shot}) より少ないです。")
            selected = class_indices
        else:
            selected = rng.choice(class_indices, size=shot, replace=False).tolist()
        selected_indices.extend(selected)
        selected_labels.extend([cls] * len(selected))
    
    # ファイルに保存
    with open(output_path, 'w') as f:
        for idx in selected_indices:
            f.write(f"{idx}\n")
    
    # 検証: 選択されたインデックスがすべて期待されるクラスに属しているか
    unique_actual_labels = set(selected_labels)
    expected_labels = set(session_classes)
    
    if unique_actual_labels != expected_labels:
        logger.warning("生成されたセッションファイルに予期しないラベルが含まれています")
        logger.warning(f"  期待されるラベル: {sorted(expected_labels)}")
        logger.warning(f"  実際のラベル: {sorted(unique_actual_labels)}")
        raise ValueError(
            f"セッションファイル {output_path} の検証に失敗しました。"
            f"期待されるラベル {sorted(expected_labels)} と実際のラベル {sorted(unique_actual_labels)} が一致しません。"
        )
    
    logger.info(f"新規セッションファイルを作成: {output_path} (データインデックス: {len(selected_indices)}件)")
    logger.info(f"  ✓ 検証完了: すべてのサンプルが期待されるクラス {sorted(session_classes)} に属しています")


def load_params_yaml(yaml_path: str = 'params.yaml') -> dict:
    """params.yamlから設定を読み込む"""
    if not os.path.exists(yaml_path):
        logger.warning(f"{yaml_path} not found. Using command-line arguments only.")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params.get('create_sessions', {}) if params else {}
    except Exception as e:
        logger.warning(f"Failed to load {yaml_path}: {e}. Using command-line arguments only.")
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
        default=None,
        help='訓練データのCSVファイルパス（指定しない場合はdata/{dataset_name}/trainを自動使用）'
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
        '--max-samples-per-label',
        type=int,
        default=yaml_params.get('max_samples_per_label'),
        metavar='N',
        help='ラベルごとのデータ数上限（指定時は処理前にこの件数を超えるラベルをサンプリングする。未指定は無制限）'
    )
    parser.add_argument(
        '--params-yaml',
        type=str,
        default='params.yaml',
        help='params.yamlファイルのパス'
    )
    
    args = parser.parse_args()
    
    # 必須パラメータのチェック
    if args.dataset_name is None:
        raise ValueError("--dataset-name または params.yaml の create_sessions.dataset_name を指定してください")
    
    # train_csvが指定されていない場合、dataset_nameから自動的に構築
    if args.train_csv is None:
        args.train_csv = f"data/{args.dataset_name}/train"
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
    # logger.info(f"classes: {classes}")
    
    # サンプリング前にフルデータでの行インデックスを記録
    # セッションファイルにはこの元のインデックスを書き出す（CICIDS2017データセットクラスがフルデータをロードするため）
    df = df.with_row_index("_original_row_index")
    
    # 処理前にラベルごとのデータ数が上限を超えていればサンプリング
    if args.max_samples_per_label is not None:
        df = sample_per_label_cap(
            df, args.label_column, args.max_samples_per_label, args.seed
        )
        classes = sorted(df[args.label_column].unique().to_list())
    
    # クラスを分割
    base_classes, new_sessions = split_classes(
        classes, args.base_class, args.num_classes, args.way, args.base_labels
    )
    
    # ラベル→インデックスのマッピングを1回だけ構築し、DataFrameを即解放
    # これにより各セッションでの繰り返しフィルタを回避し高速化
    index_col = "_original_row_index" if "_original_row_index" in df.columns else None
    if index_col is None:
        df = df.with_row_index("_original_row_index")
        index_col = "_original_row_index"
    label_to_indices = build_label_to_indices(df, args.label_column, index_col)
    del df
    gc.collect()
    
    # 出力ディレクトリ（CIFAR100形式: すべて同じディレクトリに配置）
    output_base_dir = Path(args.output_dir) / args.dataset_name
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # ベースセッションファイルを作成（session_1.txt = ベースセッション）
    # data_utils.pyでは session_0 + 1 = session_1.txt を読み込む
    base_session_path = output_base_dir / 'session_1.txt'
    create_base_session_file(
        label_to_indices, base_classes,
        str(base_session_path), use_data_index=not args.base_use_class_index
    )
    
    # 新規セッションファイルを作成（session_2.txt, session_3.txt, ...）
    # data_utils.pyでは session + 1 を読み込む（session=1のとき session_2.txt）
    for i, session_classes in enumerate(new_sessions, 1):
        session_path = output_base_dir / f'session_{i + 1}.txt'
        create_new_session_file(
            label_to_indices, session_classes,
            str(session_path), args.shot, args.seed + i
        )
    
    # マッピングを解放
    del label_to_indices
    gc.collect()
    
    logger.info(f"セッションファイルの生成が完了しました: {output_base_dir}")
    logger.info("  ベースセッション: session_1.txt")
    logger.info(f"  新規セッション: session_2.txt ~ session_{len(new_sessions) + 1}.txt ({len(new_sessions)}セッション)")


if __name__ == '__main__':
    main()
