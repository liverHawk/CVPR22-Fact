"""
CICIDS2017_flow_improvedデータセットをラベルごとにtrain/testに分割するスクリプト

指定ディレクトリ内のCSVファイルを読み込み、ラベルごとに同じ割合で
train.csvとtest.csvに分割します。
"""

import polars as pl
from pathlib import Path
from typing import Optional
import argparse
from sklearn.model_selection import train_test_split
import yaml
import os


def relabel_df(df: pl.DataFrame, label_column: str) -> pl.DataFrame:
    label_unique = df[label_column].unique().to_list()
    for label in label_unique:
        if "Attempted" in label:
            df = df.with_columns(pl.col(label_column).replace(label, "BENIGN"))
        if "DoS" in label and not "DDoS" in label:
            df = df.with_columns(pl.col(label_column).replace(label, "DoS"))
        if "Web Attack" in label:
            df = df.with_columns(pl.col(label_column).replace(label, "Web Attack"))
    return df


def split_dataset_by_label(
    input_dir: str,
    output_dir: Optional[str] = None,
    label_column: str = 'Label',
    test_ratio: float = 0.2,
    random_state: int = 42,
    merge_all_files: bool = True,
    split_by_label: bool = False,
    chunk_size: Optional[int] = None
):
    """
    ラベルごとに同じ割合でtrain/testに分割
    
    Args:
        input_dir: 入力ディレクトリのパス
        output_dir: 出力ディレクトリのパス（Noneの場合はinput_dirと同じ）
        label_column: ラベル列の名前
        test_ratio: テストデータの割合（デフォルト: 0.2 = 20%）
        random_state: ランダムシード
        merge_all_files: Trueの場合、すべてのCSVファイルをマージしてから分割
        split_by_label: Trueの場合、ラベルごとに別ファイルに保存
        chunk_size: 指定した場合、この行数ごとにファイルを分割（Noneの場合は1ファイル）
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"入力ディレクトリが見つかりません: {input_dir}")
    
    if output_dir is None:
        output_path = input_path
    else:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # CSVファイルを検索
    csv_files = list(input_path.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(f"CSVファイルが見つかりません: {input_dir}")
    
    print(f"見つかったCSVファイル数: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # すべてのCSVファイルを読み込んでマージ
    dfs = []
    for csv_file in csv_files:
        print(f"\n読み込み中: {csv_file.name}")
        df = pl.read_csv(csv_file)
        print(f"  行数: {len(df)}, 列数: {len(df.columns)}")
        
        # ラベル列の存在確認
        if label_column not in df.columns:
            print(f"  警告: ラベル列 '{label_column}' が見つかりません。スキップします。")
            print(f"  利用可能な列: {df.columns.tolist()}")
            continue
        
        dfs.append(df)
    
    if len(dfs) == 0:
        raise ValueError("読み込めるCSVファイルがありませんでした。")
    
    # データフレームをマージ
    print(f"\nデータフレームをマージ中...")
    merged_df: pl.DataFrame = pl.concat(dfs)
    print(f"合計行数: {len(merged_df)}")

    merged_df = relabel_df(merged_df, label_column)
    
    # ラベルの分布を確認
    label_counts = merged_df[label_column].value_counts().sort("count", descending=True)
    print(f"\nラベルの分布:")
    print(label_counts)
    print(f"\nユニークなラベル数: {len(label_counts)}")
    
    # ラベルごとにtrain/testに分割
    print(f"\nラベルごとに分割中（test_ratio={test_ratio}）...")

    # check if the number of samples is enough for each label
    if any(label_counts["count"] < 2):
        raise ValueError("ラベルのサンプル数が少なすぎます。2行以上のラベルが必要です。")
    
    train_df, test_df = train_test_split(
        merged_df,
        test_size=test_ratio,
        random_state=random_state,
        stratify=merged_df[label_column],
        shuffle=True
    )
    
    # 保存
    print(f"\n保存中...")
    
    if split_by_label:
        # ラベルごとにファイルを分けて保存
        train_dir = output_path / "train"
        test_dir = output_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 訓練データをラベルごとに保存
        for label in train_df[label_column].unique():
            label_train = train_df[train_df[label_column] == label].clone()
            # ファイル名に使えない文字を置換
            safe_label = str(label).replace('/', '_').replace('\\', '_').replace(':', '_')
            train_file = train_dir / f"train_{safe_label}.csv"
            label_train.write_csv(train_file, index=False)
            print(f"  train_{safe_label}.csv: {len(label_train)}行 -> {train_file}")
        
        # テストデータをラベルごとに保存
        for label in test_df[label_column].unique():
            label_test = test_df[test_df[label_column] == label].clone()
            safe_label = str(label).replace('/', '_').replace('\\', '_').replace(':', '_')
            test_file = test_dir / f"test_{safe_label}.csv"
            label_test.write_csv(test_file, index=False)
            print(f"  test_{safe_label}.csv: {len(label_test)}行 -> {test_file}")
    
    elif chunk_size is not None and chunk_size > 0:
        # チャンクサイズで分割して保存
        train_dir = output_path / "train"
        test_dir = output_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # 訓練データをチャンクに分割
        n_train_chunks = (len(train_df) + chunk_size - 1) // chunk_size
        for i in range(n_train_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(train_df))
            chunk_df = train_df[start_idx:end_idx].clone()
            train_file = train_dir / f"train_chunk_{i+1:04d}.csv"
            chunk_df.write_csv(train_file)
            print(f"  train_chunk_{i+1:04d}.csv: {len(chunk_df)}行 -> {train_file}")
        
        # テストデータをチャンクに分割
        n_test_chunks = (len(test_df) + chunk_size - 1) // chunk_size
        for i in range(n_test_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(test_df))
            chunk_df = test_df[start_idx:end_idx].clone()
            test_file = test_dir / f"test_chunk_{i+1:04d}.csv"
            chunk_df.write_csv(test_file)
            print(f"  test_chunk_{i+1:04d}.csv: {len(chunk_df)}行 -> {test_file}")
    
    else:
        # train/とtest/ディレクトリに保存
        train_dir = output_path / "train"
        test_dir = output_path / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = train_dir / "train.csv"
        test_path = test_dir / "test.csv"
        
        train_df.write_csv(train_path)
        print(f"  train.csv: {len(train_df)}行 -> {train_path}")
        
        test_df.write_csv(test_path)
        print(f"  test.csv: {len(test_df)}行 -> {test_path}")
    
    # 最終的なラベル分布を表示
    print(f"\n=== 分割後のラベル分布 ===")
    print(f"\n訓練データ:")
    print(train_df[label_column].value_counts().sort("count", descending=True))
    print(f"\nテストデータ:")
    print(test_df[label_column].value_counts().sort("count", descending=True))
    
    print(f"\n完了しました！")
    print(f"  訓練データ: {len(train_df)}行 ({len(train_df)/len(merged_df)*100:.1f}%)")
    print(f"  テストデータ: {len(test_df)}行 ({len(test_df)/len(merged_df)*100:.1f}%)")


def load_params_yaml(yaml_path: str = 'params.yaml') -> dict:
    """params.yamlから設定を読み込む"""
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found. Using command-line arguments only.")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params if params else {}
    except Exception as e:
        print(f"Warning: Failed to load {yaml_path}: {e}. Using command-line arguments only.")
        return {}


def main():
    # params.yamlからデフォルト値を読み込む
    yaml_params = load_params_yaml()
    dataset_name = yaml_params.get('dataset_name', 'CICIDS2017_flow_improved')
    default_output_dir = f"./data/{dataset_name}"
    
    parser = argparse.ArgumentParser(
        description="CICIDS2017_flow_improvedデータセットをラベルごとにtrain/testに分割"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/home/toshi/Documents/dataset/project/cleaned/CICIDS2017_flow_improved",
        help="入力ディレクトリのパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help="出力ディレクトリのパス（指定しない場合はparams.yamlのdataset_nameから自動生成）"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="ラベル列の名前（デフォルト: 'Label'）"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="テストデータの割合（デフォルト: 0.2 = 20%）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="ランダムシード（デフォルト: 42）"
    )
    parser.add_argument(
        "-s",
        "--split-by-label",
        action="store_true",
        help="ラベルごとに別ファイルに保存（train/とtest/ディレクトリに保存）"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="指定した行数ごとにファイルを分割（例: 100000）"
    )
    
    args = parser.parse_args()
    
    split_dataset_by_label(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_column=args.label_column,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        split_by_label=args.split_by_label,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
