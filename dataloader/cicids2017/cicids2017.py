"""
CICIDS2017_improvedデータセット用のデータローダー

表形式データ（CSV）を読み込み、Few-Shot Class-Incremental Learning用に
セッションごとにデータを分割します。
"""

import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import polars as pl
from pathlib import Path
from typing import Optional, List, Union


class CICIDS2017(Dataset):
    """
    CICIDS2017_improvedデータセット
    
    Args:
        root: データセットのルートディレクトリ
        train: Trueの場合は訓練データ、Falseの場合はテストデータ
        index: クラスインデックスのリスト（base_sess=Trueの場合）またはデータインデックスのリスト
        index_path: セッションファイルのパス（base_sess=Falseの場合）
        base_sess: Trueの場合はベースセッション、Falseの場合は新規セッション
        label_column: ラベル列の名前（デフォルト: 'Label'）
        normalize_method: 正規化方法（'standard', 'minmax', 'moving_minmax'）
    """
    
    # クラス変数としてキャッシュを定義
    _cached_train_features = None
    _cached_train_labels = None
    _cached_test_features = None
    _cached_test_labels = None
    _normalization_stats = None
    _label_to_idx = None
    _idx_to_label = None
    _cache_root = None
    _cache_label_column = None
    _cache_normalize_method = None
    _feature_columns = None
    
    def __init__(
        self,
        root: str = 'data/',
        train: bool = True,
        index: Optional[Union[List[int], np.ndarray]] = None,
        index_path: Optional[str] = None,
        base_sess: Optional[bool] = None,
        label_column: str = 'Label',
        normalize_method: str = 'standard',
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.label_column = label_column
        self.normalize_method = normalize_method
        
        # キャッシュが存在し、同じroot、label_column、normalize_methodの場合は再利用
        cache_valid = (
            self._cached_train_features is not None and
            self._cache_root == self.root and
            self._cache_label_column == label_column and
            self._cache_normalize_method == normalize_method
        )
        
        if not cache_valid:
            # キャッシュが存在しない、または異なる設定の場合は読み込む
            self._load_and_cache_data(self.root, label_column, normalize_method)
        
        # キャッシュからデータを取得
        if train:
            features = self._cached_train_features
            labels = self._cached_train_labels
        else:
            features = self._cached_test_features
            labels = self._cached_test_labels
        
        # ラベルマッピングを設定
        self.label_to_idx = self._label_to_idx
        self.idx_to_label = self._idx_to_label
        self.num_classes = len(self._label_to_idx)
        
        # 正規化統計量を設定
        if self._normalization_stats['method'] == 'standard':
            self.mean = self._normalization_stats['mean']
            self.std = self._normalization_stats['std']
        elif self._normalization_stats['method'] in ['minmax', 'moving_minmax']:
            self.min = self._normalization_stats['min']
            self.max = self._normalization_stats['max']
            self.range = self._normalization_stats['range']
        
        # セッション選択
        if base_sess:
            # ベースセッション: クラスインデックスから選択
            if index is None:
                raise ValueError("base_sess=Trueの場合、indexを指定してください")
            self.data, self.targets = self.SelectfromClasses(features, labels, index)
        else:
            # 新規セッション: セッションファイルからデータインデックスを読み込み
            # セッションファイルのインデックスは、concatした後の全体データフレームに対するインデックス
            if index_path is not None:
                self.data, self.targets = self.SelectfromTxt(features, labels, index_path)
            elif index is not None:
                # データインデックスのリストが直接指定された場合
                index_array = np.array(index, dtype=np.int64)
                # インデックスの範囲チェック
                if len(index_array) > 0 and (index_array.max() >= len(features) or index_array.min() < 0):
                    raise ValueError(
                        f"データインデックスの範囲が不正です: "
                        f"min={index_array.min()}, max={index_array.max()}, data_len={len(features)}"
                    )
                self.data = features[index_array]
                self.targets = labels[index_array]
            else:
                # すべてのデータを使用
                self.data = features
                self.targets = labels
        
        print(f"Loaded {len(self.data)} samples, {len(np.unique(self.targets))} classes")
    
    @classmethod
    def _load_and_cache_data(cls, root: str, label_column: str, normalize_method: str):
        """
        CSVファイルを読み込み、特徴量とラベルを抽出してキャッシュに保存
        
        Args:
            root: データセットのルートディレクトリ
            label_column: ラベル列の名前
            normalize_method: 正規化方法
        """
        root = os.path.expanduser(root)
        
        # 訓練データの読み込み
        train_dir = os.path.join(root, 'train')
        if os.path.isdir(train_dir):
            csv_files = sorted(Path(train_dir).glob("*.csv"))
            if len(csv_files) == 0:
                raise FileNotFoundError(f"CSVファイルが見つかりません: {train_dir}")
            
            print(f"Loading CSV files from {train_dir}...")
            dfs = []
            for csv_path in csv_files:
                print(f"  Reading {csv_path.name}...")
                dfs.append(pl.read_csv(csv_path))
            
            if len(dfs) == 1:
                train_df = dfs[0]
            else:
                print(f"  Merging {len(dfs)} CSV files...")
                train_df = pl.concat(dfs)
        else:
            train_csv_path = os.path.join(root, 'train.csv')
            if not os.path.exists(train_csv_path):
                raise FileNotFoundError(f"CSVファイルまたはディレクトリが見つかりません: {train_dir} または {train_csv_path}")
            print(f"Loading train.csv...")
            train_df = pl.read_csv(train_csv_path)
        
        # テストデータの読み込み
        test_dir = os.path.join(root, 'test')
        if os.path.isdir(test_dir):
            csv_files = sorted(Path(test_dir).glob("*.csv"))
            if len(csv_files) == 0:
                raise FileNotFoundError(f"CSVファイルが見つかりません: {test_dir}")
            
            print(f"Loading CSV files from {test_dir}...")
            dfs = []
            for csv_path in csv_files:
                print(f"  Reading {csv_path.name}...")
                dfs.append(pl.read_csv(csv_path))
            
            if len(dfs) == 1:
                test_df = dfs[0]
            else:
                print(f"  Merging {len(dfs)} CSV files...")
                test_df = pl.concat(dfs)
        else:
            test_csv_path = os.path.join(root, 'test.csv')
            if not os.path.exists(test_csv_path):
                raise FileNotFoundError(f"CSVファイルまたはディレクトリが見つかりません: {test_dir} または {test_csv_path}")
            print(f"Loading test.csv...")
            test_df = pl.read_csv(test_csv_path)
        
        # ラベル列のチェック
        if label_column not in train_df.columns:
            raise ValueError(
                f"ラベル列 '{label_column}' が見つかりません。"
                f"利用可能な列: {train_df.columns}"
            )
        if label_column not in test_df.columns:
            raise ValueError(
                f"ラベル列 '{label_column}' が見つかりません。"
                f"利用可能な列: {test_df.columns}"
            )
        
        # 特徴量列を取得（ラベル列と非数値列を除外）
        exclude_columns = [
            label_column, 'id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
            'Timestamp', 'Attempted Category'
        ]
        # polarsのdtypeをチェック
        feature_columns = [
            col for col in train_df.columns
            if col not in exclude_columns and train_df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]
        ]
        
        # ラベルを数値インデックスに変換（訓練データとテストデータで統一）
        train_labels_unique = train_df[label_column].unique().to_list()
        test_labels_unique = test_df[label_column].unique().to_list()
        all_labels = sorted(set(train_labels_unique + test_labels_unique))
        label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}
        
        # デバッグ出力
        print(f"Found {len(all_labels)} unique labels: {all_labels[:10]}...")
        print(f"Label to index mapping (first 10): {dict(list(label_to_idx.items())[:10])}")
        
        # 訓練データの特徴量とラベルを抽出
        train_features = train_df.select(feature_columns).to_numpy().astype(np.float32)
        train_labels_str = train_df[label_column].to_numpy()
        train_labels = np.array([label_to_idx[label] for label in train_labels_str], dtype=np.int64)
        
        # テストデータの特徴量とラベルを抽出
        test_features = test_df.select(feature_columns).to_numpy().astype(np.float32)
        test_labels_str = test_df[label_column].to_numpy()
        test_labels = np.array([label_to_idx[label] for label in test_labels_str], dtype=np.int64)
        
        # NaNや無限大の値を処理
        train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
        test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 正規化統計量を計算（訓練データのみ）
        if normalize_method == 'standard':
            mean = np.mean(train_features, axis=0, keepdims=True)
            std = np.std(train_features, axis=0, keepdims=True) + 1e-6
            normalization_stats = {'mean': mean, 'std': std, 'method': 'standard'}
            # 統計量をファイルに保存
            stats_path = os.path.join(root, 'normalization_stats.npz')
            np.savez(stats_path, mean=mean, std=std, method='standard')
        elif normalize_method == 'minmax':
            min_val = np.min(train_features, axis=0, keepdims=True)
            max_val = np.max(train_features, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            normalization_stats = {'min': min_val, 'max': max_val, 'range': range_val, 'method': 'minmax'}
            # 統計量をファイルに保存
            stats_path = os.path.join(root, 'normalization_stats.npz')
            np.savez(stats_path, min=min_val, max=max_val, method='minmax')
        elif normalize_method == 'moving_minmax':
            min_val = np.min(train_features, axis=0, keepdims=True)
            max_val = np.max(train_features, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            normalization_stats = {'min': min_val, 'max': max_val, 'range': range_val, 'method': 'moving_minmax'}
            # 統計量をファイルに保存
            stats_path = os.path.join(root, 'normalization_stats.npz')
            np.savez(stats_path, min=min_val, max=max_val, method='moving_minmax')
        else:
            raise ValueError(f"Unknown normalize_method: {normalize_method}")
        
        # 正規化を適用
        if normalize_method == 'standard':
            train_features = (train_features - normalization_stats['mean']) / normalization_stats['std']
            test_features = (test_features - normalization_stats['mean']) / normalization_stats['std']
        elif normalize_method in ['minmax', 'moving_minmax']:
            train_features = (train_features - normalization_stats['min']) / normalization_stats['range']
            test_features = (test_features - normalization_stats['min']) / normalization_stats['range']
        
        # キャッシュに保存
        cls._cached_train_features = train_features
        cls._cached_train_labels = train_labels
        cls._cached_test_features = test_features
        cls._cached_test_labels = test_labels
        cls._normalization_stats = normalization_stats
        cls._label_to_idx = label_to_idx
        cls._idx_to_label = idx_to_label
        cls._cache_root = root
        cls._cache_label_column = label_column
        cls._cache_normalize_method = normalize_method
        cls._feature_columns = feature_columns
        
        print(f"Cached train data: {len(train_features)} samples, {len(np.unique(train_labels))} classes")
        print(f"Cached test data: {len(test_features)} samples, {len(np.unique(test_labels))} classes")
    
    def _compute_normalization_stats(self, features: np.ndarray):
        """正規化統計量を計算して保存"""
        if self.normalize_method == 'standard':
            self.mean = np.mean(features, axis=0, keepdims=True)
            self.std = np.std(features, axis=0, keepdims=True) + 1e-6
            # 統計量を保存
            stats_path = os.path.join(self.root, 'normalization_stats.npz')
            np.savez(stats_path, mean=self.mean, std=self.std, method='standard')
        elif self.normalize_method == 'minmax':
            self.min = np.min(features, axis=0, keepdims=True)
            self.max = np.max(features, axis=0, keepdims=True)
            # ゼロ除算を避ける
            self.range = self.max - self.min
            self.range[self.range == 0] = 1.0
            # 統計量を保存
            stats_path = os.path.join(self.root, 'normalization_stats.npz')
            np.savez(stats_path, min=self.min, max=self.max, method='minmax')
        elif self.normalize_method == 'moving_minmax':
            # Moving Min-Maxは実装が複雑なため、標準的なMin-Maxを使用
            self.min = np.min(features, axis=0, keepdims=True)
            self.max = np.max(features, axis=0, keepdims=True)
            self.range = self.max - self.min
            self.range[self.range == 0] = 1.0
            stats_path = os.path.join(self.root, 'normalization_stats.npz')
            np.savez(stats_path, min=self.min, max=self.max, method='moving_minmax')
        else:
            raise ValueError(f"Unknown normalize_method: {self.normalize_method}")
    
    def _load_normalization_stats(self):
        """保存された正規化統計量を読み込む"""
        stats_path = os.path.join(self.root, 'normalization_stats.npz')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"正規化統計量ファイルが見つかりません: {stats_path}\n"
                "先に訓練データを読み込んで統計量を計算してください。"
            )
        stats = np.load(stats_path)
        method = stats.get('method', 'standard')
        
        if method == 'standard':
            self.mean = stats['mean']
            self.std = stats['std']
        elif method in ['minmax', 'moving_minmax']:
            self.min = stats['min']
            self.max = stats['max']
            self.range = self.max - self.min
            self.range[self.range == 0] = 1.0
        else:
            raise ValueError(f"Unknown normalization method in stats: {method}")
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """特徴量を正規化"""
        if self.normalize_method == 'standard':
            return (features - self.mean) / self.std
        elif self.normalize_method in ['minmax', 'moving_minmax']:
            return (features - self.min) / self.range
        else:
            return features
    
    def SelectfromClasses(self, data: np.ndarray, targets: np.ndarray, index: Union[List[int], np.ndarray]):
        """
        クラスインデックスからデータを選択
        
        Args:
            data: 特徴量データ
            targets: ラベルデータ
            index: クラスインデックスのリスト
        
        Returns:
            (選択された特徴量, 選択されたラベル)
        """
        index = np.array(index, dtype=np.int64)
        data_tmp = []
        targets_tmp = []
        
        # デバッグ出力
        unique_targets = np.unique(targets)
        print(f"SelectfromClasses: Requested class indices: {index}")
        print(f"SelectfromClasses: Available class indices in data: {unique_targets}")
        
        for class_idx in index:
            ind_cl = np.where(targets == class_idx)[0]
            if len(ind_cl) == 0:
                print(f"Warning: No samples found for class index {class_idx}")
                continue
            print(f"Found {len(ind_cl)} samples for class index {class_idx}")
            if len(data_tmp) == 0:
                data_tmp = data[ind_cl]
                targets_tmp = targets[ind_cl]
            else:
                data_tmp = np.vstack((data_tmp, data[ind_cl]))
                targets_tmp = np.hstack((targets_tmp, targets[ind_cl]))
        
        if len(data_tmp) == 0:
            raise ValueError(
                f"No data selected for class indices {index}. "
                f"Available classes in data: {unique_targets.tolist()}"
            )
        
        return data_tmp, targets_tmp
    
    def SelectfromTxt(self, data: np.ndarray, targets: np.ndarray, index_path: str):
        """
        セッションファイルからデータインデックスを読み込んでデータを選択
        
        Args:
            data: 特徴量データ
            targets: ラベルデータ
            index_path: セッションファイルのパス
        
        Returns:
            (選択された特徴量, 選択されたラベル)
        """
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"セッションファイルが見つかりません: {index_path}")
        
        # セッションファイルからデータインデックスを読み込む
        with open(index_path, 'r') as f:
            indices = [int(line.strip()) for line in f if line.strip()]
        
        indices = np.array(indices, dtype=np.int64)
        
        # インデックスの範囲チェック
        if len(indices) > 0 and (indices.max() >= len(data) or indices.min() < 0):
            raise ValueError(
                f"データインデックスの範囲が不正です: "
                f"min={indices.min()}, max={indices.max()}, data_len={len(data)}"
            )
        
        data_tmp = data[indices]
        targets_tmp = targets[indices]
        
        return data_tmp, targets_tmp
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        データポイントを取得
        
        Returns:
            (特徴量, ラベル) のタプル（torch.Tensor形式）
        """
        feature = torch.from_numpy(self.data[index]).float()
        target = torch.tensor(self.targets[index], dtype=torch.long)
        return feature, target
