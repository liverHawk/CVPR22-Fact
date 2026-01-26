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
import pandas as pd
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
        window_size: Moving Min-Max正規化のウィンドウサイズ（デフォルト: 1000）
    """
    
    def __init__(
        self,
        root: str = 'data/',
        train: bool = True,
        index: Optional[Union[List[int], np.ndarray]] = None,
        index_path: Optional[str] = None,
        base_sess: Optional[bool] = None,
        label_column: str = 'Label',
        normalize_method: str = 'standard',
        window_size: int = 1000,
    ):
        self.root = os.path.expanduser(root)
        self.train = train
        self.label_column = label_column
        self.normalize_method = normalize_method
        self.window_size = window_size
        
        # CSVファイルのパス（train/またはtest/ディレクトリから読み込む）
        data_dir = os.path.join(self.root, 'train' if train else 'test')
        
        # ディレクトリ内のCSVファイルを検索
        if os.path.isdir(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            if len(csv_files) == 0:
                raise FileNotFoundError(f"CSVファイルが見つかりません: {data_dir}")
            
            # 複数のCSVファイルがある場合はマージ
            print(f"Loading CSV files from {data_dir}...")
            dfs = []
            for csv_file in sorted(csv_files):
                csv_path = os.path.join(data_dir, csv_file)
                print(f"  Reading {csv_file}...")
                dfs.append(pd.read_csv(csv_path))
            
            if len(dfs) == 1:
                df = dfs[0]
            else:
                print(f"  Merging {len(dfs)} CSV files...")
                df = pd.concat(dfs, ignore_index=True)
        else:
            # 後方互換性: ルートディレクトリにtrain.csv/test.csvがある場合
            csv_filename = 'train.csv' if train else 'test.csv'
            csv_path = os.path.join(self.root, csv_filename)
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSVファイルまたはディレクトリが見つかりません: {data_dir} または {csv_path}")
            print(f"Loading {csv_filename}...")
            df = pd.read_csv(csv_path)
        
        if label_column not in df.columns:
            raise ValueError(
                f"ラベル列 '{label_column}' が見つかりません。"
                f"利用可能な列: {df.columns.tolist()}"
            )
        
        # 特徴量列を取得（ラベル列と非数値列を除外）
        # id, Flow ID, Src IP, Dst IP, Timestampなどの非数値列も除外
        exclude_columns = [
            label_column, 'id', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port',
            'Timestamp', 'Attempted Category'
        ]
        feature_columns = [
            col for col in df.columns
            if col not in exclude_columns and df[col].dtype in ['int64', 'float64']
        ]
        
        # ラベルを数値インデックスに変換
        unique_labels = sorted(df[label_column].unique())
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)
        
        # デバッグ出力
        print(f"Found {len(unique_labels)} unique labels: {unique_labels[:10]}...")  # 最初の10個を表示
        print(f"Label to index mapping (first 10): {dict(list(self.label_to_idx.items())[:10])}")
        
        # 特徴量とラベルを抽出
        features = df[feature_columns].values.astype(np.float32)
        labels_str = df[label_column].values
        labels = np.array([self.label_to_idx[label] for label in labels_str], dtype=np.int64)
        
        # NaNや無限大の値を処理
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 正規化統計量を計算
        if train:
            self._compute_normalization_stats(features)
        else:
            # テストデータの場合
            if self.normalize_method == 'moving_minmax':
                # Moving Min-Maxの場合は、テストデータでも統計量を計算
                self._compute_normalization_stats(features)
            else:
                # その他の正規化方法の場合は、訓練データの統計量を読み込む
                self._load_normalization_stats()
        
        # 正規化を適用
        features = self._normalize(features)
        
        # セッション選択
        if base_sess:
            # ベースセッション: クラスインデックスから選択
            if index is None:
                raise ValueError("base_sess=Trueの場合、indexを指定してください")
            self.data, self.targets = self.SelectfromClasses(features, labels, index)
        else:
            # 新規セッション: セッションファイルからデータインデックスを読み込み
            if index_path is not None:
                self.data, self.targets = self.SelectfromTxt(features, labels, index_path)
            elif index is not None:
                # データインデックスのリストが直接指定された場合
                index_array = np.array(index, dtype=np.int64)
                self.data = features[index_array]
                self.targets = labels[index_array]
            else:
                # すべてのデータを使用
                self.data = features
                self.targets = labels
        
        print(f"Loaded {len(self.data)} samples, {len(np.unique(self.targets))} classes")
    
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
            # Moving Min-Max正規化: 各データポイントに対して、その時点までのデータ（最大window_size個）でmin/maxを計算
            # min/maxが変化する時点のみを記録
            n_samples, n_features = features.shape
            min_max_changes = []  # (start_index, min_values, max_values)のリスト
            
            print(f"Computing Moving Min-Max normalization with window_size={self.window_size}...")
            prev_min = None
            prev_max = None
            
            for i in range(n_samples):
                # ウィンドウの範囲を決定
                if i < self.window_size:
                    # 最初のwindow_size個のデータ: 0からi+1まで
                    window_start = 0
                    window_end = i + 1
                else:
                    # window_size個以降: i-window_size+1からi+1まで
                    window_start = i - self.window_size + 1
                    window_end = i + 1
                
                # ウィンドウ内のmin/maxを計算
                window_data = features[window_start:window_end]
                current_min = np.min(window_data, axis=0)
                current_max = np.max(window_data, axis=0)
                
                # 前のmin/maxと比較（浮動小数点数の比較には注意）
                if prev_min is None or not np.allclose(current_min, prev_min) or not np.allclose(current_max, prev_max):
                    # min/maxが変化した場合のみ記録
                    min_max_changes.append((i, current_min.copy(), current_max.copy()))
                    prev_min = current_min.copy()
                    prev_max = current_max.copy()
            
            # 統計量を保存
            self.min_max_changes = min_max_changes
            stats_path = os.path.join(self.root, 'normalization_stats.npz')
            
            # 変化点のリストを保存（各要素は(start_index, min_values, max_values)）
            # npzファイルに保存するために、リストを配列に変換
            change_indices = np.array([change[0] for change in min_max_changes], dtype=np.int64)
            min_values_list = np.array([change[1] for change in min_max_changes])
            max_values_list = np.array([change[2] for change in min_max_changes])
            
            np.savez(
                stats_path,
                window_size=self.window_size,
                change_indices=change_indices,
                min_values_list=min_values_list,
                max_values_list=max_values_list,
                method='moving_minmax'
            )
            print(f"Saved {len(min_max_changes)} min/max change points")
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
        elif method == 'minmax':
            self.min = stats['min']
            self.max = stats['max']
            self.range = self.max - self.min
            self.range[self.range == 0] = 1.0
        elif method == 'moving_minmax':
            # Moving Min-Max統計量を読み込む
            self.window_size = int(stats.get('window_size', 1000))
            change_indices = stats['change_indices']
            min_values_list = stats['min_values_list']
            max_values_list = stats['max_values_list']
            
            # 変化点のリストを再構築
            self.min_max_changes = [
                (int(change_indices[i]), min_values_list[i], max_values_list[i])
                for i in range(len(change_indices))
            ]
            print(f"Loaded {len(self.min_max_changes)} min/max change points for Moving Min-Max normalization")
        else:
            raise ValueError(f"Unknown normalization method in stats: {method}")
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """特徴量を正規化"""
        if self.normalize_method == 'standard':
            return (features - self.mean) / self.std
        elif self.normalize_method == 'minmax':
            return (features - self.min) / self.range
        elif self.normalize_method == 'moving_minmax':
            # Moving Min-Max正規化: 各データポイントに対して、その時点までのデータ（最大window_size個）でmin/maxを計算
            n_samples, n_features = features.shape
            normalized_features = np.zeros_like(features)
            
            # 変化点のリストから該当するmin/maxを取得するためのインデックス
            change_indices = np.array([change[0] for change in self.min_max_changes])
            
            for i in range(n_samples):
                # データポイントiが属する変化点を検索
                # change_indicesの中で、i以下の最大のインデックスを見つける
                idx = np.searchsorted(change_indices, i, side='right') - 1
                if idx < 0:
                    idx = 0
                
                # 該当するmin/maxを取得
                _, min_values, max_values = self.min_max_changes[idx]
                
                # 正規化を適用
                range_values = max_values - min_values
                range_values[range_values == 0] = 1.0  # ゼロ除算を回避
                normalized_features[i] = (features[i] - min_values) / range_values
            
            return normalized_features
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
