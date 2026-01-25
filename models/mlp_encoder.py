"""
MLPエンコーダー（多層パーセプトロン）

表形式データ用のMLPエンコーダーを実装します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    MLPエンコーダー
    
    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 隠れ層の次元数のリスト（例: [512, 256]）
        output_dim: 出力埋め込みの次元数
        dropout: Dropout率（デフォルト: 0.0）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256],
        output_dim: int = 256,
        dropout: float = 0.0,
    ):
        super(MLPEncoder, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        # 隠れ層を構築
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # 出力層
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = output_dim
        
        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル (batch_size, input_dim)
        
        Returns:
            埋め込みベクトル (batch_size, output_dim)
        """
        return self.encoder(x)


def mlp_encoder(input_dim: int, hidden_dims: list = None, output_dim: int = 256, dropout: float = 0.0):
    """
    MLPエンコーダーを作成するファクトリー関数
    
    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 隠れ層の次元数のリスト（デフォルト: [512, 256]）
        output_dim: 出力埋め込みの次元数
        dropout: Dropout率
    
    Returns:
        MLPEncoderインスタンス
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    return MLPEncoder(input_dim, hidden_dims, output_dim, dropout)
