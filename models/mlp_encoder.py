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
        
        # 中間層（pre_encoder）を構築
        pre_layers = []
        in_dim = input_dim
        
        # 隠れ層を構築
        for hidden_dim in hidden_dims:
            pre_layers.append(nn.Linear(in_dim, hidden_dim))
            pre_layers.append(nn.BatchNorm1d(hidden_dim))
            pre_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                pre_layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        self.pre_encoder = nn.Sequential(*pre_layers)
        self.pre_output_dim = in_dim  # 中間層の出力次元
        
        # 出力層（post_encoder）を構築
        post_layers = [nn.Linear(in_dim, output_dim)]
        self.post_encoder = nn.Sequential(*post_layers)
        self.output_dim = output_dim
        
        # 後方互換性のため、全層を通すencoderも保持
        all_layers = pre_layers + post_layers
        self.encoder = nn.Sequential(*all_layers)
        
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
        フォワードパス（全層を通す）
        
        Args:
            x: 入力テンソル (batch_size, input_dim)
        
        Returns:
            埋め込みベクトル (batch_size, output_dim)
        """
        return self.encoder(x)
    
    def pre_encode(self, x):
        """
        中間層まで処理（pre_encoder）
        
        Args:
            x: 入力テンソル (batch_size, input_dim)
        
        Returns:
            中間層の出力 (batch_size, pre_output_dim)
        """
        return self.pre_encoder(x)
    
    def post_encode(self, x):
        """
        中間層の出力から最終出力まで処理（post_encoder）
        
        Args:
            x: 中間層の出力テンソル (batch_size, pre_output_dim)
        
        Returns:
            埋め込みベクトル (batch_size, output_dim)
        """
        return self.post_encoder(x)


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
