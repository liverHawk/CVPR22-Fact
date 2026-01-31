"""
CNN1Dエンコーダー（1次元畳み込み）

表形式データを1次元シーケンスとして扱うCNN1Dエンコーダーを実装します。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DEncoder(nn.Module):
    """
    CNN1Dエンコーダー
    
    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 各畳み込み層の出力チャネル数のリスト（例: [128, 256, 512]）
        kernel_sizes: 各畳み込み層のカーネルサイズのリスト（デフォルト: 3）
        output_dim: 出力埋め込みの次元数
        dropout: Dropout率（デフォルト: 0.0）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 256, 512],
        kernel_sizes: list = None,
        output_dim: int = 256,
        dropout: float = 0.0,
    ):
        super(CNN1DEncoder, self).__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3] * len(hidden_dims)
        
        if len(kernel_sizes) != len(hidden_dims):
            raise ValueError(
                f"kernel_sizesの長さ ({len(kernel_sizes)}) が "
                f"hidden_dimsの長さ ({len(hidden_dims)}) と一致しません"
            )
        
        layers = []
        in_channels = 1  # 入力は1チャネル（特徴量を1次元シーケンスとして扱う）
        seq_length = input_dim
        
        # 畳み込み層を構築
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            # パディングを追加してシーケンス長を維持
            padding = kernel_size // 2
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # グローバル平均プーリング後の次元
        self.pooled_dim = hidden_dims[-1]
        
        # 出力層
        self.fc = nn.Linear(self.pooled_dim, output_dim)
        self.output_dim = output_dim
        
        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みを初期化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def pre_encode(self, x):
        """
        畳み込み＋プールまで処理（Mixup用の中間表現）。
        fact/helper.py の base_train で pre_encode → mixup → post_encode に必要。
        
        Args:
            x: 入力テンソル (batch_size, input_dim)
        
        Returns:
            中間ベクトル (batch_size, pooled_dim)
        """
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        return x

    def post_encode(self, x):
        """
        中間表現から最終埋め込みへ（pre_encode の出力を入力とする）。
        
        Args:
            x: 中間テンソル (batch_size, pooled_dim)
        
        Returns:
            埋め込みベクトル (batch_size, output_dim)
        """
        return self.fc(x)

    def forward(self, x):
        """
        フォワードパス
        
        Args:
            x: 入力テンソル (batch_size, input_dim)
        
        Returns:
            埋め込みベクトル (batch_size, output_dim)
        """
        x = self.pre_encode(x)
        x = self.post_encode(x)
        return x


def cnn1d_encoder(
    input_dim: int,
    hidden_dims: list = None,
    kernel_sizes: list = None,
    output_dim: int = 256,
    dropout: float = 0.0,
):
    """
    CNN1Dエンコーダーを作成するファクトリー関数
    
    Args:
        input_dim: 入力特徴量の次元数
        hidden_dims: 各畳み込み層の出力チャネル数のリスト（デフォルト: [128, 256, 512]）
        kernel_sizes: 各畳み込み層のカーネルサイズのリスト
        output_dim: 出力埋め込みの次元数
        dropout: Dropout率
    
    Returns:
        CNN1DEncoderインスタンス
    """
    if hidden_dims is None:
        hidden_dims = [128, 256, 512]
    
    return CNN1DEncoder(input_dim, hidden_dims, kernel_sizes, output_dim, dropout)
