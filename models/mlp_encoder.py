# import torch
import torch.nn as nn
# import torch.nn.functional as F


class MLPEncoder(nn.Module):
    """
    Multi-Layer Perceptron encoder for tabular/feature vector data.
    Designed for CICIDS2017_improved dataset.
    """

    def __init__(
        self, input_dim=78, hidden_dims=[512, 256, 128], output_dim=128, dropout=0.1
    ):
        super(MLPEncoder, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through MLP encoder.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Encoded features of shape (batch_size, output_dim)
        """
        return self.encoder(x)


def mlp_encoder(input_dim=78, hidden_dims=[512, 256, 128], output_dim=128, dropout=0.1):
    """
    Factory function to create MLP encoder.
    Default parameters are for CICIDS2017_improved dataset.
    """
    return MLPEncoder(input_dim, hidden_dims, output_dim, dropout)
