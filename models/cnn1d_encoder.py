import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, num_features: int, num_classes: int, config: dict):
        super(CNN1D, self).__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        conv1_out = 64
        conv2_out = 128
        kernel_size = 3
        pool_size = 2
        fc1_dim = 256
        embedding_dim = 128
        dropout = 0.5

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(conv1_out)

        self.conv2 = nn.Conv1d(in_channels=conv1_out, out_channels=conv2_out, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(conv2_out)

        self.pool = nn.MaxPool1d(kernel_size=pool_size)

        pooled_size = num_features // pool_size
        flattened_size = conv2_out * pooled_size

        self.fc1 = nn.Linear(flattened_size, fc1_dim)
        self.dropout = nn.Dropout(dropout)

        self.fc_embedding = nn.Linear(fc1_dim, embedding_dim)
        self.fc_classifier = nn.Linear(fc1_dim, num_classes)

    def forward(self, x, return_logits=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)

        embedding = self.fc_embedding(x)
        embedding_normalized = F.normalize(embedding, p=2, dim=-1)

        logits = self.fc_classifier(x)

        if return_logits:
            return embedding_normalized, logits
        else:
            return logits
    
    def get_embedding(self, x):
        embedding, _ = self.forward(x, return_logits=True)
        return embedding


def cnn1d_encoder(num_features: int, num_classes: int, config: dict):
    return CNN1D(num_features, num_classes, config)
