import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.mlp_encoder import mlp_encoder
from models.cnn1d_encoder import cnn1d_encoder

# 距離メトリクスをインポート（fact/Network.pyから再利用）
from models.fact.Network import (
    DistanceMetric, CosineDistance, EuclideanDistance, 
    NormalizedEuclideanDistance, ManhattanDistance, 
    ChebyshevDistance, MahalanobisDistance, DISTANCE_METRICS
)


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000', 'mini_imagenet_withpath']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset in ['cub200','manyshotcub']:
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        if self.args.dataset == 'CICIDS2017_improved':
            if self.args.encoder == 'mlp':
                self.encoder = mlp_encoder(input_dim=66, hidden_dims=[512, 256, 128], output_dim=128, dropout=0.1)
                self.num_features = 128
            elif self.args.encoder == 'cnn1d':
                self.encoder = cnn1d_encoder(num_features=66, num_classes=self.args.num_classes, config={'conv1_out': 64, 'conv2_out': 128, 'kernel_size': 3, 'pool_size': 2, 'fc1_dim': 256, 'embedding_dim': 128, 'dropout': 0.5})
                self.num_features = 128
            else:
                raise ValueError(f'Unknown encoder: {self.args.encoder}. Available encoders: mlp, cnn1d')
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def encode(self, x):
        if self.args.dataset == 'CICIDS2017_improved':
            # For tabular data
            if self.args.encoder == 'cnn1d':
                # CNN1D expects [batch_size, channels, sequence_length]
                # Reshape from [batch_size, num_features] to [batch_size, 1, num_features]
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                # Use get_embedding to get normalized embedding
                x = self.encoder.get_embedding(x)
            else:
                # MLP encoder directly outputs features
                x = self.encoder(x)
            return x
        else:
            # For image data, use ResNet encoder
            x = self.encoder(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
            return x
    
    
    def forward_with_unknown_detection(self, x, known_class_indices=None, distance_threshold=None, distance_type='cosine'):
        """
        分類と未知クラス検出を同時に行う
        
        Returns:
            logits: 分類ロジット (batch_size, num_classes)
            is_unknown: 未知クラスかどうか (batch_size,)
            distances: 最小距離 (batch_size,)
        """
        # 通常の分類ロジットを取得
        logits = self.forward(x)
        
        # 未知クラス検出
        is_unknown, distances, nearest_class = self.detect_unknown_by_distance(
            x, known_class_indices, distance_threshold, distance_type
        )
        
        return logits, is_unknown, distances

    def detect_unknown_by_distance(self, x, known_class_indices=None, distance_threshold=None, 
                                   distance_type='cosine', distance_metric=None):
        """埋め込み空間上での距離ベースで未知クラスを検出（fact/Network.pyと同じ実装）"""
        embeddings = self.encode(x)
        
        if known_class_indices is None:
            known_class_indices = torch.arange(self.args.base_class, device=embeddings.device)
        else:
            if not isinstance(known_class_indices, torch.Tensor):
                known_class_indices = torch.tensor(known_class_indices, device=embeddings.device)
        
        known_prototypes = self.fc.weight[known_class_indices]
        
        if distance_metric is not None:
            metric = distance_metric
        elif isinstance(distance_type, str):
            if distance_type not in DISTANCE_METRICS:
                available = ', '.join(DISTANCE_METRICS.keys())
                raise ValueError(f"Unknown distance_type: {distance_type}. Available: {available}")
            metric = DISTANCE_METRICS[distance_type]
        elif isinstance(distance_type, type) and issubclass(distance_type, DistanceMetric):
            metric = distance_type
        else:
            raise ValueError(f"Invalid distance_type: {distance_type}")
        
        distances = metric.compute_distance(embeddings, known_prototypes)
        min_distances, nearest_indices = torch.min(distances, dim=1)
        nearest_class = known_class_indices[nearest_indices]
        
        if distance_threshold is None:
            inter_class_distances = metric.compute_inter_class_distance(known_prototypes)
            mean_dist = inter_class_distances.mean().item()
            std_dist = inter_class_distances.std().item()
            distance_threshold = mean_dist + 1.5 * std_dist
        
        is_unknown = min_distances > distance_threshold
        return is_unknown, min_distances, nearest_class
    
    def forward(self, input, enable_unknown_detection=False, known_class_indices=None, 
                distance_threshold=None, distance_type='cosine', distance_metric=None):
        """
        フォワードパス（分類と未知クラス検出を同時に実行可能）
        
        Args:
            input: 入力データ
            enable_unknown_detection: 未知クラス検出を有効にするか
            known_class_indices: 既知クラスのインデックスリスト
            distance_threshold: 距離の閾値
            distance_type: 距離の種類
            distance_metric: DistanceMetricインスタンス
        
        Returns:
            enable_unknown_detection=Falseの場合: logits (batch_size, num_classes)
            enable_unknown_detection=Trueの場合: (logits, is_unknown, distances, nearest_class)
        """
        if self.mode != 'encoder':
            logits = self.forward_metric(input)
        elif self.mode == 'encoder':
            logits = self.encode(input)
            return logits
        else:
            raise ValueError('Unknown mode')
        
        if enable_unknown_detection:
            is_unknown, distances, nearest_class = self.detect_unknown_by_distance(
                input, known_class_indices, distance_threshold, distance_type, distance_metric
            )
            return logits, is_unknown, distances, nearest_class
        
        return logits
    
    def forward_with_unknown_detection(self, x, known_class_indices=None, distance_threshold=None, 
                                      distance_type='cosine', distance_metric=None):
        """分類と未知クラス検出を同時に行う（後方互換性のためのラッパー）"""
        return self.forward(x, enable_unknown_detection=True, known_class_indices=known_class_indices,
                           distance_threshold=distance_threshold, distance_type=distance_type,
                           distance_metric=distance_metric)

    def update_fc(self,dataloader,class_list,session):
        for batch in dataloader:
            data, label = [_.cuda() for _ in batch]
            data=self.encode(data).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            self.update_fc_ft(new_fc,data,label,session)

    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            #print(class_index)
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
            #print(proto)
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x,fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x,fc)
        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

    def update_fc_ft(self,new_fc,data,label,session):
        new_fc=new_fc.clone().detach()
        new_fc.requires_grad=True
        optimized_parameters = [{'params': new_fc}]
        optimizer = torch.optim.SGD(optimized_parameters,lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):
                old_fc = self.fc.weight[:self.args.base_class + self.args.way * (session - 1), :].detach()
                fc = torch.cat([old_fc, new_fc], dim=0)
                logits = self.get_logits(data,fc)
                loss = F.cross_entropy(logits, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pass

        self.fc.weight.data[self.args.base_class + self.args.way * (session - 1):self.args.base_class + self.args.way * session, :].copy_(new_fc.data)

