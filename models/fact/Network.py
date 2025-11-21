import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.mlp_encoder import mlp_encoder
from models.cnn1d_encoder import cnn1d_encoder


class DistanceMetric:
    """距離メトリクスの基底クラス（拡張可能な設計）"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        """
        距離を計算
        
        Args:
            embeddings: (batch_size, num_features)
            prototypes: (num_classes, num_features)
        
        Returns:
            distances: (batch_size, num_classes)
        """
        raise NotImplementedError
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        """
        クラス間の距離を計算（閾値計算用）
        
        Args:
            prototypes: (num_classes, num_features)
        
        Returns:
            inter_distances: クラス間距離のフラットなテンソル
        """
        raise NotImplementedError


class CosineDistance(DistanceMetric):
    """コサイン距離"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        cosine_sim = F.linear(embeddings_norm, prototypes_norm)
        return 1.0 - cosine_sim
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        inter_class_distances = 1.0 - F.linear(prototypes_norm, prototypes_norm)
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


class EuclideanDistance(DistanceMetric):
    """ユークリッド距離"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        embeddings_expanded = embeddings.unsqueeze(1)  # (batch_size, 1, num_features)
        prototypes_expanded = prototypes.unsqueeze(0)  # (1, num_classes, num_features)
        return torch.norm(embeddings_expanded - prototypes_expanded, p=2, dim=-1)
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_expanded = prototypes.unsqueeze(1)  # (num_classes, 1, num_features)
        prototypes_expanded2 = prototypes.unsqueeze(0)  # (1, num_classes, num_features)
        inter_class_distances = torch.norm(prototypes_expanded - prototypes_expanded2, p=2, dim=-1)
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


class NormalizedEuclideanDistance(DistanceMetric):
    """正規化ユークリッド距離"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        embeddings_expanded = embeddings_norm.unsqueeze(1)
        prototypes_expanded = prototypes_norm.unsqueeze(0)
        return torch.norm(embeddings_expanded - prototypes_expanded, p=2, dim=-1)
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        prototypes_expanded = prototypes_norm.unsqueeze(1)
        prototypes_expanded2 = prototypes_norm.unsqueeze(0)
        inter_class_distances = torch.norm(prototypes_expanded - prototypes_expanded2, p=2, dim=-1)
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


class ManhattanDistance(DistanceMetric):
    """マンハッタン距離（L1距離）"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        embeddings_expanded = embeddings.unsqueeze(1)  # (batch_size, 1, num_features)
        prototypes_expanded = prototypes.unsqueeze(0)  # (1, num_classes, num_features)
        return torch.norm(embeddings_expanded - prototypes_expanded, p=1, dim=-1)
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_expanded = prototypes.unsqueeze(1)
        prototypes_expanded2 = prototypes.unsqueeze(0)
        inter_class_distances = torch.norm(prototypes_expanded - prototypes_expanded2, p=1, dim=-1)
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


class ChebyshevDistance(DistanceMetric):
    """チェビシェフ距離（L∞距離）"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        embeddings_expanded = embeddings.unsqueeze(1)  # (batch_size, 1, num_features)
        prototypes_expanded = prototypes.unsqueeze(0)  # (1, num_classes, num_features)
        diff = torch.abs(embeddings_expanded - prototypes_expanded)
        return torch.max(diff, dim=-1)[0]  # 各次元での最大値を取得
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_expanded = prototypes.unsqueeze(1)
        prototypes_expanded2 = prototypes.unsqueeze(0)
        diff = torch.abs(prototypes_expanded - prototypes_expanded2)
        inter_class_distances = torch.max(diff, dim=-1)[0]
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


class MahalanobisDistance(DistanceMetric):
    """マハラノビス距離（共分散行列を使用）"""
    
    @staticmethod
    def compute_distance(embeddings, prototypes):
        # 共分散行列を計算（簡易版：単位行列を使用）
        # 実際の実装では、学習データから共分散行列を推定する必要がある
        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        # 簡易版として正規化ユークリッド距離を使用
        embeddings_expanded = embeddings_norm.unsqueeze(1)
        prototypes_expanded = prototypes_norm.unsqueeze(0)
        return torch.norm(embeddings_expanded - prototypes_expanded, p=2, dim=-1)
    
    @staticmethod
    def compute_inter_class_distance(prototypes):
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        prototypes_expanded = prototypes_norm.unsqueeze(1)
        prototypes_expanded2 = prototypes_norm.unsqueeze(0)
        inter_class_distances = torch.norm(prototypes_expanded - prototypes_expanded2, p=2, dim=-1)
        mask = ~torch.eye(inter_class_distances.size(0), dtype=torch.bool, device=inter_class_distances.device)
        return inter_class_distances[mask]


# 距離メトリクスのレジストリ
DISTANCE_METRICS = {
    'cosine': CosineDistance,
    'euclidean': EuclideanDistance,
    'euclidean_normalized': NormalizedEuclideanDistance,
    'manhattan': ManhattanDistance,
    'chebyshev': ChebyshevDistance,
    'mahalanobis': MahalanobisDistance,
}


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
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

        
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)
        self.dummy_orthogonal_classifier=nn.Linear(self.num_features, self.pre_allocate-self.args.base_class, bias=False)
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        
        self.dummy_orthogonal_classifier.weight.data=self.fc.weight.data[self.args.base_class:,:]
        print(self.dummy_orthogonal_classifier.weight.data.size())
        
        print('self.dummy_orthogonal_classifier.weight initialized over.')

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            
            x1 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x2 = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
            
            x = torch.cat([x1[:,:self.args.base_class],x2],dim=1)
            
            x = self.args.temperature * x
            
        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x

    def forpass_fc(self,x):
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
    
    def detect_unknown_by_distance(self, x, known_class_indices=None, distance_threshold=None, distance_type='cosine', distance_metric=None):
        """
        埋め込み空間上での距離ベースで未知クラスを検出（拡張可能な設計）
        
        Args:
            x: 入力データ (batch_size, ...)
            known_class_indices: 既知クラスのインデックスリスト（Noneの場合はbase_classまで）
            distance_threshold: 距離の閾値（Noneの場合は自動計算）
            distance_type: 距離の種類（文字列）またはDistanceMetricインスタンス
            distance_metric: DistanceMetricインスタンス（distance_typeより優先）
        
        Returns:
            is_unknown: 未知クラスかどうかのブールテンソル (batch_size,)
            distances: 各サンプルから最も近い既知クラスプロトタイプへの距離 (batch_size,)
            nearest_class: 最も近い既知クラスのインデックス (batch_size,)
        """
        # 埋め込みを取得
        embeddings = self.encode(x)  # (batch_size, num_features)
        
        # 既知クラスのインデックスを決定
        if known_class_indices is None:
            known_class_indices = torch.arange(self.args.base_class, device=embeddings.device)
        else:
            if not isinstance(known_class_indices, torch.Tensor):
                known_class_indices = torch.tensor(known_class_indices, device=embeddings.device)
        
        # 既知クラスのプロトタイプを取得
        known_prototypes = self.fc.weight[known_class_indices]  # (num_known_classes, num_features)
        
        # 距離メトリクスを取得
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
        
        # 距離を計算
        distances = metric.compute_distance(embeddings, known_prototypes)  # (batch_size, num_known_classes)
        
        # 各サンプルから最も近い既知クラスへの距離を取得
        min_distances, nearest_indices = torch.min(distances, dim=1)  # (batch_size,)
        nearest_class = known_class_indices[nearest_indices]  # 元のクラスインデックスに変換
        
        # 閾値が指定されていない場合は、既知クラス間の距離の統計から自動計算
        if distance_threshold is None:
            inter_class_distances = metric.compute_inter_class_distance(known_prototypes)
            # 平均 + 標準偏差を閾値として使用
            mean_dist = inter_class_distances.mean().item()
            std_dist = inter_class_distances.std().item()
            distance_threshold = mean_dist + 1.5 * std_dist
        
        # 未知クラスかどうかを判定
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
        
        # 未知クラス検出が有効な場合
        if enable_unknown_detection:
            is_unknown, distances, nearest_class = self.detect_unknown_by_distance(
                input, known_class_indices, distance_threshold, distance_type, distance_metric
            )
            return logits, is_unknown, distances, nearest_class
        
        return logits
    
    def forward_with_unknown_detection(self, x, known_class_indices=None, distance_threshold=None, 
                                      distance_type='cosine', distance_metric=None):
        """
        分類と未知クラス検出を同時に行う（後方互換性のためのラッパー）
        
        Returns:
            logits: 分類ロジット (batch_size, num_classes)
            is_unknown: 未知クラスかどうか (batch_size,)
            distances: 最小距離 (batch_size,)
            nearest_class: 最も近い既知クラスのインデックス (batch_size,)
        """
        return self.forward(x, enable_unknown_detection=True, known_class_indices=known_class_indices,
                           distance_threshold=distance_threshold, distance_type=distance_type,
                           distance_metric=distance_metric)
    
    def pre_encode(self,x):
        
        if self.args.dataset in ['cifar100','manyshotcifar']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
            
        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200']:
            x = self.encoder.conv1(x)
            x = self.encoder.bn1(x)
            x = self.encoder.relu(x)
            x = self.encoder.maxpool(x)
            x = self.encoder.layer1(x)
            x = self.encoder.layer2(x)
        
        return x
        
    
    def post_encode(self,x):
        if self.args.dataset in ['cifar100','manyshotcifar']:
            
            x = self.encoder.layer3(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)

        elif self.args.dataset in ['mini_imagenet','manyshotmini','cub200']:
            
            x = self.encoder.layer3(x)
            x = self.encoder.layer4(x)
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.squeeze(-1).squeeze(-1)
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x


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
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
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

