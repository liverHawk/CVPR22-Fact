import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.mlp_encoder import mlp_encoder
from models.cnn1d_encoder import cnn1d_encoder


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
        
        elif self.args.dataset == 'CICIDS2017_improved':
            if self.args.encoder == 'mlp':
                # MLP encoder: execute first 2 blocks (66->512, 512->256)
                # Each block: Linear -> BatchNorm -> ReLU -> Dropout (4 layers)
                # First 2 blocks = 8 layers (indices 0-7)
                for i in range(8):  # First 2 blocks: 4 layers each
                    x = self.encoder.encoder[i](x)
            elif self.args.encoder == 'cnn1d':
                # CNN1D: execute conv layers (conv1, bn1, relu, conv2, bn2, relu, pool)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # Add channel dimension
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = F.relu(x)
                x = self.encoder.conv2(x)
                x = self.encoder.bn2(x)
                x = F.relu(x)
                x = self.encoder.pool(x)
                x = x.view(x.size(0), -1)  # Flatten
        
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
        
        elif self.args.dataset == 'CICIDS2017_improved':
            if self.args.encoder == 'mlp':
                # MLP encoder: execute remaining layers (256->128, 128->128)
                # Continue from layer 8 (after first 2 blocks)
                # Remaining: 3rd block (4 layers) + output layer (1 layer) = 5 layers (indices 8-12)
                for i in range(8, len(self.encoder.encoder)):
                    x = self.encoder.encoder[i](x)
            elif self.args.encoder == 'cnn1d':
                # CNN1D: execute fc layers (fc1, dropout, relu, fc_embedding)
                x = self.encoder.fc1(x)
                x = self.encoder.dropout(x)
                x = F.relu(x)
                x = self.encoder.fc_embedding(x)
                x = F.normalize(x, p=2, dim=-1)  # Normalize like in get_embedding
        
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
            
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

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

