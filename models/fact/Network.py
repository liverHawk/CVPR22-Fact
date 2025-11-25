import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import resnet18
from models.resnet20_cifar import resnet20


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
        if self.args.dataset == 'cicids2017_improved':
            # For tabular data, create a simple MLP encoder
            # Input dimension is 88 (after removing Label column and other preprocessing)
            input_dim = getattr(self.args, 'input_dim', 88)  # Default to 88, can be overridden
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),  # 88 input features after preprocessing
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            self.num_features = 512
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
        x = self.encoder(x)
        if self.args.dataset == 'cicids2017_improved':
            # For tabular data, no pooling needed
            return x
        else:
            # For image data, apply pooling
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
        
        elif self.args.dataset == 'cicids2017_improved':
            # For tabular data, apply first part of MLP
            x = self.encoder[0](x)  # Linear(69, 128)
            x = self.encoder[1](x)  # ReLU
            x = self.encoder[2](x)  # Dropout
            x = self.encoder[3](x)  # Linear(128, 256)
            x = self.encoder[4](x)  # ReLU
            x = self.encoder[5](x)  # Dropout
        
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
        
        elif self.args.dataset == 'cicids2017_improved':
            # For tabular data, apply final part of MLP
            x = self.encoder[6](x)  # Linear(256, 512)
            x = self.encoder[7](x)  # ReLU
            x = self.encoder[8](x)  # Dropout
        
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
        device = next(self.parameters()).device  # Get device from model parameters
        for batch in dataloader:
            data, label = [_.to(device) for _ in batch]
            data=self.encode(data).detach()

        # Check if we need to expand fc.weight for new classes
        max_class_index = max(class_list)
        current_fc_size = self.fc.weight.size(0)
        
        if max_class_index >= current_fc_size:
            # Expand fc.weight to accommodate new classes
            new_fc_weight = torch.zeros(max_class_index + 1, self.num_features, device=device)
            new_fc_weight[:current_fc_size] = self.fc.weight.data
            self.fc.weight = nn.Parameter(new_fc_weight, requires_grad=True)

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device=device),
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

