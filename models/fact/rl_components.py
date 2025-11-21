"""
FACT-DQN用の強化学習コンポーネント
FACTエンコーダを固定し、分類器のみをDQNで学習
"""
from __future__ import annotations
import random
import math
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RLBatch:
    """強化学習用のバッチデータ"""
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor


class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = capacity
        self.device = device
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.int64)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)
        self.index = 0
        self.full = False

    def push(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
        done: bool,
    ) -> None:
        """バッファに経験を追加"""
        idx = self.index
        self.states[idx] = state.cpu() if isinstance(state, Tensor) else torch.from_numpy(state)
        self.actions[idx] = int(action)
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.cpu() if isinstance(next_state, Tensor) else torch.from_numpy(next_state)
        self.dones[idx] = float(done)
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def sample(self, batch_size: int) -> RLBatch:
        """ランダムにバッチをサンプリング"""
        max_index = self.capacity if self.full else self.index
        indices = torch.randint(0, max_index, (batch_size,), device=self.device)
        return RLBatch(
            states=self.states[indices].to(self.device),
            actions=self.actions[indices].to(self.device),
            rewards=self.rewards[indices].to(self.device),
            next_states=self.next_states[indices].to(self.device),
            dones=self.dones[indices].to(self.device),
        )


class FACTDQNHead(nn.Module):
    """
    FACTエンコーダの上に配置するDQN用のヘッド
    エンコーダは固定し、このヘッドのみを学習
    """
    def __init__(
        self,
        embedding_dim: int,
        num_actions: int,
        virtual_classes: int,
        temperature: float,
        use_cosine: bool = True,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.virtual_classes = virtual_classes
        self.temperature = temperature
        self.use_cosine = use_cosine

        # Real class classifier
        self.real_classifier = nn.Linear(embedding_dim, num_actions, bias=False)

        # Virtual class classifier (for FACT forward compatibility)
        self.virtual_classifier = (
            nn.Linear(embedding_dim, virtual_classes, bias=False)
            if virtual_classes > 0
            else None
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """パラメータの初期化"""
        nn.init.orthogonal_(self.real_classifier.weight)
        if self.virtual_classifier is not None:
            nn.init.orthogonal_(self.virtual_classifier.weight)

    def forward(self, embeddings: Tensor, include_virtual: bool = True) -> Tensor:
        """
        順伝播
        Args:
            embeddings: エンコーダからの埋め込みベクトル [batch, embedding_dim]
            include_virtual: 仮想クラスを含めるかどうか
        Returns:
            Q値 [batch, num_actions (+ virtual_classes)]
        """
        if self.use_cosine:
            # Cosine similarity based classification
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            real_weight = F.normalize(self.real_classifier.weight, p=2, dim=-1)
            logits = self.temperature * F.linear(embeddings, real_weight)

            if include_virtual and self.virtual_classifier is not None:
                virtual_weight = F.normalize(self.virtual_classifier.weight, p=2, dim=-1)
                virtual_logits = self.temperature * F.linear(embeddings, virtual_weight)
                logits = torch.cat([logits, virtual_logits], dim=1)
        else:
            # Dot product based classification
            logits = self.real_classifier(embeddings)
            if include_virtual and self.virtual_classifier is not None:
                virtual_logits = self.virtual_classifier(embeddings)
                logits = torch.cat([logits, virtual_logits], dim=1)

        return logits

    def q_values(self, embeddings: Tensor) -> Tensor:
        """Q値のみを取得（仮想クラスを除く）"""
        return self.forward(embeddings, include_virtual=False)


class RewardCalculator:
    """
    報酬計算器
    FACT損失ベースと距離ベースの報酬を計算
    """
    def __init__(
        self,
        reward_type: str = 'simple',
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        temperature: float = 1.0,
    ):
        """
        Args:
            reward_type: 'simple', 'fact_loss', 'distance'
            reward_correct: 正解時の報酬
            reward_incorrect: 不正解時の報酬
            temperature: 距離ベース報酬のtemperature
        """
        self.reward_type = reward_type
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.temperature = temperature

    def calculate(
        self,
        action: int,
        target: int,
        embeddings: Optional[Tensor] = None,
        prototypes: Optional[Tensor] = None,
        logits: Optional[Tensor] = None,
    ) -> float:
        """
        報酬を計算
        Args:
            action: 選択した行動（クラスID）
            target: 正解ラベル
            embeddings: エンコーダからの埋め込み [embedding_dim]
            prototypes: クラスプロトタイプ [num_classes, embedding_dim]
            logits: モデルの出力logits [num_classes]
        """
        is_correct = (action == target)

        if self.reward_type == 'simple':
            return self.reward_correct if is_correct else self.reward_incorrect

        elif self.reward_type == 'distance':
            # 距離ベースの報酬
            if embeddings is None or prototypes is None:
                raise ValueError("distance reward requires embeddings and prototypes")

            # Cosine similarity
            emb_norm = F.normalize(embeddings.unsqueeze(0), p=2, dim=-1)
            proto_norm = F.normalize(prototypes, p=2, dim=-1)
            similarities = F.linear(emb_norm, proto_norm).squeeze(0)

            # Target similarity
            target_sim = similarities[target].item()
            action_sim = similarities[action].item()

            # Reward based on similarity to target
            if is_correct:
                # Correct action: reward proportional to confidence
                reward = self.reward_correct * target_sim
            else:
                # Incorrect action: penalty proportional to distance from target
                reward = self.reward_incorrect * (1.0 - target_sim + action_sim)

            return float(reward)

        elif self.reward_type == 'fact_loss':
            # FACT損失ベースの報酬
            if logits is None:
                raise ValueError("fact_loss reward requires logits")

            # Cross entropy loss
            loss = F.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([target], device=logits.device)
            ).item()

            # Convert loss to reward (lower loss = higher reward)
            if is_correct:
                reward = self.reward_correct * math.exp(-loss)
            else:
                reward = self.reward_incorrect * (1.0 + loss)

            return float(reward)

        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")


def exponential_decay(start: float, end: float, decay: float, step: int) -> float:
    """Epsilon-greedyのための指数減衰"""
    return end + (start - end) * math.exp(-1.0 * step / decay)
