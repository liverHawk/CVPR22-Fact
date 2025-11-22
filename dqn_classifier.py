"""FACT 風の前方互換性を取り入れた DQN 分類スクリプト."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ClassificationBatch:
    states: Tensor
    actions: Tensor
    rewards: Tensor
    next_states: Tensor
    dones: Tensor


class ClassificationEnv:
    """表形式データを1ステップ=1サンプルとして返す簡易環境."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
    ) -> None:
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.rng = np.random.default_rng(42)
        self._current_idx = 0

    @property
    def num_actions(self) -> int:
        return int(self.labels.max() + 1)

    @property
    def state_dim(self) -> int:
        return self.features.shape[1]

    def reset(self) -> np.ndarray:
        self._current_idx = int(self.rng.integers(0, len(self.features)))
        return self.features[self._current_idx]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        target = self.labels[self._current_idx]
        reward = self.reward_correct if action == target else self.reward_incorrect
        self._current_idx = int(self.rng.integers(0, len(self.features)))
        next_state = self.features[self._current_idx]
        done = False
        info = {"label": int(target), "correct": action == target}
        return next_state, reward, done, info


class ReplayBuffer:
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
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.index
        self.states[idx] = torch.from_numpy(state)
        self.actions[idx] = int(action)
        self.rewards[idx] = reward
        self.next_states[idx] = torch.from_numpy(next_state)
        self.dones[idx] = float(done)
        self.index = (self.index + 1) % self.capacity
        if self.index == 0:
            self.full = True

    def __len__(self) -> int:
        return self.capacity if self.full else self.index

    def sample(self, batch_size: int) -> ClassificationBatch:
        max_index = self.capacity if self.full else self.index
        indices = torch.randint(0, max_index, (batch_size,), device=self.device)
        return ClassificationBatch(
            states=self.states[indices].to(self.device),
            actions=self.actions[indices].to(self.device),
            rewards=self.rewards[indices].to(self.device),
            next_states=self.next_states[indices].to(self.device),
            dones=self.dones[indices].to(self.device),
        )


class FactQNetwork(nn.Module):
    """FACT の仮想プロトタイプを取り入れた Q ネットワーク."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        virtual_classes: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.virtual_classes = virtual_classes
        self.temperature = temperature

        layers: List[nn.Module] = []
        in_dim = state_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, embedding_dim))
        self.encoder = nn.Sequential(*layers)

        self.real_classifier = nn.Linear(embedding_dim, num_actions, bias=False)
        self.virtual_classifier = (
            nn.Linear(embedding_dim, virtual_classes, bias=False)
            if virtual_classes > 0
            else None
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.real_classifier.weight)
        if self.virtual_classifier is not None:
            nn.init.orthogonal_(self.virtual_classifier.weight)

    def forward(self, states: Tensor) -> Tensor:
        embeddings = self.encoder(states)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        real_weight = F.normalize(self.real_classifier.weight, p=2, dim=-1)
        logits = self.temperature * F.linear(embeddings, real_weight)
        if self.virtual_classifier is not None:
            virtual_weight = F.normalize(self.virtual_classifier.weight, p=2, dim=-1)
            virtual_logits = self.temperature * F.linear(embeddings, virtual_weight)
            logits = torch.cat([logits, virtual_logits], dim=1)
        return logits

    def q_values(self, states: Tensor) -> Tensor:
        return self.forward(states)[:, : self.num_actions]


def build_fact_mask(
    num_actions: int,
    total_classes: int,
    mask_size: int,
) -> Optional[Tensor]:
    if mask_size <= 0 or total_classes <= num_actions:
        return None
    dummy_count = total_classes - num_actions
    mask = torch.zeros(num_actions, total_classes, dtype=torch.float32)
    dummy_indices = torch.arange(num_actions, total_classes)
    for cls in range(num_actions):
        sample_size = min(mask_size, dummy_count)
        chosen = torch.randperm(dummy_count)[:sample_size]
        mask[cls, dummy_indices[chosen]] = 1.0
    return mask


def fact_forward_compatibility_loss(
    logits: Tensor,
    actions: Tensor,
    mask: Optional[Tensor],
    base_class: int,
) -> Tensor:
    if logits.size(1) <= base_class:
        return torch.tensor(0.0, device=logits.device)
    batch = logits.size(0)
    masked_logits = logits.clone()
    row_idx = torch.arange(batch, device=logits.device)
    masked_logits[row_idx, actions] = -1e9
    if mask is not None:
        action_mask = mask[actions].to(logits.device)
        masked_logits = masked_logits.masked_fill(action_mask == 0, -1e9)
    dummy_part = masked_logits[:, base_class:]
    pseudo = dummy_part.argmax(dim=-1) + base_class
    return F.cross_entropy(masked_logits, pseudo)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        embedding_dim: int,
        lr: float,
        gamma: float,
        device: torch.device,
        virtual_classes: int,
        temperature: float,
        fact_mask: Optional[Tensor],
        fact_loss_weight: float,
    ) -> None:
        self.num_actions = action_dim
        self.virtual_classes = virtual_classes
        self.fact_loss_weight = fact_loss_weight
        self.fact_mask = fact_mask.to(device) if fact_mask is not None else None
        self.q_net = FactQNetwork(
            state_dim,
            action_dim,
            hidden_dims,
            embedding_dim,
            virtual_classes,
            temperature,
        ).to(device)
        self.target_net = FactQNetwork(
            state_dim,
            action_dim,
            hidden_dims,
            embedding_dim,
            virtual_classes,
            temperature,
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.device = device

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net.q_values(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def update(self, batch: ClassificationBatch) -> Tuple[float, float]:
        logits = self.q_net(batch.states)
        q_values = (
            logits[:, : self.num_actions]
            .gather(1, batch.actions.unsqueeze(1))
            .squeeze(1)
        )
        with torch.no_grad():
            next_logits = self.target_net(batch.next_states)
            next_q = next_logits[:, : self.num_actions].max(1).values
            target = batch.rewards + self.gamma * (1 - batch.dones) * next_q
        td_loss = F.mse_loss(q_values, target)
        reg_loss = torch.tensor(0.0, device=self.device)
        if self.virtual_classes > 0 and self.fact_loss_weight > 0.0:
            reg_loss = fact_forward_compatibility_loss(
                logits,
                batch.actions,
                self.fact_mask,
                base_class=self.num_actions,
            )
        total_loss = td_loss + self.fact_loss_weight * reg_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()
        return float(td_loss.item()), float(reg_loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.q_net.state_dict())


def exponential_decay(start: float, end: float, decay: float, step: int) -> float:
    return end + (start - end) * math.exp(-1.0 * step / decay)


def prepare_dataset(
    csv_path: str,
    label_column: str,
    feature_columns: Optional[Sequence[str]],
    normalize: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    df = pd.read_csv(csv_path)
    if feature_columns:
        features_df = df.loc[:, feature_columns]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != label_column]
        if not numeric_cols:
            raise ValueError("数値特徴量が選択されていません")
        features_df = df.loc[:, numeric_cols]

    labels_series = df[label_column].astype("category")
    class_names = list(labels_series.cat.categories)
    labels = labels_series.cat.codes.to_numpy()
    features = features_df.to_numpy(dtype=np.float32)
    if normalize:
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-6
        features = (features - mean) / std
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, labels, class_names


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    features, labels, class_names = prepare_dataset(
        args.train_csv,
        args.label_column,
        args.feature_columns,
        args.normalize,
    )
    env = ClassificationEnv(features, labels)
    total_classes = env.num_actions + args.fact_virtual_classes
    fact_mask = build_fact_mask(
        num_actions=env.num_actions,
        total_classes=total_classes,
        mask_size=args.fact_mask_size,
    )
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.num_actions,
        hidden_dims=args.hidden_dims,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        gamma=args.gamma,
        device=device,
        virtual_classes=args.fact_virtual_classes,
        temperature=args.temperature,
        fact_mask=fact_mask,
        fact_loss_weight=args.fact_loss_weight,
    )
    buffer = ReplayBuffer(args.replay_size, env.state_dim, device)

    global_step = 0
    stats = []
    state = env.reset()
    for episode in range(1, args.max_episodes + 1):
        episode_reward = 0.0
        episode_td_loss = 0.0
        episode_fact_loss = 0.0
        for _ in range(args.steps_per_episode):
            epsilon = exponential_decay(
                args.epsilon_start,
                args.epsilon_end,
                args.epsilon_decay,
                global_step,
            )
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                td_loss, fact_loss = agent.update(batch)
                episode_td_loss += td_loss
                episode_fact_loss += fact_loss

            if global_step % args.target_update_interval == 0:
                agent.update_target()
            global_step += 1

        denom = max(1, args.steps_per_episode)
        stats.append(
            (
                episode_reward,
                episode_td_loss / denom,
                episode_fact_loss / denom,
            )
        )
        if episode % args.log_interval == 0:
            recent = stats[-args.log_interval :]
            mean_reward = sum(r for r, _, _ in recent) / args.log_interval
            mean_td = sum(t for _, t, _ in recent) / args.log_interval
            mean_fact = sum(f for _, _, f in recent) / args.log_interval
            print(
                f"[Episode {episode:04d}] reward={mean_reward:.3f} "
                f"td_loss={mean_td:.5f} fact_loss={mean_fact:.5f} epsilon={epsilon:.3f}"
            )

    print("学習完了")
    print("クラス一覧:", class_names)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FACT+DQN による表形式データ分類")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/CICIDS2017_improved/train.csv",
        help="学習に使用する CSV ファイル",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="Label",
        help="ターゲット列名",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        nargs="+",
        default=None,
        help="使用する特徴量列名 (未指定なら数値列すべて)",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256],
        help="FACT エンコーダの隠れ層ユニット",
    )
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=5_000.0)
    parser.add_argument("--max-episodes", type=int, default=500)
    parser.add_argument("--steps-per-episode", type=int, default=256)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument(
        "--fact-virtual-classes",
        type=int,
        default=8,
        help="未来クラス用の仮想プロトタイプ数",
    )
    parser.add_argument(
        "--fact-mask-size",
        type=int,
        default=3,
        help="1クラスあたり接続する仮想プロトタイプ数",
    )
    parser.add_argument(
        "--fact-loss-weight",
        type=float,
        default=0.1,
        help="FACT 前方互換性ロスの重み",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch デバイス指定 (cuda か cpu)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="特徴量を標準化するかどうか",
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
