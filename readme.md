# FACT + DQN Few-Shot Class Incremental Learning

Forward-Compatible Few-Shot Class-Incremental Learning (FACT, CVPR 2022) をベースに、DQN を用いた強化学習でクラス選択方針を学習する研究用実装です。ベースセッションでは FACT をそのまま再現し、インクリメンタルセッションでは DQN による行動選択と FACT の損失を組み合わせることで、低ショット・逐次クラス追加シナリオに対応します。

## 目標とアーキテクチャ

- **Base Session (session 0)**: 大量の既知クラスで FACT を学習し、エンコーダ＋コサイン分類器を獲得。
- **Incremental Sessions (session 1+)**: ごく少数の新クラスを逐次追加し、FACT の前方互換性を保ちつつ、DQN がクラス決定ポリシーを改善。
- **強化学習統合**: Replay Buffer、Target Network、ε-greedy を備えた標準的な DQN を FACT ヘッドに接続。
- **報酬設計**: Simple / Distance / FACT Loss の 3 種類（`models/fact/rl_components.py` の `RewardCalculator` を参照）。

## セットアップ

```bash
# 依存関係のインストール（uv + pyproject.toml）
make install

# プロジェクトの同期
uv sync
```

PyTorch などの主要依存は `pyproject.toml`/`uv.lock` に定義されています。必要に応じて `PYTHONPATH` にリポジトリ直下を追加してください。

## データセット

`data/` 以下に CSV や画像データ、`data/index_list/<dataset>/session_*.txt` に各セッションで使用するクラスインデックスが格納されています。  
現状の実装は以下のデータセットを想定しています。

- `CICIDS2017_improved`（デフォルト、表形式データ）
- `cifar100`, `cub200`, `mini_imagenet`（画像データ：dataloader/* に前処理実装あり）

必要に応じて `-dataroot` で外部パスを指定してください。

## 典型的な実行コマンド

### FACT（標準 FSCIL）

```bash
python train.py \
    -project fact \
    -dataset CICIDS2017_improved \
    -encoder mlp \
    -epochs_base 100 \
    -lr_base 0.001 \
    -schedule Cosine \
    -batch_size_base 128 \
    -temperature 16 \
    -balance 1.0 \
    -loss_iter 50 \
    --use_wandb \
    --wandb_project fact-cicids2017
```

### FACT + DQN（強化学習付き）

```bash
python train.py \
    -project fact_rl \
    -dataset CICIDS2017_improved \
    -encoder mlp \
    -epochs_base 100 \
    -lr_base 0.001 \
    -schedule Cosine \
    -batch_size_base 128 \
    -temperature 16 \
    -balance 1.0 \
    -loss_iter 50 \
    --rl_reward_type simple \
    --rl_buffer_size 50000 \
    --rl_batch_size 64 \
    --rl_lr 0.001 \
    --rl_gamma 0.99 \
    --rl_num_updates 1000 \
    --use_wandb \
    --wandb_project fact-rl-cicids2017
```

`-encoder` は `mlp`（CICIDS）または `cnn1d` を指定できます。画像データセット向けには `models/resnet18_encoder.py`/`resnet20_cifar.py` を参照してください。

## 強化学習オプション

- `--rl_reward_type simple`  
  正解で `reward_correct`（既定 +1.0）、不正解で `reward_incorrect`（既定 -1.0）。
- `--rl_reward_type distance`  
  埋め込みとクラスプロトタイプのコサイン類似度を利用。正解時は `reward_correct * similarity(target)`、不正解時は `reward_incorrect * (1 - similarity(target) + similarity(action))`。
- `--rl_reward_type fact_loss`  
  FACT のクロスエントロピー損失を報酬へ変換。正解時は `reward_correct * exp(-loss)`、不正解時は `reward_incorrect * (1 + loss)`。

その他主要ハイパーパラメータ:

| 引数 | 既定値 | 説明 |
| --- | --- | --- |
| `--rl_buffer_size` | 50000 | Replay Buffer の容量 |
| `--rl_batch_size` | 64 | DQN 更新バッチサイズ |
| `--rl_lr` | 1e-3 | DQN ヘッドの学習率 |
| `--rl_gamma` | 0.99 | 割引率 |
| `--rl_epsilon_start/end/decay` | 1.0 / 0.05 / 5000 | ε-greedy のスケジュール（`rl_components.exponential_decay` を使用） |
| `--rl_num_updates` | 1000 | 各セッションでの DQN 更新回数 |
| `--rl_target_update` | 500 | Target network を同期するステップ間隔 |
| `--rl_virtual_classes` | 0 | FACT 前方互換性のための仮想クラス数 |

## Makefile レシピ

反復実験向けに代表的なターゲットを用意しています。GPU ID は `DEVICE` 変数で制御します。

```bash
# CIFAR100 で FACT（README 準拠のハイパラ）
make train_fact_cifar DEVICE=0 DATAROOT=/path/to/cifar

# CUB200 / miniImageNet も同様（train_fact_cub / train_fact_mini）

# FACT + DQN を CICIDS2017_improved で実行
make train_fact_rl DEVICE=0

# 速いデバッグ用（エポック短縮）
make train_fact_rl_debug DEVICE=0
```

ハイパーパラメータは `Makefile` 冒頭の `FACT_*_EXTRA` 変数で確認できます。

## ログと成果物

- `checkpoint/<dataset>/<project>/...` に各セッションのモデル、`results.txt`、混同行列 (`session*_rl_confusion_matrix.png`) などを保存。
- `--use_wandb` 時は Weights & Biases に学習カーブを送信（`--wandb_project`, `--wandb_entity`, `--wandb_group` 等を適宜設定）。
- `-debug` フラグでデータ件数やエポック数を減らし、素早く実行可能。

## 典型的なワークフロー

1. **Base session**: FACT で既知クラスを学習し、エンコーダとプロトタイプを保存。
2. **Incremental session**: 新クラスデータを読み込み、ε-greedy に従いサンプルを選択→DQN で行動価値を更新→FACT の損失でエンコーダ互換性を維持。
3. **評価**: 各セッション終了後に全クラスで Few-Shot テストを実施。

## プロジェクト構成（抜粋）

```
.
├── train.py                   # エントリポイント（FACT/FACT+RL切り替え）
├── models/
│   ├── base/                  # FACT (CVPR22) オリジナル実装
│   ├── fact/                  # 強化学習版トレーナ、ヘルパー、Network
│   │   ├── fscil_trainer_rl.py
│   │   ├── rl_components.py   # Replay Buffer, DQN Head, RewardCalculator
│   │   └── rl_trainer_helper.py
│   ├── resnet18_encoder.py
│   ├── resnet20_cifar.py
│   └── mlp_encoder.py / cnn1d_encoder.py
├── dataloader/                # データセット別ローダ
└── checkpoint/                # 学習済み重み／解析結果
```

## 参考文献

- FACT: Forward-Compatible Few-Shot Class-Incremental Learning (CVPR 2022)
- Deep Q-Network (DQN) for Reinforcement Learning
