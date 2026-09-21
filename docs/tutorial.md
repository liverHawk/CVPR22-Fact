# FACT 拡張チュートリアル: 新規データセット & 新規バックボーン追加ガイド

本ドキュメントでは、FACT (Forward Compatible Few-Shot Class-Incremental Learning) コードベースに **「新しいデータセット」** および **「新しいバックボーンネットワーク」** を追加・カスタマイズする手順をステップバイステップで解説します。

---

## 目次

1. [全体アーキテクチャのおさらい](#1-全体アーキテクチャのおさらい)
2. [新しいデータセットの追加手順](#2-新しいデータセットの追加手順)
   - Step 2.1: データ配置とセッション分割インデックスの作成
   - Step 2.2: `Dataset` クラスの実装
   - Step 2.3: `dataloader/data_utils.py` への登録
   - Step 2.4: `train.py` の引数選択肢への追加
   - Step 2.5: `models/fact/Network.py` のデータセット分岐設定
3. [新しいバックボーンの追加手順](#3-新しいバックボーンの追加手順)
   - Step 3.1: バックボーンモデルの実装要件 (FACTの重要制約)
   - Step 3.2: `models/` への新規バックボーン実装
   - Step 3.3: `models/fact/Network.py` への統合 (`pre_encode` / `post_encode`)
   - Step 3.4: `models/base/Network.py` への統合
4. [完全な実装例 (End-to-End Walkthrough)](#4-完全な実装例-end-to-end-walkthrough)
   - 例1: 新規データセット `cars196` の追加例
   - 例2: 新規バックボーン `ResNet-50` の追加例
5. [トラブルシューティング & 実装時の注意点](#5-トラブルシューティング--実装時の注意点)

---

## 1. 全体アーキテクチャのおさらい

FACT に新コンポーネントを追加する際、以下の3つの主要コンポーネントが互いに依存しています：

```
[データセット追加]
  data/index_list/<dataset>/  ──>  dataloader/<dataset>/  ──>  dataloader/data_utils.py
                                                                         │
                                                                         ▼
                                                                     train.py
                                                                         │
                                                                         ▼
[バックボーン追加]                                                  models.fact.Network
  models/<backbone>.py       ──────────────────────────────────>  (pre/post_encode, fc)
```

---

## 2. 新しいデータセットの追加手順

### Step 2.1: データ配置とセッション分割インデックスの作成

FSCIL では、データを **ベースセッション (Session 0)** と **インクリメンタルセッション (Session 1〜N)** に分割する必要があります。

1. **元画像の配置**:
   ```bash
   data/
   └── my_dataset/
       ├── images/         # 画像ファイル群
       └── ...
   ```

2. **インデックスファイルの作成**:
   `data/index_list/<my_dataset>/` ディレクトリを作成し、各セッションで使用する画像リストまたはクラス指定ファイルを配置します。
   ```bash
   data/index_list/my_dataset/
   ├── session_1.txt    # ベースセッション (全ベースクラスの画像パスまたはID)
   ├── session_2.txt    # Session 1 の Few-shot サンプル (N-way × K-shot 分の画像パス/ID)
   ├── session_3.txt    # Session 2 の Few-shot サンプル
   └── ...
   ```

> [!TIP]
> CUB-200 などの既存形式に合わせる場合、テキストファイル内に `画像相対パス ラベル番号` または画像インデックス番号を行ごとに記述します。

---

### Step 2.2: `Dataset` クラスの実装

`dataloader/<my_dataset>/<my_dataset>.py` を作成し、PyTorch の `Dataset` を継承して実装します。

```python
# dataloader/my_dataset/my_dataset.py
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(
        self,
        root="data/",
        train=True,
        transform=None,
        index_path=None,
        index=None,
        base_sess=False,
    ):
        self.root = os.path.expanduser(root)
        self.train = train

        # 1. 画像変換 (データ拡張) の設定
        if transform:
            self.transform = transform
        else:
            if train:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        # 2. データのロードとセッションフィルタリング
        self._load_data()

        if base_sess:
            # ベースクラス (index に含まれるクラスのみを抽出)
            self.data, self.targets = self._select_classes(
                self.data, self.targets, index
            )
        elif index_path is not None:
            # インクリメンタルセッション (txt に記載されたサンプルを抽出)
            self.data, self.targets = self._select_from_txt(index_path)
        elif index is not None:
            # 評価用 (現在までに登場した全クラスのテスト画像)
            self.data, self.targets = self._select_classes(
                self.data, self.targets, index
            )

    def _load_data(self):
        # 全画像パスとラベルを読み込む処理
        # self.data = [...]
        # self.targets = [...]
        pass

    def _select_classes(self, data, targets, class_indices):
        mask = np.isin(targets, class_indices)
        return np.array(data)[mask], np.array(targets)[mask]

    def _select_from_txt(self, txt_path):
        # txt ファイルから対象サンプルをロードする処理
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
```

---

### Step 2.3: `dataloader/data_utils.py` への登録

`dataloader/data_utils.py` に新規データセットのメタ情報とローダー初期化ロジックを追加します。

#### ① `set_up_datasets(args)` に設定を追加
```python
# dataloader/data_utils.py の set_up_datasets 内に追加
    if args.dataset == 'my_dataset':
        import dataloader.my_dataset.my_dataset as Dataset
        args.base_class = 60       # ベースクラス数
        args.num_classes = 100     # 全クラス数
        args.way = 5               # セッションごとの新規クラス数
        args.shot = 5              # 1クラスあたりのサンプル数 (Few-shot)
        args.sessions = 9          # 総セッション数 (1 + (100-60)/5 = 9)
```

#### ② `get_base_dataloader(args)` にローダー生成を追加
```python
# dataloader/data_utils.py の get_base_dataloader 内に追加
    if args.dataset == 'my_dataset':
        trainset = args.Dataset.MyDataset(root=args.dataroot, train=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.MyDataset(root=args.dataroot, train=False, index=class_index)
```

#### ③ `get_new_dataloader(args, session)` にローダー生成を追加
```python
# dataloader/data_utils.py の get_new_dataloader 内に追加
    if args.dataset == 'my_dataset':
        trainset = args.Dataset.MyDataset(root=args.dataroot, train=True,
                                         index_path=txt_path)
        testset = args.Dataset.MyDataset(root=args.dataroot, train=False,
                                        index=class_new)
```

---

### Step 2.4: `train.py` の引数選択肢への追加

`train.py` の `get_command_line_parser()` で、`-dataset` の choices に追加します：

```python
# train.py
parser.add_argument(
    "-dataset",
    type=str,
    default="cub200",
    choices=["mini_imagenet", "cub200", "cifar100", "my_dataset"],
)
```

---

### Step 2.5: `models/fact/Network.py` のデータセット分岐設定

データセットの画像サイズや性質に応じて、使用するエンコーダと特徴量次元を `models/fact/Network.py` に指定します：

```python
# models/fact/Network.py の MYNET.__init__
if self.args.dataset in ["cifar100", "manyshotcifar"]:
    self.encoder = resnet20()
    self.num_features = 64
if self.args.dataset in ["mini_imagenet", "my_dataset"]:
    self.encoder = resnet18(False, args)  # pretrained=False
    self.num_features = 512
```

---

## 3. 新しいバックボーンの追加手順

### Step 3.1: バックボーンモデルの実装要件 (FACTの重要制約)

> [!IMPORTANT]
> **FACT 特有の最重要制約: 浅い層と深い層の分割処理**
> FACT ではベースセッションの学習時（`helper.py` の `base_train`）、中間特徴マップ同士を Mixup して未知クラスの疑似インスタンスを合成します。
> したがって、バックボーンは通常の `forward()` だけではなく、以下の2つのフェーズに分割して呼び出せる構造を持つ必要があります：
> 1. **`pre_encode(x)`**: 入力画像 $\to$ 浅い畳み込み層群 $\to$ 中間特徴マップ
> 2. **`post_encode(feat)`**: 中間特徴マップ $\to$ 深い畳み込み層群 $\to$ Global Pooling $\to$ 特徴ベクトル $\to$ 分類器

```
入力画像 x  ──>  [ pre_encode ]  ──> 中間特徴マップ feat1 ──┐
                                                           ├─> Mixup: beta*feat1 + (1-beta)*feat2
別画像 x'   ──>  [ pre_encode ]  ──> 中間特徴マップ feat2 ──┘       │
                                                                   ▼
                                                          [ post_encode ]
                                                                   │
                                                                   ▼
                                                          疑似インスタンス特徴量
```

---

### Step 3.2: `models/` への新規バックボーン実装

例として `models/resnet50_encoder.py` を作成する場合の設計：

```python
# models/resnet50_encoder.py
import torch
import torch.nn as nn
from torchvision.models import resnet50


class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)

        # 浅い層 (pre_encode 用)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2

        # 深い層 (post_encode 用)
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.out_features = 2048  # 出力特徴量次元

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet50_backbone(pretrained=False):
    return ResNet50Encoder(pretrained=pretrained)
```

---

### Step 3.3: `models/fact/Network.py` への統合

#### ① インポートと初期化
```python
# models/fact/Network.py
from models.resnet50_encoder import resnet50_backbone


class MYNET(nn.Module):
    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        # 新しいバックボーンの選択ロジック
        if getattr(args, "backbone", "resnet18") == "resnet50":
            self.encoder = resnet50_backbone(pretrained=False)
            self.num_features = self.encoder.out_features  # 2048
        elif self.args.dataset in ["cifar100"]:
            self.encoder = resnet20()
            self.num_features = 64
        else:
            self.encoder = resnet18(False, args)
            self.num_features = 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分類器重み（全クラス分直交初期化）は自動的に self.num_features に適応
        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        nn.init.orthogonal_(self.fc.weight)

        self.dummy_orthogonal_classifier = nn.Linear(
            self.num_features, self.pre_allocate - self.args.base_class, bias=False
        )
        self.dummy_orthogonal_classifier.weight.requires_grad = False
        self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[
            self.args.base_class :, :
        ]
```

#### ② `pre_encode` の対応
```python
def pre_encode(self, x):
    if getattr(self.args, "backbone", "resnet18") == "resnet50":
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        return x
    # 既存の cifar / resnet18 処理...
```

#### ③ `post_encode` の対応
```python
def post_encode(self, x):
    if getattr(self.args, "backbone", "resnet18") == "resnet50":
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)

        # コサイン類似度計算
        if "cos" in self.mode:
            x = F.linear(
                F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1)
            )
            x = self.args.temperature * x
        elif "dot" in self.mode:
            x = self.fc(x)
            x = self.args.temperature * x
        return x
    # 既存の cifar / resnet18 処理...
```

---

### Step 3.4: `models/base/Network.py` への統合

ベースライン手法（`models/base`）でも同じバックボーンで比較検証できるよう、`models/base/Network.py` にも同様に初期化と `self.num_features` の設定を追加します。ベースラインには `pre_encode` / `post_encode` は不要で、`encode(x)` のみで動作します。

---

## 4. 完全な実装例 (End-to-End Walkthrough)

### 例: 新しいバックボーンで実行する場合のコマンド

`train.py` に `-backbone` 引数を追加した場合の実行例：

```bash
python train.py -project fact -dataset cub200 -backbone resnet50 \
    -base_mode ft_cos -new_mode avg_cos \
    -gamma 0.25 -lr_base 0.005 -lr_new 0.1 -epochs_base 400 \
    -temperature 16 -balance 0.01 -loss_iter 0
```

---

## 5. トラブルシューティング & 実装時の注意点

| 現象・エラー | 原因 | 対処法 |
| :--- | :--- | :--- |
| **`RuntimeError: shape mismatch in linear`** | バックボーンの出力特徴量次元と `self.fc` の入力次元が不一致 | `self.num_features` をバックボーンの最終出力チャネル数（例: 64, 512, 2048）と完全に一致させてください。 |
| **ベース学習時 `post_encode` で次元エラー** | `pre_encode` の出力特徴マップ形状が後続層の期待する入力と不一致 | 中間層の切り出しポイント（例: `layer2` の後）が `layer3` の入力チャネル数と合致しているか確認してください。 |
| **Session 1 以降の精度が極端に低い** | プロトタイプ更新時に特徴量正規化が崩れている | コサイン分類器（`avg_cos`）を使用する場合、プロトタイプ平均計算時および推論時に `F.normalize(..., p=2, dim=-1)` が適切に適用されているか確認してください。 |
| **メモリ不足 (OOM)** | バッチサイズやバックボーンが大きすぎる | `-batch_size_base` を小さくするか、GPU数を増やす、または中間特徴量テンソルの勾配保持に不要な `.detach()` が漏れていないか確認してください。 |
