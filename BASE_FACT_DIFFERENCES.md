# Base vs FACT の違い

このドキュメントでは、BaseモデルとFACTモデルの主な違いを説明します。

## 概要

- **Base**: 標準的なFew-Shot Class-Incremental Learningの実装
- **FACT**: Forward Compatible Training（前方互換性を考慮した学習方法）
  - 将来の新しいクラスのために埋め込み空間を予約
  - 前方互換性を実現するための追加機能

---

## 1. Network.py の違い

### 1.1 初期化の違い

#### Base版
```python
self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
```
- 標準的なFC層のみ
- デフォルトのPyTorch初期化

#### FACT版
```python
self.pre_allocate = self.args.num_classes
self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
nn.init.orthogonal_(self.fc.weight)  # 直交初期化

# ダミー分類器の追加
self.dummy_orthogonal_classifier = nn.Linear(
    self.num_features, 
    self.pre_allocate - self.args.base_class, 
    bias=False
)
self.dummy_orthogonal_classifier.weight.requires_grad = False
self.dummy_orthogonal_classifier.weight.data = self.fc.weight.data[self.args.base_class:,:]
```
- **ダミー分類器（`dummy_orthogonal_classifier`）を追加**
- FC層の重みを直交初期化
- 将来クラス用の埋め込み空間を予約

### 1.2 forward_metric() の違い

#### Base版
```python
def forward_metric(self, x):
    x = self.encode(x)
    if 'cos' in self.mode:
        x = F.linear(F.normalize(x, p=2, dim=-1), 
                     F.normalize(self.fc.weight, p=2, dim=-1))
        x = self.args.temperature * x
    elif 'dot' in self.mode:
        x = self.fc(x)
        x = self.args.temperature * x
    return x
```
- 標準的なFC層のみを使用

#### FACT版
```python
def forward_metric(self, x):
    x = self.encode(x)
    if 'cos' in self.mode:
        # 通常のFC層とダミー分類器の両方を使用
        x1 = F.linear(F.normalize(x, p=2, dim=-1), 
                     F.normalize(self.fc.weight, p=2, dim=-1))
        x2 = F.linear(F.normalize(x, p=2, dim=-1), 
                     F.normalize(self.dummy_orthogonal_classifier.weight, p=2, dim=-1))
        # 連結: ベースクラス + 将来クラス
        x = torch.cat([x1[:,:self.args.base_class], x2], dim=1)
        x = self.args.temperature * x
    elif 'dot' in self.mode:
        x = self.fc(x)
        x = self.args.temperature * x
    return x
```
- **通常のFC層とダミー分類器を組み合わせて使用**
- ベースクラスは通常のFC層、将来クラスはダミー分類器で予測

### 1.3 FACT版の追加メソッド

#### pre_encode() と post_encode()
```python
def pre_encode(self, x):
    # エンコーダの前半部分のみ（layer1, layer2まで）
    # Mixupで使用
    
def post_encode(self, x):
    # エンコーダの後半部分（layer3, layer4）
    # その後、FC層を適用
```
- **エンコーダを分割**して、Mixupによる拡張学習を実現
- Base版には存在しない

#### forpass_fc()
```python
def forpass_fc(self, x):
    # 通常のFC層のみを使用（ダミー分類器なし）
    # セッション1以降の評価で使用
```
- ダミー分類器を使わない順伝播
- セッション1以降の評価時に使用

---

## 2. helper.py (base_train) の違い

### 2.1 Base版のbase_train()

```python
def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.to(device) for _ in batch]
        
        logits = model(data)
        logits = logits[:, :args.base_class]
        loss = F.cross_entropy(logits, train_label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
- **シンプルなCross Entropy Lossのみ**
- 標準的な分類学習

### 2.2 FACT版のbase_train()

```python
def base_train(model, trainloader, optimizer, scheduler, epoch, args, mask, device=None):
    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.to(device) for _ in batch]
        
        logits = model(data)
        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)  # 基本損失
        
        if epoch >= args.loss_iter:
            # 1. 疑似ラベル生成（マスク使用）
            logits_masked = logits.masked_fill(...)
            logits_masked_chosen = logits_masked * mask[train_label]
            pseudo_label = torch.argmax(...) + args.base_class
            loss2 = F.cross_entropy(logits_masked, pseudo_label)
            
            # 2. Mixupによる拡張
            index = torch.randperm(data.size(0)).to(device)
            pre_emb1 = model.pre_encode(data)
            mixed_data = beta * pre_emb1 + (1-beta) * pre_emb1[index]
            mixed_logits = model.post_encode(mixed_data)
            
            # 3. Mixup後の疑似ラベル生成
            pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], ...)
            pseudo_label2 = torch.argmax(mixed_logits[:,:args.base_class], ...)
            loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
            loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
            
            # 複合損失
            total_loss = loss + args.balance * (loss2 + loss3 + loss4)
        else:
            total_loss = loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

**FACT版の追加機能:**
1. **マスクによる疑似ラベル生成**
   - 将来クラスへの疑似ラベルを生成
   - `loss2`: 疑似ラベルとの損失

2. **Mixupによるデータ拡張**
   - `pre_encode()`と`post_encode()`を使用
   - エンコーダの中間でMixupを実行
   - `loss3`, `loss4`: Mixup後の損失

3. **複合損失**
   - `total_loss = loss + balance * (loss2 + loss3 + loss4)`
   - 基本損失 + 前方互換性損失

---

## 3. fscil_trainer.py の違い

### 3.1 セッション0の処理

#### Base版
- 標準的な学習ループ
- エポック終了後にプロトタイプ更新

#### FACT版
```python
# マスク生成（セッション0開始前）
masknum = 3
mask = np.zeros((args.base_class, args.num_classes))
for i in range(args.num_classes - args.base_class):
    picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
    mask[:, i+args.base_class][picked_dummy] = 1

# base_trainにmaskを渡す
tl, ta = base_train(..., mask, self.device)

# セッション0終了後、ダミー分類器を保存
self.dummy_classifiers = deepcopy(self.model.module.fc.weight.detach())
self.dummy_classifiers = F.normalize(
    self.dummy_classifiers[self.args.base_class:,:], p=2, dim=-1
)
```
- **マスク生成**: 将来クラス用のマスクを事前に生成
- **ダミー分類器の保存**: セッション0終了時に保存

### 3.2 セッション1以降の評価

#### Base版
```python
# 標準的なテスト
tsl, tsa = test(self.model, testloader, 0, args, session, validation=False)
```

#### FACT版
```python
# 統合評価（test_intergrate）を使用
tsl, tsa = self.test_intergrate(self.model, testloader, 0, args, session, validation=True)
```

### 3.3 test_intergrate() メソッド（FACT版のみ）

```python
def test_intergrate(self, model, testloader, epoch, args, session, validation=True):
    # 1. 埋め込みを取得
    emb = model.encode(data)
    
    # 2. ダミー分類器との投影
    proj = torch.mm(F.normalize(emb, p=2, dim=-1), 
                    torch.transpose(self.dummy_classifiers, 1, 0))
    
    # 3. Top-K選択
    k = min(40, proj.size(1))
    topk, indices = torch.topk(proj, k)
    
    # 4. 投影行列の計算
    proj_matrix = torch.mm(self.dummy_classifiers, 
                           F.normalize(torch.transpose(fc_weight, 1, 0), p=2, dim=-1))
    
    # 5. 2つのロジットを統合
    logits1 = torch.mm(res_logit, proj_matrix)  # ダミー分類器経由
    logits2 = model.forpass_fc(data)[:, :test_class]  # 通常のFC層
    
    # 6. 重み付き平均
    logits = eta * F.softmax(logits1, dim=1) + (1-eta) * F.softmax(logits2, dim=1)
```
- **2つの予測を統合**: ダミー分類器経由の予測と通常のFC層の予測
- **前方互換性の実現**: セッション0で学習したダミー分類器を活用

---

## 4. 主な概念の違い

### 4.1 Base版
- **後方互換性**: 過去のクラスを忘れない
- **標準的なFew-Shot Learning**: プロトタイプベースの分類
- **シンプルな実装**: 追加の仕組みなし

### 4.2 FACT版
- **前方互換性**: 将来のクラスに備える
- **Forward Compatible Training**: 
  - 将来クラス用の埋め込み空間を予約
  - ダミー分類器で将来クラスを予測
  - マスクとMixupで前方互換性を学習
- **統合評価**: 複数の予測を組み合わせ

---

## 5. 学習パラメータの違い

### Base版で使用されるパラメータ
- `epochs_base`: ベースセッションのエポック数
- `lr_base`: ベースセッションの学習率
- `base_mode`: ベースセッションのモード（'ft_cos' or 'ft_dot'）
- `new_mode`: 新規セッションのモード（'avg_cos', 'ft_cos', 'ft_dot'）

### FACT版で追加されるパラメータ
- `balance`: 複合損失のバランス係数（デフォルト: 1.0）
- `loss_iter`: 追加損失を開始するエポック（デフォルト: 200）
- `alpha`: MixupのBeta分布パラメータ（デフォルト: 2.0）
- `eta`: 統合評価時の重み（デフォルト: 0.1）

---

## 6. まとめ

| 項目 | Base版 | FACT版 |
|------|--------|--------|
| **目的** | 標準的なFew-Shot Learning | 前方互換性を考慮したFew-Shot Learning |
| **FC層** | 標準的なFC層のみ | FC層 + ダミー分類器 |
| **学習方法** | Cross Entropy Lossのみ | 複合損失（マスク + Mixup） |
| **エンコーダ** | 通常のエンコーダ | 分割可能（pre_encode/post_encode） |
| **セッション0** | 標準学習 | マスク生成 + ダミー分類器保存 |
| **セッション1以降** | 標準評価 | 統合評価（test_intergrate） |
| **前方互換性** | ❌ | ✅ |

FACT版は、将来の新しいクラスに備えて埋め込み空間を予約し、前方互換性を実現することで、Few-Shot Class-Incremental Learningの性能を向上させます。

