# DVCパラメータ分離の説明

## 問題点

以前は、`train_base`と`train_new`の両方が`params.train`全体に依存していたため、新規セッション専用のパラメータ（`epochs_new`など）を変更しても、ベースセッション（`train_base`）が不必要に再実行されてしまう問題がありました。

## 解決策

### 1. params.yamlの構造変更

`params.yaml`の`train`セクションを3つのサブセクションに分離しました：

```yaml
train:
  # 両方のセッションで使用される共通パラメータ
  common:
    dataset: CICIDS2017_improved
    encoder: cnn1d
    project: fact
    # ...

  # ベースセッション（session 0）専用パラメータ
  base:
    epochs_base: 10
    lr_base: 0.005
    batch_size_base: 128
    # ...

  # 新規セッション（session 1+）専用パラメータ
  new:
    epochs_new: 10000
    lr_new: 0.01
    new_mode: avg_cos
    # ...
```

### 2. dvc.yamlの依存関係の明確化とステージ分離

#### セッションファイル生成の分離

**create_base_session** (新規):
- コマンド: `uv run create_base_session_file.py`
- パラメータ: `create_sessions.base_class`, `create_sessions.train_csv`, `create_sessions.output_dir`のみ
- 出力: `session_0.txt`のみ

**create_new_sessions** (旧 create_session_files):
- コマンド: `uv run create_session_files.py`
- パラメータ: `create_sessions`（全体）, `common.seed`
- 出力: `session_1.txt`〜`session_6.txt`, `class_distribution.txt`, `column_names.txt`

#### トレーニングステージ

**train_base**:
- パラメータ: `train.common`, `train.base`のみ
- 依存: `session_0.txt`のみ

**train_new**:
- パラメータ: `train.common`, `train.new`のみ
- 依存: `session_1.txt`〜`session_6.txt`
- foreachを削除し、単一ステージで全セッションを処理

#### 主な改善点

1. **セッションファイル生成の分離**: ベースセッションと新規セッションのファイル生成を完全に分離
2. **パラメータの最小化**: 各ステージが本当に必要なパラメータのみに依存
3. **ファイルレベルの依存**: ディレクトリではなく個別ファイルに依存することでDVCのオーバーラップエラーを回避

### 3. train.pyの後方互換性

`train.py`の`load_defaults_from_yaml()`関数に後方互換性を追加：
- 新しい構造（`train.common`、`train.base`、`train.new`）をサポート
- 古いフラットな構造（`train.*`）も引き続きサポート

## 効果

### 変更前
- `train.epochs_new`を変更 → `train_base`も`train_new@1`, `train_new@2`, ... すべて再実行
- セッション数の変更 → `dvc.yaml`の`foreach`を手動で更新する必要がある

### 変更後
- `train.new.epochs_new`を変更 → `train_new`のみ再実行（`train_base`は再実行されない）
- `train.base.epochs_base`を変更 → `train_base`のみ再実行
- `train.common.encoder`を変更 → 両方とも再実行（正しい挙動）
- セッション数の変更 → `train_new_sessions.py`が自動的に検出して処理（`dvc.yaml`の変更不要）

## テストケース

### ケース1: 新規セッションパラメータのみ変更
```bash
# params.yamlで train.new.epochs_new を 10000 → 20000 に変更
dvc status
# 期待結果: train_new のみが changed と表示される
```

### ケース2: ベースセッションパラメータのみ変更
```bash
# params.yamlで train.base.epochs_base を 10 → 20 に変更
dvc status
# 期待結果: train_base のみが changed と表示される
```

### ケース3: 共通パラメータの変更
```bash
# params.yamlで train.common.encoder を cnn1d → mlp に変更
dvc status
# 期待結果: train_base と train_new の両方が changed と表示される
```

### ケース4: create_sessionsのway/shotのみ変更
```bash
# params.yamlで create_sessions.way を 1 → 2 に変更
dvc status
# 期待結果: create_session_files と train_new が changed、train_base は変更なし
```

## マイグレーション手順

既存の環境から新しい構造に移行する場合：

1. パラメータをバックアップ
2. `params.yaml`を新しい構造に更新（完了済み）
3. `dvc.yaml`を更新（完了済み）
4. DVCキャッシュをクリア（オプション）:
   ```bash
   dvc cache dir  # キャッシュディレクトリを確認
   dvc gc --workspace  # 未使用キャッシュを削除
   ```
5. パイプラインを再実行:
   ```bash
   dvc repro
   ```
