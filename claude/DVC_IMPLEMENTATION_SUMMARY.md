# DVC パラメータ分離実装 - 最終サマリー

## 実装完了日
2025-11-27

## 解決した問題

**元の問題**: `train_base_session`と`train_new_sessions`で処理を分けているのに、新規セッション専用パラメータ（`epochs_new`、`way`、`shot`など）を変更すると、ベースセッションも不必要に再実行されてしまう。

## 実装した解決策

### 1. パラメータ構造の階層化 ([params.yaml](params.yaml))

```yaml
train:
  common:    # 両セッション共通
    encoder: cnn1d
    dataset: CICIDS2017_improved
    project: fact
    # ...
  
  base:      # ベースセッション専用
    epochs_base: 10
    lr_base: 0.005
    batch_size_base: 128
    # ...
  
  new:       # 新規セッション専用
    epochs_new: 10000
    lr_new: 0.01
    new_mode: avg_cos
    # ...
```

### 2. DVCステージの完全分離 ([dvc.yaml](dvc.yaml))

#### セッションファイル生成の分離

| ステージ | コマンド | パラメータ依存 | 出力 |
|---------|---------|--------------|------|
| `create_base_session` | `create_base_session_file.py` | `create_sessions.{base_class, train_csv, output_dir}` | `session_0.txt` |
| `create_new_sessions` | `create_session_files.py` | `create_sessions` (全体), `common.seed` | `session_1.txt` ~ `session_6.txt` |

#### トレーニングステージ

| ステージ | パラメータ依存 | ファイル依存 |
|---------|--------------|-------------|
| `train_base` | `train.common`, `train.base` | `session_0.txt` |
| `train_new` | `train.common`, `train.new` | `session_1.txt` ~ `session_6.txt`, `session0_max_acc.pth` |

### 3. 新規ファイル

- **[create_base_session_file.py](create_base_session_file.py)**: ベースセッションファイルのみを生成
  - `create_session_files.py`から必要な関数を再利用
  - `base_class`パラメータのみに依存

### 4. 変更ファイル

- **[train.py](train.py:15-109)**: `load_defaults_from_yaml()`を更新
  - 新しい階層構造をサポート
  - 後方互換性を維持（古いフラット構造も動作）

- **[train_new_sessions.py](train_new_sessions.py:18-55)**: 
  - 完了マーカーファイル生成
  - 全セッション自動処理

- **[CLAUDE.md](CLAUDE.md:125-136)**: プロジェクトドキュメント更新

## 効果と動作

### パラメータ変更時の再実行パターン

| 変更するパラメータ | 再実行されるステージ | 理由 |
|------------------|-------------------|------|
| `train.new.epochs_new` | `train_new` のみ | ✅ 新規セッション専用 |
| `train.base.epochs_base` | `train_base` のみ | ✅ ベースセッション専用 |
| `train.common.encoder` | `train_base` と `train_new` | ✅ 両方で使用 |
| `create_sessions.way` | `create_new_sessions` → `train_new` | ✅ 新規セッションのみ影響 |
| `create_sessions.shot` | `create_new_sessions` → `train_new` | ✅ 新規セッションのみ影響 |
| `create_sessions.base_class` | すべて | ✅ ベースクラス数変更のため |

### foreachの削除による利点

**変更前**:
```yaml
train_new:
  foreach: [1, 2, 3, 4, 5, 6]
  do:
    cmd: uv run train_new_sessions.py --select_sessions ${item}
```

**変更後**:
```yaml
train_new:
  cmd: uv run train_new_sessions.py
```

**利点**:
1. セッション数の変更時に`dvc.yaml`を手動編集する必要がない
2. 単一ステージで管理が簡潔
3. `train_new_sessions.py`が自動的に全セッションを処理

## 技術的詳細

### DVCオーバーラップエラーの回避

**問題**: 
```
ERROR: The output paths overlap:
'data/index_list/CICIDS2017_improved' (create_session_files)
'data/index_list/CICIDS2017_improved/session_0.txt' (create_session_files)
```

**解決**:
- ディレクトリ全体ではなく、個別ファイル（`session_0.txt`、`session_1.txt`など）を出力として定義
- ベースセッションと新規セッションのファイル生成を完全に分離

### パラメータ依存の最小化

各ステージが**本当に必要なパラメータのみ**に依存することで：
- 不要な再実行を防止
- 依存関係が明確
- メンテナンスが容易

## 検証方法

```bash
# 1. 新規セッションパラメータの変更
sed -i 's/epochs_new: 10000/epochs_new: 15000/' params.yaml
dvc status  # → train_new のみ changed

# 2. ベースセッションパラメータの変更
sed -i 's/epochs_base: 10/epochs_base: 20/' params.yaml
dvc status  # → train_base のみ changed

# 3. 共通パラメータの変更
sed -i 's/encoder: cnn1d/encoder: mlp/' params.yaml
dvc status  # → train_base と train_new 両方 changed
```

## 次のステップ

1. **初回実行**: `dvc repro` でパイプライン全体を実行し、DVCのロックファイルを更新
2. **パラメータ調整**: 新しい構造でパラメータを変更し、分離が機能することを確認
3. **実験管理**: 新規セッションの実験（`epochs_new`の調整など）がベースセッションに影響しないことを活用

## 参考ドキュメント

- **詳細説明**: [DVC_PARAMS_SEPARATION.md](DVC_PARAMS_SEPARATION.md)
- **変更履歴**: [CHANGELOG_DVC.md](CHANGELOG_DVC.md)
- **プロジェクト概要**: [CLAUDE.md](CLAUDE.md)

## トラブルシューティング

### 問題: パラメータ変更が検出されない

```bash
# DVCロックファイルを削除して再実行
rm dvc.lock
dvc repro
```

### 問題: オーバーラップエラー

現在の実装では解決済み。もし発生する場合は：
- `dvc.yaml`で個別ファイルが正しく指定されているか確認
- ディレクトリ全体の出力指定を削除

### 問題: セッション数の変更

`dvc.yaml`で`session_1.txt`〜`session_6.txt`が固定されているため、セッション数を変更する場合は：
1. `create_new_sessions`の`outs`セクションを更新
2. `train_new`の`deps`セクションを更新

将来的には、これも動的化することが可能です。

---

**実装者**: Claude Code  
**実装日**: 2025-11-27  
**バージョン**: v1.0
