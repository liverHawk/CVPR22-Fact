# DVC Pipeline 変更履歴

## 2025-11-27: パラメータ分離とforeachの削除

### 変更の目的
1. **不必要な再実行の防止**: 新規セッション専用パラメータの変更時に、ベースセッションが再実行されないようにする
2. **柔軟なセッション管理**: セッション数の変更に応じて自動的に対応する

### 主な変更

#### 1. `params.yaml`の構造変更
```yaml
# 変更前（フラット構造）
train:
  epochs_base: 10
  epochs_new: 10000
  encoder: cnn1d
  # ... すべてのパラメータが同じレベル

# 変更後（階層構造）
train:
  common:    # 共通パラメータ
    encoder: cnn1d
    dataset: CICIDS2017_improved
  base:      # ベースセッション専用
    epochs_base: 10
    lr_base: 0.005
  new:       # 新規セッション専用
    epochs_new: 10000
    lr_new: 0.01
```

#### 2. `dvc.yaml`の変更
- **セッションファイル生成の分離**:
  ```yaml
  # 新規: ベースセッション専用
  create_base_session:
    cmd: uv run create_base_session_file.py
    params:
      - create_sessions.base_class
      - create_sessions.train_csv
      - create_sessions.output_dir
    outs:
      - session_0.txt

  # 変更: 新規セッション専用
  create_new_sessions:
    cmd: uv run create_session_files.py
    params:
      - create_sessions  # 全体
      - common.seed
    outs:
      - session_1.txt ~ session_6.txt
  ```

- **パラメータ依存関係の明確化**:
  - `train_base`: `train.common`, `train.base`のみ
  - `train_new`: `train.common`, `train.new`のみ

- **foreachの削除**:
  ```yaml
  # 変更前
  train_new:
    foreach: [1, 2, 3, 4, 5, 6]
    do:
      cmd: uv run train_new_sessions.py --select_sessions ${item}

  # 変更後
  train_new:
    cmd: uv run train_new_sessions.py
  ```

- **ファイルレベルの依存**:
  - `train_base`: `session_0.txt`のみに依存
  - `train_new`: `session_1.txt`〜`session_6.txt`に個別に依存
  - ディレクトリ全体への依存を排除してDVCのオーバーラップエラーを回避

#### 3. `train_new_sessions.py`の改善
- 完了マーカーファイル（`new_sessions_complete.txt`）を生成してDVCの出力追跡を可能に
- `select_sessions`未指定時に、すべての新規セッションを自動的に処理

#### 4. `train.py`の後方互換性
- 新しいパラメータ構造と古いフラット構造の両方をサポート
- 既存のコマンドライン引数の動作を維持

### 効果

| パラメータの変更 | 変更前 | 変更後 |
|-----------------|-------|-------|
| `epochs_new` | `train_base`と`train_new`全て再実行 | `train_new`のみ再実行 ✓ |
| `epochs_base` | 両方再実行 | `train_base`のみ再実行 ✓ |
| `encoder` | 両方再実行 | 両方再実行（正しい）✓ |
| セッション数増加 | `dvc.yaml`を手動編集 | 自動対応 ✓ |

### マイグレーション手順

既存のプロジェクトでこの変更を適用する場合：

1. **バックアップ**:
   ```bash
   cp params.yaml params.yaml.backup
   cp dvc.yaml dvc.yaml.backup
   ```

2. **ファイルを更新**（既に完了）:
   - `params.yaml`: 新しい階層構造
   - `dvc.yaml`: ステージの分離とパラメータ依存関係の変更
   - `create_base_session_file.py`: 新規作成（ベースセッション専用）
   - `train.py`: `load_defaults_from_yaml()`の更新
   - `train_new_sessions.py`: マーカーファイルの生成

3. **DVCの状態を確認**:
   ```bash
   dvc status
   ```

4. **パイプラインを再実行**:
   ```bash
   # ベースセッションのみ
   dvc repro train_base

   # 新規セッション
   dvc repro train_new

   # または全体を再実行
   dvc repro
   ```

### 既知の制限事項

- **初回移行時**: すべてのステージが`new params`として検出され、完全な再実行が必要
- **キャッシュ**: 既存のDVCキャッシュは新しい出力形式と互換性がない可能性がある

### トラブルシューティング

**問題**: `dvc repro`でエラーが発生する

**解決策**:
```bash
# DVCロックファイルを削除
rm dvc.lock

# パイプラインを再実行
dvc repro
```

**問題**: パラメータ変更が検出されない

**解決策**:
```bash
# DVCのステータスを詳細表示
dvc status -v

# 強制的に再実行
dvc repro -f <stage_name>
```

### 参考資料

- 詳細な説明: [DVC_PARAMS_SEPARATION.md](DVC_PARAMS_SEPARATION.md)
- DVCドキュメント: https://dvc.org/doc/user-guide/project-structure/pipelines-files
