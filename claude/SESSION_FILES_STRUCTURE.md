# セッションファイル構造の説明

## ディレクトリ構造

```
data/index_list/CICIDS2017_improved/
├── session_0.txt              # ベースセッション（session 0）専用
└── new_sessions/              # 新規セッション（session 1+）専用ディレクトリ
    ├── session_1.txt
    ├── session_2.txt
    ├── session_3.txt
    └── ...
```

## 設計の理由

### 問題点（以前の構造）
- すべてのセッションファイルが同じディレクトリに混在
- DVCが個別ファイル（`session_0.txt`）とディレクトリ全体を同時に追跡しようとしてオーバーラップエラーが発生
- 新規セッションファイルの数が変わるたびに`dvc.yaml`を手動で更新する必要があった

### 解決策（新しい構造）
1. **ベースセッションファイルの分離**: `session_0.txt`は親ディレクトリに配置
2. **新規セッション専用ディレクトリ**: `new_sessions/`ディレクトリにすべての新規セッションファイルを格納
3. **DVCの依存関係の簡素化**:
   - `create_base_session` → `session_0.txt`のみ出力
   - `create_new_sessions` → `new_sessions/`ディレクトリ全体を出力
   - `train_new` → `new_sessions/`ディレクトリ全体に依存

## 利点

### 1. DVCオーバーラップエラーの回避
```
# エラー（以前）:
ERROR: The output paths overlap:
'data/index_list/CICIDS2017_improved' (create_session_files)
'data/index_list/CICIDS2017_improved/session_0.txt' (create_session_files)

# 解決（現在）:
- session_0.txt: create_base_session が管理
- new_sessions/: create_new_sessions が管理
```

### 2. セッション数の動的対応
- `new_sessions/`ディレクトリに何個ファイルが追加されてもDVCは自動的に追跡
- `dvc.yaml`で個別ファイル（`session_1.txt`, `session_2.txt`, ...）を列挙する必要がない
- セッション数が変わってもDVC設定の変更不要

### 3. 明確な責任分離
| ステージ | 責任 | 出力 |
|---------|-----|------|
| `create_base_session` | ベースセッションファイルのみ生成 | `session_0.txt` |
| `create_new_sessions` | 新規セッションファイルのみ生成 | `new_sessions/` |
| `train_base` | ベースセッションのトレーニング | - |
| `train_new` | 新規セッションのトレーニング | - |

## 実装詳細

### create_session_files.py
```python
def create_session_files(..., new_sessions_dir="new_sessions"):
    # 新規セッション用のサブディレクトリを作成
    new_sessions_path = os.path.join(output_dir, new_sessions_dir)
    os.makedirs(new_sessions_path, exist_ok=True)

    # ベースセッション: 親ディレクトリに保存
    session_0_path = os.path.join(output_dir, "session_0.txt")

    # 新規セッション: new_sessions/ に保存
    session_path = os.path.join(new_sessions_path, f"session_{session_num}.txt")
```

### dataloader/data_utils.py
```python
def get_base_dataloader_meta(args):
    # ベースセッションは親ディレクトリから読み込み
    txt_path = os.path.join(..., "session_0.txt")

def get_new_dataloader(args, session):
    # 新規セッションは new_sessions/ から読み込み
    txt_path = os.path.join(..., "new_sessions/session_" + str(session) + ".txt")
```

### dvc.yaml
```yaml
create_new_sessions:
  outs:
    - data/index_list/CICIDS2017_improved/new_sessions  # ディレクトリ全体

train_new:
  deps:
    - data/index_list/CICIDS2017_improved/new_sessions  # ディレクトリ全体に依存
```

## パラメータ分離との統合

この構造は、パラメータ分離戦略と完璧に統合されています：

| パラメータ変更 | 影響を受けるファイル | 再実行されるステージ |
|-------------|-------------------|-------------------|
| `create_sessions.base_class` | `session_0.txt` | `create_base_session` → `train_base` |
| `create_sessions.way` / `shot` | `new_sessions/*` | `create_new_sessions` → `train_new` |
| `train.base.*` | なし | `train_base`のみ |
| `train.new.*` | なし | `train_new`のみ |

## セッション数の追加方法

新しいクラスを追加してセッション数を増やす場合：

1. **`params.yaml`を更新**:
   ```yaml
   create_sessions:
     num_classes: 15  # 例: 10 → 15 に増加
     way: 2
   ```

2. **パイプラインを再実行**:
   ```bash
   dvc repro create_new_sessions
   ```

3. **自動的に対応**:
   - `new_sessions/`に新しいセッションファイルが追加される
   - `train_new`が自動的に新しいセッションを検出して処理
   - **`dvc.yaml`の変更は不要**

## トラブルシューティング

### 問題: 古いセッションファイルが残っている
```bash
# 親ディレクトリの古い session_*.txt (session_0.txt以外) を削除
cd data/index_list/CICIDS2017_improved/
ls session_*.txt | grep -v session_0.txt | xargs rm -f
```

### 問題: new_sessions/ ディレクトリが見つからない
```bash
# create_new_sessions ステージを再実行
dvc repro create_new_sessions
```

### 問題: dataloaderがファイルを見つけられない
- `dataloader/data_utils.py`の`get_new_dataloader()`が正しく`new_sessions/`を参照しているか確認
- セッションファイルが実際に存在するか確認:
  ```bash
  ls -la data/index_list/CICIDS2017_improved/new_sessions/
  ```

## 関連ドキュメント

- **パラメータ分離**: [DVC_PARAMS_SEPARATION.md](DVC_PARAMS_SEPARATION.md)
- **実装サマリー**: [DVC_IMPLEMENTATION_SUMMARY.md](DVC_IMPLEMENTATION_SUMMARY.md)
- **変更履歴**: [CHANGELOG_DVC.md](CHANGELOG_DVC.md)
