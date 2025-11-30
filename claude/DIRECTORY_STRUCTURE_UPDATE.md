# セッションファイルのディレクトリ構造変更

## 変更日
2025-11-27

## 変更内容

新規セッションファイル（`session_1.txt`以降）を専用ディレクトリ`new_sessions/`に格納する構造に変更しました。

### 以前の構造
```
data/index_list/CICIDS2017_improved/
├── session_0.txt
├── session_1.txt
├── session_2.txt
├── session_3.txt
├── session_4.txt
├── session_5.txt
└── session_6.txt
```

**問題点**:
- すべてのファイルが同じディレクトリに混在
- DVCで個別ファイルとディレクトリ全体を同時に追跡できない（オーバーラップエラー）
- セッション数が変わるたびに`dvc.yaml`を手動更新

### 新しい構造
```
data/index_list/CICIDS2017_improved/
├── session_0.txt              # ベースセッション専用
└── new_sessions/              # 新規セッション専用ディレクトリ
    ├── session_1.txt
    ├── session_2.txt
    └── session_3.txt
```

**利点**:
- ✅ ベースセッションと新規セッションを明確に分離
- ✅ DVCがディレクトリ全体を追跡（個別ファイルの列挙不要）
- ✅ セッション数の変更に自動対応

## 変更されたファイル

### 1. [create_session_files.py](create_session_files.py:93-242)
```python
def create_session_files(..., new_sessions_dir="new_sessions"):
    # 新規セッション用のサブディレクトリを作成
    new_sessions_path = os.path.join(output_dir, new_sessions_dir)
    os.makedirs(new_sessions_path, exist_ok=True)
    
    # ベースセッション: 親ディレクトリに保存
    session_0_path = os.path.join(output_dir, "session_0.txt")
    
    # 新規セッション: new_sessions/ ディレクトリに保存
    session_path = os.path.join(new_sessions_path, f"session_{session_num}.txt")
```

### 2. [dataloader/data_utils.py](dataloader/data_utils.py:241-242)
```python
def get_new_dataloader(args, session):
    # 新規セッションは new_sessions/ から読み込む
    txt_path = os.path.join(
        os.path.dirname(args.dataroot),
        "data/index_list/" + args.dataset + "/new_sessions/session_" + str(session) + ".txt"
    )
```

### 3. [dvc.yaml](dvc.yaml:25-36,53-65)
```yaml
create_new_sessions:
  outs:
    - data/index_list/CICIDS2017_improved/new_sessions  # ディレクトリ全体

train_new:
  deps:
    - data/index_list/CICIDS2017_improved/new_sessions  # ディレクトリ全体に依存
```

**変更前**（個別ファイルを列挙）:
```yaml
create_new_sessions:
  outs:
    - data/index_list/CICIDS2017_improved/session_1.txt
    - data/index_list/CICIDS2017_improved/session_2.txt
    - data/index_list/CICIDS2017_improved/session_3.txt
    # ... セッション数が増えるたびに追加が必要
```

## パラメータ分離との統合

この構造変更により、パラメータ分離がより効果的に機能します：

| パラメータ変更 | 影響ファイル | 再実行ステージ |
|-------------|------------|--------------|
| `create_sessions.way` | `new_sessions/*` | `create_new_sessions` → `train_new` |
| `create_sessions.shot` | `new_sessions/*` | `create_new_sessions` → `train_new` |
| `create_sessions.base_class` | `session_0.txt` + `new_sessions/*` | 全ステージ（正しい） |
| `train.new.epochs_new` | なし | `train_new`のみ |
| `train.base.epochs_base` | なし | `train_base`のみ |

## DVCの動作

### ディレクトリ追跡の利点

```bash
# セッション数を増やす
# params.yaml: num_classes を 10 → 15 に変更

# パイプラインを再実行
dvc repro create_new_sessions

# 自動的に:
# - new_sessions/session_4.txt が追加される
# - DVCが自動的に追跡（dvc.yaml の変更不要）
# - train_new が新しいセッションを検出
```

### 完全な分離

```
create_base_session  →  session_0.txt  →  train_base
       ↓
prepare_data
       ↓
create_new_sessions  →  new_sessions/  →  train_new
```

- `session_0.txt`の変更 → `train_base`のみ再実行
- `new_sessions/`の変更 → `train_new`のみ再実行
- オーバーラップなし、明確な責任分離

## マイグレーション

既存のプロジェクトで古い構造から移行する場合：

```bash
# 1. セッションファイルを再生成
dvc repro create_new_sessions

# 2. 古いファイルを削除（オプション）
cd data/index_list/CICIDS2017_improved/
rm -f session_1.txt session_2.txt session_3.txt session_4.txt session_5.txt session_6.txt

# 3. 新しい構造を確認
find . -name "*.txt" | sort
# 出力:
# ./session_0.txt
# ./new_sessions/session_1.txt
# ./new_sessions/session_2.txt
# ./new_sessions/session_3.txt
```

## 関連ドキュメント

- **詳細説明**: [SESSION_FILES_STRUCTURE.md](SESSION_FILES_STRUCTURE.md)
- **プロジェクト概要**: [CLAUDE.md](CLAUDE.md)

## テスト

```bash
# DVCの状態確認
dvc status

# DAG表示
dvc dag

# セッションファイルの確認
find data/index_list/CICIDS2017_improved/ -name "*.txt" | sort
```

---

**実装者**: Claude Code  
**実装日**: 2025-11-27  
**バージョン**: v2.0 (ディレクトリ構造変更)
