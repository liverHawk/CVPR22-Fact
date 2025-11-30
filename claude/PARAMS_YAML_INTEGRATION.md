# params.yaml統合の修正

## 問題

`dataloader/data_utils.py`でCICIDS2017のためにハードコードされた設定（`way=1`、`sessions=7`）が、`params.yaml`の設定（`way=2`）を上書きしていました。

### エラー内容

```
FileNotFoundError: Index file not found:
/home/hawk/Documents/school/test/CVPR22-Fact/data/index_list/CICIDS2017_improved/new_sessions/session_4.txt
```

### 原因

1. **`params.yaml`の設定**:
   - `base_class: 4`
   - `num_classes: 10`
   - `way: 2`
   - 期待されるセッション数: `(10-4)/2 + 1 = 4` (session 0-3)

2. **`data_utils.py`のハードコード**:
   - `args.way = 1` (上書き)
   - `args.sessions = 7` (固定値)
   - トレーナーが期待するセッション数: 7 (session 0-6)

3. **結果**:
   - 実際に生成されたファイル: session_0.txt, session_1.txt, session_2.txt, session_3.txt (4つ)
   - トレーナーが探すファイル: session_0.txt ~ session_6.txt (7つ)
   - → session_4.txt以降が見つからずエラー

## 解決策

`dataloader/data_utils.py`を修正して、`params.yaml`の設定を優先的に使用するように変更しました。

### 修正内容

**変更前** ([dataloader/data_utils.py](../dataloader/data_utils.py)):
```python
if args.dataset == "CICIDS2017_improved":
    args.base_class = 4  # ハードコード
    args.num_classes = 10  # ハードコード
    args.way = 1  # ハードコード - params.yamlを上書き!
    if args.shot is None:
        args.shot = 5
    args.sessions = 7  # ハードコード
```

**変更後**:
```python
if args.dataset == "CICIDS2017_improved":
    # Load from params.yaml if not set via command line
    from utils import load_params_yaml
    try:
        params = load_params_yaml("params.yaml")
        create_sessions = params.get("create_sessions", {})

        if not hasattr(args, 'base_class') or args.base_class is None:
            args.base_class = create_sessions.get("base_class", 4)
        if not hasattr(args, 'num_classes') or args.num_classes is None:
            args.num_classes = create_sessions.get("num_classes", 10)
        if not hasattr(args, 'way') or args.way is None:
            args.way = create_sessions.get("way", 1)
        if not hasattr(args, 'shot') or args.shot is None:
            args.shot = create_sessions.get("shot", 5)
    except:
        # Fallback to defaults if params.yaml loading fails
        # (デフォルト値の設定)

    # Calculate sessions based on configuration
    args.sessions = (args.num_classes - args.base_class) // args.way + 1
```

### 変更のポイント

1. **`params.yaml`を優先**: `create_sessions`セクションから設定を読み込み
2. **動的なセッション数計算**: `args.sessions`をハードコードせず、計算で求める
3. **後方互換性**: params.yamlの読み込みに失敗した場合はデフォルト値を使用
4. **コマンドライン引数の尊重**: すでに設定されている場合は上書きしない

## 効果

### 設定の一貫性

| 項目 | 変更前 | 変更後 |
|------|--------|--------|
| `way` | 1 (ハードコード) | 2 (params.yamlから) |
| `sessions` | 7 (ハードコード) | 4 (計算: (10-4)/2+1) |
| セッションファイル | 不一致（4個しかない） | 一致（4個必要、4個存在） |

### トレーナーの動作

```bash
=== Trainer Configuration ===
Base class: 4
Num classes: 10
Way: 2                    # ✓ params.yamlから正しく読み込み
Shot: 3
Total sessions: 4         # ✓ 正しく計算

Calculation: (10 - 4) / 2 + 1 = 4

New sessions: [1, 2, 3]   # ✓ 実際のファイルと一致
```

## テスト

### 設定の確認
```bash
uv run python -c "
from train import initialize_trainer, get_command_line_parser
parser = get_command_line_parser()
args = parser.parse_args([])
trainer = initialize_trainer(args)
print(f'Way: {trainer.args.way}')
print(f'Sessions: {trainer.args.sessions}')
print(f'New sessions: {list(range(1, trainer.args.sessions))}')
"
```

**期待される出力**:
```
Way: 2
Sessions: 4
New sessions: [1, 2, 3]
```

### セッションファイルの確認
```bash
ls data/index_list/CICIDS2017_improved/session_0.txt
ls data/index_list/CICIDS2017_improved/new_sessions/*.txt
```

**期待される出力**:
```
session_0.txt
new_sessions/session_1.txt
new_sessions/session_2.txt
new_sessions/session_3.txt
```

## パラメータの変更

今後、セッション構成を変更したい場合は`params.yaml`を編集するだけでOK：

```yaml
create_sessions:
  base_class: 4
  num_classes: 10
  way: 2           # ← これを変更
  shot: 3
```

変更後、セッションファイルを再生成：
```bash
dvc repro create_new_sessions
```

トレーナーが自動的に新しい設定を使用します。

## 関連する修正

この修正により、以下の一貫性も確保されました：

1. **`create_session_files.py`** → `params.yaml`の`create_sessions`を使用
2. **`train_new_sessions.py`** → トレーナーの`args.sessions`を使用
3. **`dataloader/data_utils.py`** → `params.yaml`の`create_sessions`を使用（今回の修正）

すべてが`params.yaml`の`create_sessions`セクションから設定を取得するようになりました。

## トラブルシューティング

### 問題: まだ古い設定が使われている

キャッシュをクリア：
```bash
# Pythonキャッシュを削除
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# セッションファイルを再生成
dvc repro create_new_sessions
```

### 問題: セッション数が一致しない

`params.yaml`の設定を確認：
```bash
grep -A 10 "create_sessions:" params.yaml
```

計算式を確認：
```
sessions = (num_classes - base_class) / way + 1
```

## 関連ドキュメント

- **パラメータ分離**: [DVC_PARAMS_SEPARATION.md](DVC_PARAMS_SEPARATION.md)
- **ディレクトリ構造**: [SESSION_FILES_STRUCTURE.md](SESSION_FILES_STRUCTURE.md)
- **実装サマリー**: [DVC_IMPLEMENTATION_SUMMARY.md](DVC_IMPLEMENTATION_SUMMARY.md)

---

**修正日**: 2025-11-27
**影響範囲**: CICIDS2017データセットのセッション管理
