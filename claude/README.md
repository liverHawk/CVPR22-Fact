# Claude Code ドキュメント

このディレクトリには、Claude Codeによる実装とDVCパイプラインの改善に関するドキュメントが含まれています。

## 📚 ドキュメント一覧

### 実装サマリー
- **[DVC_IMPLEMENTATION_SUMMARY.md](DVC_IMPLEMENTATION_SUMMARY.md)** - DVC実装の全体サマリーとクイックリファレンス
- **[DIRECTORY_STRUCTURE_UPDATE.md](DIRECTORY_STRUCTURE_UPDATE.md)** - セッションファイルのディレクトリ構造変更のサマリー

### 詳細説明
- **[DVC_PARAMS_SEPARATION.md](DVC_PARAMS_SEPARATION.md)** - パラメータ分離の技術的詳細
- **[SESSION_FILES_STRUCTURE.md](SESSION_FILES_STRUCTURE.md)** - セッションファイル構造の詳細説明
- **[PARAMS_YAML_INTEGRATION.md](PARAMS_YAML_INTEGRATION.md)** - params.yaml統合の修正（wayパラメータの問題解決）

### 変更履歴
- **[CHANGELOG_DVC.md](CHANGELOG_DVC.md)** - DVC関連の変更履歴とマイグレーション手順

## 🎯 主な改善点

### 1. パラメータの階層化と分離

**問題**: 新規セッション専用パラメータ（`epochs_new`など）を変更すると、ベースセッションも不必要に再実行される

**解決**: `params.yaml`を3階層に分離
```yaml
train:
  common:    # 共通パラメータ
  base:      # ベースセッション専用
  new:       # 新規セッション専用
```

**効果**:
- ✅ `train.new.epochs_new`変更 → `train_new`のみ再実行
- ✅ `train.base.epochs_base`変更 → `train_base`のみ再実行

### 2. DVCステージの完全分離

**問題**: `train_base`と`train_new`が`train`パラメータ全体に依存

**解決**: 各ステージが必要なパラメータのみに依存
- `train_base` → `train.common` + `train.base`
- `train_new` → `train.common` + `train.new`

### 3. セッションファイルのディレクトリ構造化

**問題**:
- すべてのセッションファイルが混在
- DVCオーバーラップエラー
- セッション数変更時に`dvc.yaml`を手動更新

**解決**: 新規セッション専用ディレクトリの導入
```
data/index_list/CICIDS2017_improved/
├── session_0.txt              # ベースセッション
└── new_sessions/              # 新規セッション専用
    ├── session_1.txt
    ├── session_2.txt
    └── session_3.txt
```

**効果**:
- ✅ DVCがディレクトリ全体を追跡（個別ファイル列挙不要）
- ✅ セッション数の変更に自動対応
- ✅ オーバーラップエラーの解消

### 4. foreachの削除

**問題**: `train_new`が`foreach: [1,2,3,4,5,6]`で個別ステージに分割

**解決**: 単一ステージで全セッションを処理
```yaml
train_new:
  cmd: uv run train_new_sessions.py  # foreachなし
```

**効果**: セッション数変更時に`dvc.yaml`を更新不要

## 📊 パラメータ変更の影響マトリクス

| パラメータ変更 | 再実行されるステージ | 理由 |
|-------------|-------------------|------|
| `train.new.epochs_new` | `train_new`のみ | ✅ 新規セッション専用 |
| `train.base.epochs_base` | `train_base`のみ | ✅ ベースセッション専用 |
| `train.common.encoder` | 両方 | ✅ 共通パラメータ |
| `create_sessions.way` | `create_new_sessions` → `train_new` | ✅ 新規セッションのみ |
| `create_sessions.shot` | `create_new_sessions` → `train_new` | ✅ 新規セッションのみ |
| `create_sessions.base_class` | すべて | ✅ ベースクラス数変更 |

## 🏗️ アーキテクチャ

### DVCパイプライン構造

```
prepare_data
    ↓
    ├─→ create_base_session → session_0.txt → train_base
    │
    └─→ create_new_sessions → new_sessions/ → train_new
                                                  ↑
                                    (depends on train_base)
```

### ファイル構成

```
プロジェクトルート/
├── params.yaml                 # 階層化されたパラメータ
├── dvc.yaml                    # 分離されたDVCステージ
│
├── create_base_session_file.py # ベースセッションファイル生成
├── create_session_files.py     # 新規セッションファイル生成
├── train_base_session.py       # ベースセッショントレーニング
├── train_new_sessions.py       # 新規セッショントレーニング
│
├── data/
│   └── index_list/CICIDS2017_improved/
│       ├── session_0.txt       # ベースセッション
│       └── new_sessions/       # 新規セッション
│
└── claude/                     # このドキュメント
```

## 🚀 クイックスタート

### 現在の状態確認
```bash
dvc status
dvc dag
```

### パイプラインの実行
```bash
# 全体を実行
dvc repro

# 特定のステージのみ
dvc repro train_base
dvc repro train_new
```

### パラメータ分離のテスト
```bash
# 新規セッションパラメータのみ変更
# params.yaml: train.new.epochs_new を変更
dvc status  # → train_new のみ changed

# ベースセッションパラメータのみ変更
# params.yaml: train.base.epochs_base を変更
dvc status  # → train_base のみ changed
```

## 📝 変更されたファイル

### コアファイル
- `params.yaml` - パラメータ階層化
- `dvc.yaml` - ステージ分離
- `train.py` - 後方互換性追加

### 新規作成
- `create_base_session_file.py` - ベースセッション専用スクリプト
- `claude/` - ドキュメントディレクトリ

### 変更
- `create_session_files.py` - `new_sessions/`ディレクトリサポート
- `dataloader/data_utils.py` - `new_sessions/`から読み込み
- `train_new_sessions.py` - マーカーファイル生成

## 🔧 トラブルシューティング

### パラメータ変更が検出されない
```bash
rm dvc.lock
dvc repro
```

### DVCオーバーラップエラー
現在の実装で解決済み。`session_0.txt`と`new_sessions/`が完全に分離されています。

### セッション数の変更
`params.yaml`の`create_sessions.num_classes`を変更して`dvc repro create_new_sessions`を実行。
`dvc.yaml`の変更は不要です。

## 📖 関連ドキュメント

- **プロジェクト概要**: [../CLAUDE.md](../CLAUDE.md)
- **メインREADME**: [../README.md](../README.md)

## 📅 実装履歴

- **2025-11-27**: パラメータ分離とディレクトリ構造化の実装完了
- **バージョン**: v2.0

---

**実装者**: Claude Code
**連絡先**: このプロジェクトに関する質問は、GitHubのissuesで受け付けています
