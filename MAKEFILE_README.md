# CICIDS2017 Dataset Training and Evaluation Makefile

このMakefileを使用して、CICIDS2017データセットでのbaseモデルとfactモデルの訓練・評価を簡単に実行できます。

## 前提条件

- Python 3.13+
- uv (Python package manager)
- CUDA対応GPU（推奨）

## 基本的な使用方法

### ヘルプの表示
```bash
make help
```

### データローディングのテスト
```bash
make test-data
```

### 現在のステータス確認
```bash
make status
```

## 訓練コマンド

### Baseモデルの訓練
```bash
# デフォルト設定で訓練
make train-base

# カスタム設定で訓練
make train-base EPOCHS_BASE=3 SESSIONS=4 MAX_SAMPLES=2000
```

### Factモデルの訓練
```bash
# デフォルト設定で訓練
make train-fact

# カスタム設定で訓練
make train-fact EPOCHS_BASE=5 EPOCHS_NEW=2
```

### 両方のモデルを訓練
```bash
make train-all
```

## 評価コマンド

### Baseモデルの評価
```bash
make eval-base
```

### Factモデルの評価
```bash
make eval-fact
```

### 両方のモデルを評価
```bash
make eval-all
```

## クイックテスト

デバッグモードで短時間の訓練を実行：

```bash
# Baseモデルのクイックテスト
make quick-base

# Factモデルのクイックテスト
make quick-fact

# 両方のクイックテスト
make quick-all
```

## 設定可能な変数

| 変数 | デフォルト値 | 説明 |
|------|-------------|------|
| `DATASET` | `cicids2017_improved` | データセット名 |
| `DATAROOT` | `data` | データディレクトリ |
| `EPOCHS_BASE` | `1` | ベースセッションのエポック数 |
| `EPOCHS_NEW` | `1` | 新規セッションのエポック数 |
| `SESSIONS` | `6` | 総セッション数 |
| `START_SESSION` | `0` | 開始セッション |
| `MAX_SAMPLES` | `1000` | 最大サンプル数（デバッグ用） |
| `DEBUG` | `false` | デバッグモード |
| `BATCH_SIZE_BASE` | `128` | ベースセッションのバッチサイズ |
| `BATCH_SIZE_NEW` | `16` | 新規セッションのバッチサイズ |
| `TEST_BATCH_SIZE` | `100` | テストのバッチサイズ |

## 使用例

### 本格的な訓練（デバッグモード無効）
```bash
make train-base EPOCHS_BASE=10 EPOCHS_NEW=5 SESSIONS=6 DEBUG=false
```

### デバッグ用の短時間訓練
```bash
make quick-all
```

### 特定のセッション数のみで訓練
```bash
make train-all SESSIONS=3
```

### より多くのサンプルで訓練
```bash
make train-base MAX_SAMPLES=5000
```

## 出力ファイル

### チェックポイント
- `checkpoint/cicids2017_improved/base/` - Baseモデルのチェックポイント
- `checkpoint/cicids2017_improved/fact/` - Factモデルのチェックポイント

### 分析結果
- `confusion_analysis/cicids2017_base/` - Baseモデルの分析結果
- `confusion_analysis/cicids2017_fact/` - Factモデルの分析結果

各分析ディレクトリには以下が含まれます：
- `session_X/confusion_matrix.png` - 混同行列
- `session_X/per_class_accuracy.png` - クラス別精度
- `session_X/accuracy_stats.json` - 統計情報（JSON）
- `session_X/accuracy_summary.txt` - 統計情報（テキスト）
- `overall_accuracy_trend.png` - 全体精度の推移
- `base_accuracy_trend.png` - ベースクラス精度の推移
- `comparison_summary.txt` - セッション間の比較

## クリーンアップ

### チェックポイントのみ削除
```bash
make clean-checkpoints
```

### 分析結果のみ削除
```bash
make clean-analysis
```

### すべての生成ファイルを削除
```bash
make clean
```

## トラブルシューティング

### データローディングエラー
```bash
make test-data
```

### チェックポイントが見つからない
```bash
make status
```

### メモリ不足の場合
```bash
make train-base MAX_SAMPLES=500
```

## 開発環境のセットアップ

```bash
make dev-setup
```

## 注意事項

1. **GPU使用**: CUDA対応GPUが利用可能な場合、自動的にGPUを使用します
2. **メモリ使用量**: `MAX_SAMPLES`を調整してメモリ使用量を制御できます
3. **デバッグモード**: `DEBUG=true`に設定すると、より詳細なログが出力されます
4. **チェックポイント**: 訓練中にエラーが発生した場合、既存のチェックポイントから再開できます

## サポート

問題が発生した場合は、以下を確認してください：

1. データセットが正しく配置されているか
2. 必要な依存関係がインストールされているか
3. GPUメモリが十分か
4. ファイルパスが正しいか

```bash
# 環境確認
make test-data
make status
```
