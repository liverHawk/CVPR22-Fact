# Base vs FACT 比較機能

このプロジェクトには、BaseモデルとFACTモデルを同時に比較する機能が実装されています。

## 機能概要

- **同時評価**: BaseとFACTモデルを同じデータで同時に評価
- **詳細比較**: 精度、混同行列、クラス別性能の詳細な比較
- **可視化**: 比較結果の包括的な可視化
- **統計分析**: 改善率、破滅的忘却の分析

## 使用方法

### 1. 基本的な比較

```bash
# 両モデルを訓練
make train-all

# BaseとFACTを比較
make compare-base-fact
```

### 2. デバッグモードでの比較

```bash
# クイック訓練（デバッグモード）
make quick-all

# デバッグモードでの比較
make compare-base-fact DEBUG=true MAX_SAMPLES=100
```

### 3. 直接スクリプト実行

```bash
python compare_base_fact.py \
    -dataset cicids2017_improved \
    -dataroot data/ \
    -base_checkpoint_dir checkpoint/cicids2017_improved/base \
    -fact_checkpoint_dir checkpoint/cicids2017_improved/fact \
    -output_dir base_fact_comparison \
    -sessions 0 1 2 3 4 5 6
```

## 出力結果

比較結果は `base_fact_comparison/` ディレクトリに保存されます：

### セッション別結果 (`session_X/`)
- `confusion_matrix_comparison.png`: 混同行列の比較
- `accuracy_comparison.png`: 精度の比較（全体、ベースクラス、新規クラス）
- `per_class_comparison.png`: クラス別精度の比較
- `comparison_stats.json`: 統計情報（JSON形式）
- `comparison_summary.txt`: 比較サマリー（テキスト形式）

### 全体比較結果
- `overall_accuracy_comparison.png`: 全セッションの精度推移比較
- `base_accuracy_comparison.png`: ベースクラス精度の推移比較
- `improvement_comparison.png`: FACTの改善率
- `overall_comparison_summary.txt`: 全体比較サマリー

## 比較指標

### 1. 全体精度
- FACTとBaseの全体精度の差
- 改善率（パーセンテージ）

### 2. ベースクラス精度
- 既存クラスの性能維持
- 破滅的忘却の評価

### 3. 新規クラス精度
- 新しく学習したクラスの性能
- インクリメンタル学習の効果

### 4. クラス別分析
- 各クラスでの性能差
- 混同パターンの比較

## パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `-dataset` | `cicids2017_improved` | データセット名 |
| `-dataroot` | `data/` | データのルートディレクトリ |
| `-base_checkpoint_dir` | `checkpoint/cicids2017_improved/base` | Baseモデルのチェックポイントディレクトリ |
| `-fact_checkpoint_dir` | `checkpoint/cicids2017_improved/fact` | FACTモデルのチェックポイントディレクトリ |
| `-output_dir` | `base_fact_comparison` | 出力ディレクトリ |
| `-sessions` | `0 1 2 3 4 5 6` | 比較するセッション |
| `-max_samples` | `None` | デバッグ用の最大サンプル数制限 |

## 例：結果の解釈

### 成功例
```
Overall Accuracy:
  FACT: 0.8542
  Base: 0.8231
  Difference: +0.0311
  Improvement: +3.78%

Winner: FACT
```

### 破滅的忘却の評価
```
Catastrophic Forgetting Analysis:
FACT Forgetting Rate: 0.0234
Base Forgetting Rate: 0.0456
Forgetting Control: FACT
```

## トラブルシューティング

### 1. チェックポイントが見つからない
```bash
# ステータス確認
make status

# 必要に応じて再訓練
make train-all
```

### 2. メモリ不足
```bash
# サンプル数を制限
make compare-base-fact MAX_SAMPLES=500
```

### 3. デバッグモード
```bash
# デバッグ情報を表示
make compare-base-fact DEBUG=true
```

## 技術的詳細

### アーキテクチャ
- **BaseFactComparator**: メインの比較クラス
- **同時評価**: 両モデルを同じデータで評価
- **詳細分析**: 混同行列、精度、信頼度の分析

### 可視化機能
- **混同行列**: 正規化された混同行列の比較
- **精度比較**: バーチャートとライングラフ
- **クラス別分析**: 各クラスの性能差の可視化

### 統計分析
- **改善率計算**: パーセンテージでの改善度
- **破滅的忘却**: ベースクラス性能の維持度
- **信頼度分析**: 予測の信頼度の比較
