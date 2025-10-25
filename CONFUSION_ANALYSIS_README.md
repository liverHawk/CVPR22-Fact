# 混同行列分析ツール

FACTモデルのテスト結果から混同行列を生成し、詳細な分析を行うツールです。

## 機能

### 1. 混同行列の可視化
- 各セッションの混同行列をヒートマップで表示
- 正規化された混同行列（行ごとに正規化）
- 全体精度の表示

### 2. クラス別精度分析
- ベースクラス（既知クラス）の精度分析
- 新規クラスの精度分析
- クラス別精度の可視化

### 3. セッション間比較
- 全体精度の推移
- ベースクラス精度の推移
- Catastrophic Forgettingの評価

### 4. 統計情報の保存
- JSON形式での詳細統計
- テキスト形式でのサマリー
- 比較分析レポート

## 使用方法

### 基本的な使用方法

```bash
# 全セッションを分析
python run_confusion_analysis.py

# 特定のセッションのみ分析
python run_confusion_analysis.py -sessions 0 1 2

# カスタム出力ディレクトリ
python run_confusion_analysis.py -output_dir my_analysis

# 異なるデータセット
python run_confusion_analysis.py -dataset cub200 -dataroot /path/to/cub200
```

### パラメータ

- `-dataset`: データセット（cifar100, cub200, mini_imagenet）
- `-dataroot`: データセットのルートディレクトリ
- `-checkpoint_dir`: モデルチェックポイントのディレクトリ
- `-output_dir`: 分析結果の出力ディレクトリ
- `-sessions`: 分析するセッション（デフォルト: 0-8）

### 出力ファイル

各セッションのディレクトリ（`session_X/`）に以下が保存されます：

1. **confusion_matrix.png**: 混同行列のヒートマップ
2. **per_class_accuracy.png**: クラス別精度のグラフ
3. **accuracy_stats.json**: 詳細な統計情報（JSON）
4. **accuracy_summary.txt**: 統計サマリー（テキスト）

ルート出力ディレクトリに以下が保存されます：

1. **overall_accuracy_trend.png**: 全体精度の推移
2. **base_accuracy_trend.png**: ベースクラス精度の推移
3. **comparison_summary.txt**: セッション間比較サマリー

## 分析結果の解釈

### 混同行列
- **対角成分**: 各クラスの正解率
- **非対角成分**: クラス間の混同率
- **色の濃さ**: 混同の程度（濃いほど混同が多い）

### クラス別精度
- **ベースクラス**: 最初に学習したクラス（0-59）
- **新規クラス**: インクリメンタル学習で追加されたクラス（60以降）
- **精度の低下**: Catastrophic Forgettingの指標

### Catastrophic Forgetting評価
- **0.1以上**: 重大な忘却
- **0.05-0.1**: 中程度の忘却
- **0.05未満**: 良好な制御

## 例：CIFAR-100での分析

```bash
# CIFAR-100の全セッションを分析
python run_confusion_analysis.py \
    -dataset cifar100 \
    -dataroot data/ \
    -checkpoint_dir checkpoint/cifar100/fact/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_3-Lr_0.1000Bal0.00-LossIter0-T_16.00 \
    -output_dir cifar100_analysis

# 結果の確認
ls cifar100_analysis/
# session_0/ session_1/ ... session_8/
# overall_accuracy_trend.png
# base_accuracy_trend.png
# comparison_summary.txt
```

## トラブルシューティング

### よくある問題

1. **チェックポイントが見つからない**
   ```
   エラー: チェックポイントが見つかりません
   解決策: -checkpoint_dir パスを確認
   ```

2. **CUDA エラー**
   ```
   エラー: CUDA out of memory
   解決策: バッチサイズを小さくするか、CPUで実行
   ```

3. **データセットが見つからない**
   ```
   エラー: Dataset not found
   解決策: -dataroot パスを確認し、データセットをダウンロード
   ```

### 必要な依存関係

```bash
pip install torch torchvision matplotlib seaborn scikit-learn numpy
```

## カスタマイズ

### 新しい可視化の追加

`run_confusion_analysis.py`の`analyze_session`関数を修正して、新しい可視化を追加できます。

### 異なるメトリクスの計算

`analyze_session`関数内で、追加のメトリクスを計算・保存できます。

## 注意事項

- モデルチェックポイントが存在するセッションのみ分析されます
- 大きなデータセットでは分析に時間がかかる場合があります
- GPUメモリが不足する場合は、バッチサイズを調整してください
