# パラメータ説明

このドキュメントでは、`params.yaml`で設定可能なすべてのパラメータについて説明します。

## 目次

1. [基本設定](#基本設定)
2. [トレーニング設定](#トレーニング設定)
3. [モード設定](#モード設定)
4. [学習率スケジュール](#学習率スケジュール)
5. [FACT固有のパラメータ](#fact固有のパラメータ)
6. [FSCIL用のパラメータ](#fscil用のパラメータ)
7. [その他の設定](#その他の設定)
8. [フラグ](#フラグ)
9. [Comet ML設定](#comet-ml設定)
10. [セッションファイル生成設定](#セッションファイル生成設定)

---

## 基本設定

### `project`
- **型**: `str`
- **デフォルト値**: `fact`
- **選択肢**: `'base'` または `'fact'`
- **説明**: 使用するプロジェクトを指定します。
  - `base`: ベースライン手法
  - `fact`: FACT（Forward Compatible Few-Shot Class-Incremental Learning）手法
- **例**: `project: fact`

### `dataset_type`
- **型**: `str`
- **デフォルト値**: `CICIDS2017_improved`
- **選択肢**: `'mini_imagenet'`, `'cub200'`, `'cifar100'`, `'CICIDS2017_improved'`
- **説明**: データセットタイプを指定します。データローダーの種類を決定します。
- **例**: `dataset_type: CICIDS2017_improved`

### `dataset_name`
- **型**: `str`
- **デフォルト値**: `CICIDS2017_flow_improved`
- **説明**: データセット名を指定します。パス構築に使用されます。指定しない場合は`dataset_type`と同じ値が使用されます。
- **例**: `dataset_name: CICIDS2017_flow_improved`

### `dataroot`
- **型**: `str`
- **デフォルト値**: `data/`
- **説明**: データセットのルートディレクトリを指定します。
- **例**: `dataroot: data/`

### `encoder`
- **型**: `str`
- **デフォルト値**: `mlp`
- **選択肢**: `'mlp'` または `'cnn1d'`
- **説明**: エンコーダーの種類を指定します。
  - `mlp`: MLP（Multi-Layer Perceptron）エンコーダー
  - `cnn1d`: 1次元CNNエンコーダー（CICIDS2017_improved用）
- **例**: `encoder: mlp`

### `normalize_method`
- **型**: `str`
- **デフォルト値**: `standard`
- **選択肢**: `'standard'`, `'minmax'`, `'moving_minmax'`
- **説明**: データの正規化方法を指定します。
  - `standard`: 標準化（平均0、分散1）
  - `minmax`: Min-Max正規化（0-1スケール）
  - `moving_minmax`: 移動Min-Max正規化
- **例**: `normalize_method: standard`

### `label_column`
- **型**: `str`
- **デフォルト値**: `Label`
- **説明**: ラベル列の名前を指定します（CICIDS2017_improved用）。
- **例**: `label_column: Label`

---

## トレーニング設定

### `epochs_base`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: ベースセッションのエポック数を指定します。
- **例**: `epochs_base: 400`

### `epochs_new`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: 新規セッションのエポック数を指定します。
- **例**: `epochs_new: 100`

### `lr_base`
- **型**: `float`
- **デフォルト値**: `0.005`
- **説明**: ベースセッションの学習率を指定します。
- **例**: `lr_base: 0.005`

### `lr_new`
- **型**: `float`
- **デフォルト値**: `0.1`
- **説明**: 新規セッションの学習率を指定します。
- **例**: `lr_new: 0.1`

### `batch_size_base`
- **型**: `int`
- **デフォルト値**: `256`
- **説明**: ベースセッションのバッチサイズを指定します。
- **例**: `batch_size_base: 256`

### `batch_size_new`
- **型**: `int`
- **デフォルト値**: `0`
- **説明**: 新規セッションのバッチサイズを指定します。`0`を指定すると全データを使用します。
- **例**: `batch_size_new: 0`

### `test_batch_size`
- **型**: `int`
- **デフォルト値**: `100`
- **説明**: テスト時のバッチサイズを指定します。
- **例**: `test_batch_size: 100`

---

## モード設定

### `base_mode`
- **型**: `str`
- **デフォルト値**: `ft_cos`
- **選択肢**: `'ft_dot'` または `'ft_cos'`
- **説明**: ベースセッションのモードを指定します。
  - `ft_dot`: 線形分類器（ドット積）
  - `ft_cos`: コサイン分類器
- **例**: `base_mode: ft_cos`

### `new_mode`
- **型**: `str`
- **デフォルト値**: `avg_cos`
- **選択肢**: `'ft_dot'`, `'ft_cos'`, `'avg_cos'`
- **説明**: 新規セッションのモードを指定します。
  - `ft_dot`: 線形分類器（ドット積）
  - `ft_cos`: コサイン分類器
  - `avg_cos`: 平均コサイン分類器
- **例**: `new_mode: avg_cos`

---

## 学習率スケジュール

### `schedule`
- **型**: `str`
- **デフォルト値**: `Milestone`
- **選択肢**: `'Step'`, `'Milestone'`, `'Cosine'`
- **説明**: 学習率スケジュールの種類を指定します。
  - `Step`: 一定間隔（`step`）ごとに学習率を減衰
  - `Milestone`: 指定エポック（`milestones`）で学習率を減衰
  - `Cosine`: コサイン関数に沿って学習率を減衰
- **例**: `schedule: Milestone`

### `milestones`
- **型**: `list[int]`
- **デフォルト値**: `[50, 100, 150, 200, 250, 300]`
- **説明**: Milestoneスケジュールで学習率を減衰させるエポック番号のリストを指定します。`schedule`が`Milestone`の場合に使用されます。
- **例**: 
  ```yaml
  milestones:
    - 50
    - 100
    - 150
    - 200
    - 250
    - 300
  ```

### `step`
- **型**: `int`
- **デフォルト値**: `20`
- **説明**: Stepスケジュールのステップサイズを指定します。`schedule`が`Step`の場合に使用されます。
- **例**: `step: 20`

### `decay`
- **型**: `float`
- **デフォルト値**: `0.0005`
- **説明**: 重み減衰（weight decay）の値を指定します。L2正則化の強度を制御します。
- **例**: `decay: 0.0005`

### `gamma`
- **型**: `float`
- **デフォルト値**: `0.25`
- **説明**: 学習率減衰率を指定します。スケジュールに従って学習率を`gamma`倍に減衰させます。
- **例**: `gamma: 0.25`

### `momentum`
- **型**: `float`
- **デフォルト値**: `0.9`
- **説明**: SGDオプティマイザーのモーメンタムを指定します。
- **例**: `momentum: 0.9`

### `temperature`
- **型**: `float`
- **デフォルト値**: `16`
- **説明**: 温度パラメータを指定します。知識蒸留などで使用される温度スケーリングのパラメータです。
- **例**: `temperature: 16`

---

## FACT固有のパラメータ

### `balance`
- **型**: `float`
- **デフォルト値**: `0.01`
- **説明**: FACT手法のbalanceパラメータを指定します。損失関数のバランスを調整します。
- **例**: `balance: 0.01`

### `alpha`
- **型**: `float`
- **デフォルト値**: `2.0`
- **説明**: FACT手法のalphaパラメータを指定します。
- **例**: `alpha: 2.0`

### `eta`
- **型**: `float`
- **デフォルト値**: `0.1`
- **説明**: FACT手法のetaパラメータを指定します。
- **例**: `eta: 0.1`

### `loss_iter`
- **型**: `int`
- **デフォルト値**: `0`
- **説明**: FACT手法のloss_iterパラメータを指定します。損失計算を開始するイテレーションを指定します。
- **例**: `loss_iter: 0`

---

## FSCIL用のパラメータ

### `base_class`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: ベースクラス数を指定します。Few-Shot Class-Incremental Learning（FSCIL）で使用されます。
- **例**: `base_class: 5`

### `num_classes`
- **型**: `int`
- **デフォルト値**: `10`
- **説明**: 総クラス数を指定します。
- **例**: `num_classes: 10`

### `way`
- **型**: `int`
- **デフォルト値**: `1`
- **説明**: 各新規セッションのクラス数を指定します。
- **例**: `way: 1`

### `shot`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: 各クラスのショット数（サンプル数）を指定します。Few-Shot学習で使用されます。
- **例**: `shot: 5`

---

## その他の設定

### `start_session`
- **型**: `int`
- **デフォルト値**: `0`
- **説明**: 開始セッション番号を指定します。途中のセッションから再開する場合に使用します。
- **例**: `start_session: 0`

### `model_dir`
- **型**: `str` または `null`
- **デフォルト値**: `null`
- **説明**: モデルパラメータの読み込み元ディレクトリを指定します。`null`の場合は新規に学習を開始します。
- **例**: `model_dir: checkpoint/cifar100/fact/...`

### `num_workers`
- **型**: `int`
- **デフォルト値**: `8`
- **説明**: データローダーのワーカー数を指定します。システム推奨値を超える場合は自動調整されます。
- **例**: `num_workers: 8`

### `gpu`
- **型**: `str`
- **デフォルト値**: `cpu`
- **説明**: 使用するGPUを指定します。カンマ区切りで複数指定可能です（例: `'0,1,2,3'`）。CPUを使用する場合は`"cpu"`を指定します。
- **例**: 
  - `gpu: cpu`（CPU使用）
  - `gpu: 0`（GPU 0を使用）
  - `gpu: 0,1,2,3`（複数GPU使用）

### `seed`
- **型**: `int`
- **デフォルト値**: `1`
- **説明**: 乱数シードを指定します。再現性を確保するために使用されます。
- **例**: `seed: 1`

---

## フラグ

### `not_data_init`
- **型**: `bool`
- **デフォルト値**: `false`
- **説明**: 平均データ埋め込みで初期化しない場合に`true`に設定します。
- **例**: `not_data_init: false`

### `set_no_val`
- **型**: `bool`
- **デフォルト値**: `false`
- **説明**: バリデーションを使用しない場合に`true`に設定します。
- **例**: `set_no_val: false`

### `debug`
- **型**: `bool`
- **デフォルト値**: `false`
- **説明**: デバッグモードを有効にする場合に`true`に設定します。
- **例**: `debug: false`

---

## Comet ML設定

### `comet_project`
- **型**: `str` または `null`
- **デフォルト値**: `null`
- **説明**: Comet MLのプロジェクト名を指定します。`null`の場合はデータセット名が使用されます。
- **例**: `comet_project: fact-cicids2017`

### `comet_workspace`
- **型**: `str` または `null`
- **デフォルト値**: `null`
- **説明**: Comet MLのワークスペース名を指定します。`null`の場合はデフォルト値が使用されます。
- **例**: `comet_workspace: my-workspace`

### `comet_disabled`
- **型**: `bool`
- **デフォルト値**: `false`
- **説明**: Comet MLを無効化する場合に`true`に設定します。
- **例**: `comet_disabled: false`

---

## セッションファイル生成設定

`create_sessions`セクションのパラメータは、`create_session_files.py`スクリプトで使用されます。

### `create_sessions.label_column`
- **型**: `str`
- **デフォルト値**: `Label`
- **説明**: ラベル列の名前を指定します。
- **例**: `label_column: Label`

### `create_sessions.output_dir`
- **型**: `str`
- **デフォルト値**: `data/index_list`
- **説明**: セッションファイルの出力ディレクトリを指定します。
- **例**: `output_dir: data/index_list`

### `create_sessions.dataset_name`
- **型**: `str`
- **デフォルト値**: `CICIDS2017_flow_improved`
- **説明**: データセット名を指定します。
- **例**: `dataset_name: CICIDS2017_flow_improved`

### `create_sessions.base_class`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: ベースクラス数を指定します。
- **例**: `base_class: 5`

### `create_sessions.num_classes`
- **型**: `int`
- **デフォルト値**: `10`
- **説明**: 総クラス数を指定します。
- **例**: `num_classes: 10`

### `create_sessions.way`
- **型**: `int`
- **デフォルト値**: `1`
- **説明**: 各新規セッションのクラス数を指定します。
- **例**: `way: 1`

### `create_sessions.shot`
- **型**: `int`
- **デフォルト値**: `5`
- **説明**: 各クラスのショット数（Few-Shot用）を指定します。
- **例**: `shot: 5`

### `create_sessions.seed`
- **型**: `int`
- **デフォルト値**: `42`
- **説明**: 乱数シードを指定します。
- **例**: `seed: 42`

### `create_sessions.base_use_class_index`
- **型**: `bool`
- **デフォルト値**: `false`
- **説明**: ベースセッションファイルにクラスインデックスを使用する場合に`true`に設定します。`false`の場合はデータインデックスを使用します。
- **例**: `base_use_class_index: false`

### `create_sessions.base_labels`
- **型**: `list[str]` または `null`
- **デフォルト値**: `null`
- **説明**: ベースセッションに使用するラベルのリストを指定します。`null`の場合は自動で選択されます。リストを指定した場合は、そのラベルのみが使用されます。
- **例**: 
  ```yaml
  base_labels:
    - BENIGN
    - Botnet
    - DDoS
    - DoS
    - FTP-Patator
  ```

---

## パラメータの優先順位

パラメータは以下の優先順位で適用されます（高い順）：

1. コマンドライン引数
2. `params.yaml`の設定値
3. コード内のデフォルト値

コマンドライン引数で指定した値が最も優先されます。

---

## 関連ドキュメント

- [README.md](../readme.md) - プロジェクトの概要と使用方法
- [params.yaml](../params.yaml) - パラメータ設定ファイル
