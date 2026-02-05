import argparse
import importlib
import os
import yaml
import torch
from utils import set_seed, set_gpu, pprint

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='fact' # base, fact
PARAMS_FILE = 'params.yaml'

def load_params_yaml(yaml_path=PARAMS_FILE):
    """params.yamlから設定を読み込む"""
    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found. Using default values.")
        return {}
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
        return params if params else {}
    except Exception as e:
        print(f"Warning: Failed to load {yaml_path}: {e}. Using default values.")
        return {}

def get_command_line_parser():
    # params.yamlからデフォルト値を読み込む
    yaml_params = load_params_yaml()
    
    parser = argparse.ArgumentParser(
        description='FACT: Forward Compatible Few-Shot Class-Incremental Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基本設定（params.yamlから読み込み、コマンドラインで上書き可能）
    parser.add_argument('-p', '--project', type=str, 
                        default=yaml_params.get('project', PROJECT),
                        choices=['base', 'fact'],
                        help='使用するプロジェクト (base: ベースライン, fact: FACT手法)')
    parser.add_argument('-d', '--dataset-type', type=str, 
                        default=yaml_params.get('dataset_type', 'cifar100'),
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'CICIDS2017_improved'],
                        dest='dataset_type',
                        help='データセットタイプ（データローダーの種類を決定）')
    parser.add_argument('-n', '--dataset-name', type=str, 
                        default=yaml_params.get('dataset_name', None),
                        dest='dataset_name',
                        help='データセット名（パス構築に使用、指定しない場合はdataset_typeと同じ値）')
    parser.add_argument('--dataroot', type=str, 
                        default=DATA_DIR,
                        help='データセットのルートディレクトリ（デフォルト: data/）')
    parser.add_argument('--encoder', type=str,
                        default=yaml_params.get('encoder', 'mlp'),
                        choices=['mlp', 'cnn1d'],
                        help='エンコーダーの種類（mlp: MLP, cnn1d: CNN1D）')

    # トレーニング設定
    parser.add_argument('--epochs-base', type=int, 
                        default=yaml_params.get('epochs_base', 400),
                        dest='epochs_base',
                        help='ベースセッションのエポック数')
    parser.add_argument('--epochs-new', type=int, 
                        default=yaml_params.get('epochs_new', 100),
                        dest='epochs_new',
                        help='新規セッションのエポック数')
    parser.add_argument('--lr-base', type=float, 
                        default=yaml_params.get('lr_base', 0.005),
                        dest='lr_base',
                        help='ベースセッションの学習率')
    parser.add_argument('--lr-new', type=float, 
                        default=yaml_params.get('lr_new', 0.1),
                        dest='lr_new',
                        help='新規セッションの学習率')
    parser.add_argument('--batch-size-base', type=int, 
                        default=yaml_params.get('batch_size_base', 256),
                        dest='batch_size_base',
                        help='ベースセッションのバッチサイズ')
    parser.add_argument('--batch-size-new', type=int, 
                        default=yaml_params.get('batch_size_new', 0),
                        dest='batch_size_new',
                        help='新規セッションのバッチサイズ (0で全データ使用)')
    parser.add_argument('--test-batch-size', type=int, 
                        default=yaml_params.get('test_batch_size', 100),
                        dest='test_batch_size',
                        help='テスト時のバッチサイズ')

    # モード設定
    parser.add_argument('--base-mode', type=str, 
                        default=yaml_params.get('base_mode', 'ft_cos'),
                        choices=['ft_dot', 'ft_cos'],
                        dest='base_mode',
                        help='ベースセッションのモード (ft_dot: 線形分類器, ft_cos: コサイン分類器)')
    parser.add_argument('--new-mode', type=str, 
                        default=yaml_params.get('new_mode', 'avg_cos'),
                        choices=['ft_dot', 'ft_cos', 'avg_cos'],
                        dest='new_mode',
                        help='新規セッションのモード')
    
    # 学習率スケジュール
    parser.add_argument('--schedule', type=str, 
                        default=yaml_params.get('schedule', 'Milestone'),
                        choices=['Step', 'Milestone','Cosine'],
                        help='学習率スケジュール')
    parser.add_argument('--milestones', nargs='+', type=int, 
                        default=yaml_params.get('milestones', [50, 100, 150, 200, 250, 300]),
                        help='Milestoneスケジュールのエポック')
    parser.add_argument('--step', type=int, 
                        default=yaml_params.get('step', 20),
                        help='Stepスケジュールのステップサイズ')
    parser.add_argument('--decay', type=float, 
                        default=yaml_params.get('decay', 0.0005),
                        help='重み減衰')
    parser.add_argument('--gamma', type=float, 
                        default=yaml_params.get('gamma', 0.25),
                        help='学習率減衰率')
    parser.add_argument('--momentum', type=float, 
                        default=yaml_params.get('momentum', 0.9),
                        help='モーメンタム')
    parser.add_argument('--temperature', type=float, 
                        default=yaml_params.get('temperature', 16),
                        help='温度パラメータ')
    
    # FACT固有のパラメータ
    parser.add_argument('--balance', type=float, 
                        default=yaml_params.get('balance', 0.01),
                        help='FACT: balanceパラメータ')
    parser.add_argument('--alpha', type=float, 
                        default=yaml_params.get('alpha', 2.0),
                        help='FACT: alphaパラメータ')
    parser.add_argument('--eta', type=float, 
                        default=yaml_params.get('eta', 0.1),
                        help='FACT: etaパラメータ')
    parser.add_argument('--loss-iter', type=int, 
                        default=yaml_params.get('loss_iter', 0),
                        dest='loss_iter',
                        help='FACT: loss_iterパラメータ')
    
    # その他の設定
    parser.add_argument('--start-session', type=int, 
                        default=yaml_params.get('start_session', 0),
                        dest='start_session',
                        help='開始セッション番号')
    parser.add_argument('--model-dir', type=str, 
                        default=yaml_params.get('model_dir', MODEL_DIR),
                        dest='model_dir',
                        help='モデルパラメータの読み込み元ディレクトリ')
    parser.add_argument('--num-workers', type=int, 
                        default=yaml_params.get('num_workers', 8),
                        dest='num_workers',
                        help='データローダーのワーカー数（システム推奨値を超える場合は自動調整）')
    parser.add_argument('-g', '--gpu', type=str, 
                        default=yaml_params.get('gpu', '0'),
                        help='使用するGPU（カンマ区切り、例: 0,1,2,3）。CPUを使用する場合は "cpu" を指定')
    parser.add_argument('-s', '--seed', type=int, 
                        default=yaml_params.get('seed', 1),
                        help='乱数シード')
    
    # フラグ
    parser.add_argument('--not-data-init', 
                        action='store_true',
                        dest='not_data_init',
                        help='平均データ埋め込みで初期化しない')
    parser.add_argument('--set-no-val', 
                        action='store_true',
                        dest='set_no_val',
                        help='バリデーションを使用しない')
    parser.add_argument('--debug', 
                        action='store_true',
                        help='デバッグモード')
    
    # CICIDS2017_improved用のパラメータ
    parser.add_argument('--normalize-method', type=str,
                        default=yaml_params.get('normalize_method', 'standard'),
                        choices=['standard', 'minmax', 'moving_minmax'],
                        dest='normalize_method',
                        help='正規化方法（standard: 標準化, minmax: Min-Max正規化, moving_minmax: Moving Min-Max正規化）')
    parser.add_argument('--label-column', type=str,
                        default=yaml_params.get('label_column', 'Label'),
                        dest='label_column',
                        help='ラベル列の名前（CICIDS2017_improved用）')
    
    # FSCIL用のパラメータ（CICIDS2017_improvedなどで使用）
    parser.add_argument('--base-class', type=int,
                        default=yaml_params.get('base_class', None),
                        dest='base_class',
                        help='ベースクラス数')
    parser.add_argument('--num-classes', type=int,
                        default=yaml_params.get('num_classes', None),
                        dest='num_classes',
                        help='総クラス数')
    parser.add_argument('--way', type=int,
                        default=yaml_params.get('way', None),
                        help='各新規セッションのクラス数')
    parser.add_argument('--shot', type=int,
                        default=yaml_params.get('shot', 5),
                        help='各クラスのショット数（Few-Shot用）')
    create_sessions = yaml_params.get('create_sessions', {})
    base_labels = create_sessions.get('base_labels', None)
    print(base_labels)
    parser.add_argument('--base-labels', type=list,
                        default=base_labels,
                        help='ベースセッションに使用するラベルのリスト')
    
    return parser, yaml_params


if __name__ == '__main__':
    parser, yaml_params = get_command_line_parser()
    args = parser.parse_args()
    
    # YAMLから読み込んだフラグの値を適用
    # コマンドラインで指定されていない場合のみYAMLの値を使用
    # (argparseのaction='store_true'は指定されないとFalseになるため)
    import sys
    if '--not-data-init' not in sys.argv and '--not_data_init' not in sys.argv:
        args.not_data_init = yaml_params.get('not_data_init', False)
    if '--set-no-val' not in sys.argv and '--set_no_val' not in sys.argv:
        args.set_no_val = yaml_params.get('set_no_val', False)
    if '--debug' not in sys.argv:
        args.debug = yaml_params.get('debug', False)
    
    # dataset_nameが指定されていない場合、dataset_typeと同じ値を使用
    if args.dataset_name is None:
        args.dataset_name = args.dataset_type
    
    # 後方互換性のため、args.datasetも設定（既存コードでargs.datasetを使用している場合）
    args.dataset = args.dataset_type
    
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    # pin_memoryはGPUが利用可能な場合のみTrueにする
    args.pin_memory = len(args.num_gpu) > 0
    
    # CPU環境での最適化
    if len(args.num_gpu) == 0:  # CPU環境
        # PyTorchスレッド数の最適化
        num_threads = os.cpu_count() or 1
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(min(num_threads, 4))  # inter-op threads
        print(f"CPU環境: PyTorchスレッド数を{num_threads}に設定")
        
        # num_workersの最適化（CPU環境）
        max_workers = os.cpu_count() or 1
        # データローディングと計算を並行するため、少し控えめに設定
        optimal_workers = max(1, min(max_workers - 1, 4))
        if args.num_workers > optimal_workers:
            print(f"CPU環境: num_workersを{args.num_workers}から{optimal_workers}に調整")
            args.num_workers = optimal_workers
    else:
        # GPU環境では既存のロジックを維持
        max_workers = min(os.cpu_count() or 1, 4)
        if args.num_workers > max_workers:
            print(f"Warning: num_workers ({args.num_workers}) exceeds system recommendation ({max_workers}). Adjusting to {max_workers}.")
            args.num_workers = max_workers
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()