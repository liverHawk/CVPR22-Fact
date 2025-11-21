import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='CICIDS2017_improved',
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'CICIDS2017_improved'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-encoder', type=str, default='mlp',
                        choices=['mlp', 'cnn1d'])

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=100)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Step',
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=[60, 70])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=128)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #for fact
    parser.add_argument('-balance', type=float, default=1.0)
    parser.add_argument('-loss_iter', type=int, default=200)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)

    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    # wandb logging
    parser.add_argument('--use_wandb', action='store_true', help='Weights & Biasesへのログ送信を有効化')
    parser.add_argument('--wandb_project', type=str, default=None, help='wandbのプロジェクト名')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandbのエンティティ（ユーザーまたはチーム）')
    parser.add_argument('--wandb_group', type=str, default=None, help='wandbのグループ名')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='wandb上の実行名')
    parser.add_argument('--wandb_tags', nargs='*', default=None, help='wandbタグ（スペース区切り）')
    parser.add_argument('--wandb_mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'], help='wandbの実行モード')
    parser.add_argument('--wandb_watch', type=str, default='gradients',
                        choices=['gradients', 'parameters', 'all', 'none'], help='wandb.watchの対象')
    parser.add_argument('--wandb_watch_freq', type=int, default=100, help='wandb.watchのログ頻度（step単位）')
    
    # unknown detection
    parser.add_argument('--enable_unknown_detection', action='store_true', 
                        help='未知クラス検出を有効化（埋め込み空間での距離ベース検出）')
    parser.add_argument('--distance_type', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'euclidean_normalized', 'manhattan', 'chebyshev', 'mahalanobis'],
                        help='未知クラス検出に使用する距離メトリクスの種類')
    parser.add_argument('--distance_threshold', type=float, default=None,
                        help='未知クラス検出の距離閾値（Noneの場合は自動計算）')
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()