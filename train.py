import argparse
import importlib
from utils import *

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='fact' # base, fact

def load_defaults_from_yaml(yaml_path='params.yaml'):
    """params.yamlからデフォルト値を読み込む"""
    try:
        params = load_params_yaml(yaml_path)
        train_params = params.get('train', {})
        common_params = params.get('common', {})
        wandb_params = train_params.get('wandb', {})
        
        defaults = {}
        
        # 共通パラメータ
        defaults['gpu'] = str(common_params.get('gpu', 0))
        defaults['seed'] = common_params.get('seed', 1)
        
        # 訓練パラメータ
        defaults['project'] = train_params.get('project', PROJECT)
        defaults['dataset'] = train_params.get('dataset', 'CICIDS2017_improved')
        defaults['dataroot'] = train_params.get('dataroot', DATA_DIR)
        defaults['encoder'] = train_params.get('encoder', 'mlp')
        defaults['base_mode'] = train_params.get('base_mode', 'ft_cos')
        defaults['new_mode'] = train_params.get('new_mode', 'avg_cos')
        defaults['epochs_base'] = train_params.get('epochs_base', 400)
        defaults['epochs_new'] = train_params.get('epochs_new', 100)
        defaults['batch_size_base'] = train_params.get('batch_size_base', 256)
        defaults['test_batch_size'] = train_params.get('test_batch_size', 100)
        defaults['start_session'] = train_params.get('start_session', 0)
        defaults['lr_base'] = train_params.get('lr_base', 0.005)
        defaults['lr_new'] = train_params.get('lr_new', 0.1)
        defaults['schedule'] = train_params.get('schedule', 'Milestone')
        defaults['milestones'] = train_params.get('milestones', [50, 100, 150, 200, 250, 300])
        defaults['step'] = train_params.get('step', 20)
        defaults['decay'] = train_params.get('decay', 0.0005)
        defaults['momentum'] = train_params.get('momentum', 0.9)
        defaults['gamma'] = train_params.get('gamma', 0.25)
        defaults['temperature'] = train_params.get('temperature', 16)
        defaults['balance'] = train_params.get('balance', 0.01)
        defaults['loss_iter'] = train_params.get('loss_iter', 0)
        defaults['alpha'] = train_params.get('alpha', 2.0)
        defaults['eta'] = train_params.get('eta', 0.1)
        defaults['batch_size_new'] = train_params.get('batch_size_new', 0)
        defaults['not_data_init'] = train_params.get('not_data_init', False)
        defaults['set_no_val'] = train_params.get('set_no_val', False)
        defaults['num_workers'] = train_params.get('num_workers', 8)
        defaults['debug'] = train_params.get('debug', False)
        defaults['model_dir'] = train_params.get('model_dir', MODEL_DIR)
        
        # wandbパラメータ
        defaults['use_wandb'] = wandb_params.get('use_wandb', False)
        defaults['wandb_project'] = wandb_params.get('project', None)
        defaults['wandb_entity'] = wandb_params.get('entity', None)
        defaults['wandb_group'] = wandb_params.get('group', None)
        defaults['wandb_run_name'] = wandb_params.get('run_name', None)
        defaults['wandb_tags'] = wandb_params.get('tags', None)
        defaults['wandb_mode'] = wandb_params.get('mode', 'online')
        defaults['wandb_watch'] = wandb_params.get('watch', 'gradients')
        defaults['wandb_watch_freq'] = wandb_params.get('watch_freq', 100)
        
        # データセット別のプリセットを適用（オプション）
        dataset = defaults['dataset']
        dataset_presets = params.get('dataset_presets', {})
        if dataset in dataset_presets:
            preset = dataset_presets[dataset]
            # プリセットの値で上書き（ただし、コマンドライン引数で上書き可能）
            for key, value in preset.items():
                if key in defaults:
                    defaults[key] = value
        
        return defaults
    except Exception as e:
        print(f"Warning: Failed to load params.yaml: {e}, using hardcoded defaults")
        return {}

def get_command_line_parser():
    # YAMLからデフォルト値を読み込む
    yaml_defaults = load_defaults_from_yaml()
    
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=yaml_defaults.get('project', PROJECT))
    parser.add_argument('-dataset', type=str, default=yaml_defaults.get('dataset', 'CICIDS2017_improved'),
                        choices=['mini_imagenet', 'cub200', 'cifar100', 'CICIDS2017_improved'])
    parser.add_argument('-dataroot', type=str, default=yaml_defaults.get('dataroot', DATA_DIR))
    parser.add_argument('-encoder', type=str, default=yaml_defaults.get('encoder', 'mlp'),
                        choices=['mlp', 'cnn1d'])

    # about pre-training
    parser.add_argument('-epochs_base', type=int, default=yaml_defaults.get('epochs_base', 400))
    parser.add_argument('-epochs_new', type=int, default=yaml_defaults.get('epochs_new', 100))
    parser.add_argument('-lr_base', type=float, default=yaml_defaults.get('lr_base', 0.005))
    parser.add_argument('-lr_new', type=float, default=yaml_defaults.get('lr_new', 0.1))
    parser.add_argument('-schedule', type=str, default=yaml_defaults.get('schedule', 'Milestone'),
                        choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-milestones', nargs='+', type=int, default=yaml_defaults.get('milestones', [50, 100, 150, 200, 250, 300]))
    parser.add_argument('-step', type=int, default=yaml_defaults.get('step', 20))
    parser.add_argument('-decay', type=float, default=yaml_defaults.get('decay', 0.0005))
    parser.add_argument('-momentum', type=float, default=yaml_defaults.get('momentum', 0.9))
    parser.add_argument('-gamma', type=float, default=yaml_defaults.get('gamma', 0.25))
    parser.add_argument('-temperature', type=float, default=yaml_defaults.get('temperature', 16))
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=yaml_defaults.get('batch_size_base', 256))
    parser.add_argument('-batch_size_new', type=int, default=yaml_defaults.get('batch_size_new', 0), help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=yaml_defaults.get('test_batch_size', 100))
    parser.add_argument('-base_mode', type=str, default=yaml_defaults.get('base_mode', 'ft_cos'),
                        choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default=yaml_defaults.get('new_mode', 'avg_cos'),
                        choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier

    #for fact
    parser.add_argument('-balance', type=float, default=yaml_defaults.get('balance', 0.01))
    parser.add_argument('-loss_iter', type=int, default=yaml_defaults.get('loss_iter', 0))
    parser.add_argument('-alpha', type=float, default=yaml_defaults.get('alpha', 2.0))
    parser.add_argument('-eta', type=float, default=yaml_defaults.get('eta', 0.1))

    parser.add_argument('-start_session', type=int, default=yaml_defaults.get('start_session', 0))
    parser.add_argument('-model_dir', type=str, default=yaml_defaults.get('model_dir', MODEL_DIR), help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')

    # about training
    parser.add_argument('-gpu', default=yaml_defaults.get('gpu', '0'))
    parser.add_argument('-num_workers', type=int, default=yaml_defaults.get('num_workers', 8))
    parser.add_argument('-seed', type=int, default=yaml_defaults.get('seed', 1))
    parser.add_argument('-debug', action='store_true')

    # wandb logging
    parser.add_argument('--use_wandb', action='store_true', help='Weights & Biasesへのログ送信を有効化')
    parser.add_argument('--wandb_project', type=str, default=yaml_defaults.get('wandb_project', None), help='wandbのプロジェクト名')
    parser.add_argument('--wandb_entity', type=str, default=yaml_defaults.get('wandb_entity', None), help='wandbのエンティティ（ユーザーまたはチーム）')
    parser.add_argument('--wandb_group', type=str, default=yaml_defaults.get('wandb_group', None), help='wandbのグループ名')
    parser.add_argument('--wandb_run_name', type=str, default=yaml_defaults.get('wandb_run_name', None), help='wandb上の実行名')
    parser.add_argument('--wandb_tags', nargs='*', default=yaml_defaults.get('wandb_tags', None), help='wandbタグ（スペース区切り）')
    parser.add_argument('--wandb_mode', type=str, default=yaml_defaults.get('wandb_mode', 'online'),
                        choices=['online', 'offline', 'disabled'], help='wandbの実行モード')
    parser.add_argument('--wandb_watch', type=str, default=yaml_defaults.get('wandb_watch', 'gradients'),
                        choices=['gradients', 'parameters', 'all', 'none'], help='wandb.watchの対象')
    parser.add_argument('--wandb_watch_freq', type=int, default=yaml_defaults.get('wandb_watch_freq', 100), help='wandb.watchのログ頻度（step単位）')
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    
    # not_data_initとset_no_valはaction='store_true'なので、YAMLから設定する場合は別途処理
    yaml_defaults = load_defaults_from_yaml()
    if yaml_defaults.get('not_data_init', False):
        args.not_data_init = True
    if yaml_defaults.get('set_no_val', False):
        args.set_no_val = True
    if yaml_defaults.get('debug', False):
        args.debug = True
    if yaml_defaults.get('use_wandb', False):
        args.use_wandb = True
    
    set_seed(args.seed)
    pprint(vars(args))
    args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()