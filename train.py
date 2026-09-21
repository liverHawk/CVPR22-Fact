import argparse
import importlib
from utils import *
import wandb

MODEL_DIR=None
DATA_DIR = 'data/'
PROJECT='base'

def get_command_line_parser():
    parser = argparse.ArgumentParser()

    # about dataset and network
    parser.add_argument('-project', type=str, default=PROJECT)
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)

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
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    
    # wandb configuration
    parser.add_argument('--use_wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='FACT-FSCIL', help='wandb project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (username or team)')
    parser.add_argument('--wandb_cm_freq', type=int, default=20, help='log lightweight confusion-matrix image every N base epochs (0 disables)')
    
    return parser


if __name__ == '__main__':
    parser = get_command_line_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    pprint(vars(args))
    
    # Initialize wandb if requested
    if args.use_wandb:
        run_name = f"{args.dataset}_{args.project}_session{args.start_session}"
        config_dict = vars(args)
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config_dict,
            notes="Few-Shot Class Incremental Learning with FACT"
        )
    
    if args.gpu == "":
        args.device = "cpu"
        args.num_gpu = 0
    else:
        args.device = "cuda"
        args.num_gpu = set_gpu(args)
    trainer = importlib.import_module('models.%s.fscil_trainer' % (args.project)).FSCILTrainer(args)
    trainer.train()
    
    # Log final results and finish wandb run
    if args.use_wandb:
        # Save the best metrics to wandb
        final_metrics = {
            'best_acc_session_0': trainer.trlog['max_acc'][0],
            'best_epoch_base': trainer.trlog['max_acc_epoch'],
        }
        
        # Handle max_acc as a list - log individual session accuracies for chart
        max_acc = trainer.trlog['max_acc']
        if isinstance(max_acc, (list, tuple)):
            # Log average accuracy across all sessions
            avg_accuracy = sum(float(acc) for acc in max_acc) / len(max_acc)
            final_metrics['avg_test_accuracy'] = float(avg_accuracy)
            
            # Also log each session's accuracy separately for reference
            for i, acc in enumerate(max_acc):
                final_metrics[f'session_{i}_acc'] = float(acc)
        else:
            final_metrics['avg_test_accuracy'] = float(max_acc.tolist()[0])
        
        wandb.log(final_metrics)
        wandb.finish()