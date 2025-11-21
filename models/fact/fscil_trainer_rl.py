"""
FACT + 強化学習（DQN）統合トレーナー

Base session (session 0): 教師あり学習でFACTモデルを訓練
Incremental sessions (session 1+): 強化学習（DQN）でFew-shotクラスを学習
"""
from .base import Trainer
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from .helper import *
from .rl_components import FACTDQNHead, ReplayBuffer, RewardCalculator
from .rl_trainer_helper import rl_train_session, rl_test
from utils import *
from dataloader.data_utils import *


class FSCILTrainerRL(Trainer):
    """
    強化学習を統合したFSCILトレーナー
    Session 0: 教師あり学習
    Session 1+: 強化学習（DQN）
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        # FACTモデルの初期化
        self.model = MYNET(self.args, mode=self.args.base_mode)
        if getattr(self.args, 'use_cuda', True) and self.args.num_gpu > 0:
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.to(self.args.device)
        self.wandb.watch(self.model)

        # モデルのロード
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARNING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        # 強化学習用の変数（後で初期化）
        self.dqn_head = None
        self.target_head = None
        self.replay_buffer = None
        self.reward_calculator = None

    def get_optimizer_base(self):
        """Base sessionの教師あり学習用オプティマイザ"""
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.args.lr_base,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.args.decay
        )
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.step, gamma=self.args.gamma
            )
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=self.args.gamma
            )
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs_base
            )

        return optimizer, scheduler

    def get_optimizer_rl(self):
        """強化学習用オプティマイザ"""
        lr_rl = getattr(self.args, 'rl_lr', 1e-3)
        optimizer = torch.optim.Adam(self.dqn_head.parameters(), lr=lr_rl)
        return optimizer

    def get_dataloader(self, session):
        """データローダー取得"""
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def initialize_rl_components(self):
        """強化学習コンポーネントの初期化"""
        print("Initializing RL components...")

        # Embedding dimension
        from utils import get_model_module
        model_module = get_model_module(self.model)
        embedding_dim = model_module.num_features

        # Virtual classes for FACT
        virtual_classes = getattr(self.args, 'rl_virtual_classes', 0)

        # DQN head
        self.dqn_head = FACTDQNHead(
            embedding_dim=embedding_dim,
            num_actions=self.args.num_classes,
            virtual_classes=virtual_classes,
            temperature=self.args.temperature,
            use_cosine='cos' in self.args.new_mode,
        ).to(self.args.device)

        # Target network
        self.target_head = FACTDQNHead(
            embedding_dim=embedding_dim,
            num_actions=self.args.num_classes,
            virtual_classes=virtual_classes,
            temperature=self.args.temperature,
            use_cosine='cos' in self.args.new_mode,
        ).to(self.args.device)
        self.target_head.load_state_dict(self.dqn_head.state_dict())

        # Replay buffer
        buffer_size = getattr(self.args, 'rl_buffer_size', 50000)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            state_dim=embedding_dim,
            device=self.args.device
        )

        # Reward calculator
        reward_type = getattr(self.args, 'rl_reward_type', 'simple')
        self.reward_calculator = RewardCalculator(
            reward_type=reward_type,
            reward_correct=getattr(self.args, 'rl_reward_correct', 1.0),
            reward_incorrect=getattr(self.args, 'rl_reward_incorrect', -1.0),
            temperature=self.args.temperature,
        )

        print(f"RL components initialized with reward_type={reward_type}")

    def freeze_encoder(self):
        """エンコーダを固定"""
        print("Freezing FACT encoder...")
        from utils import get_model_module
        model_module = get_model_module(self.model)
        for param in model_module.encoder.parameters():
            param.requires_grad = False
        self.model.eval()

    def train(self):
        """メイン訓練ループ"""
        args = self.args
        t_start_time = time.time()

        # 訓練統計の初期化
        result_list = [args]

        # FACT用マスク生成
        masknum = 3
        mask = np.zeros((args.base_class, args.num_classes))
        for i in range(args.num_classes - args.base_class):
            picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
            mask[:, i + args.base_class][picked_dummy] = 1
        mask = torch.tensor(mask).to(self.args.device)

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)

            if session == 0:
                # ==================== Base Session: 教師あり学習 ====================
                print('=' * 50)
                print(f'Base Session {session}: Supervised Learning with FACT')
                print('=' * 50)
                print('New classes for this session:\n', np.unique(train_set.targets))

                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()

                    # 訓練
                    tl, ta = base_train(
                        self.model, trainloader, optimizer, scheduler, epoch, args, mask
                    )

                    # テスト
                    tsl, tsa = test(
                        self.model, testloader, epoch, args, session, wandb_logger=self.wandb
                    )

                    # ベストモデル保存
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(
                            args.save_path, f'session{session}_max_acc.pth'
                        )
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(
                            optimizer.state_dict(),
                            os.path.join(args.save_path, 'optimizer_best.pth')
                        )
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)

                    print(
                        'best epoch {}, best test acc={:.3f}'.format(
                            self.trlog['max_acc_epoch'], self.trlog['max_acc'][session]
                        )
                    )

                    # ログ記録
                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        f'epoch:{epoch:03d},lr:{lrc:.4f},training_loss:{tl:.5f},'
                        f'training_acc:{ta:.5f},test_loss:{tsl:.5f},test_acc:{tsa:.5f}'
                    )
                    self.global_step += 1
                    self.wandb.log_metrics(
                        {
                            'session': session,
                            'epoch': epoch,
                            'lr': lrc,
                            'train/loss': tl,
                            'train/acc': ta,
                            'test/loss': tsl,
                            'test/acc': tsa,
                        },
                        step=self.global_step
                    )
                    print(
                        f'This epoch takes {time.time() - start_time:.0f} seconds\n'
                        f'still need around {(time.time() - start_time) * (args.epochs_base - epoch) / 60:.2f} mins to finish this session'
                    )
                    scheduler.step()

                result_list.append(
                    f'Session {session}, Test Best Epoch {self.trlog["max_acc_epoch"]},\n'
                    f'best test Acc {self.trlog["max_acc"][session]:.4f}\n'
                )

                # Data initialization
                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    transform = getattr(testloader.dataset, 'transform', None) if args.dataset != 'CICIDS2017_improved' else None
                    self.model = replace_base_fc(train_set, transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, f'session{session}_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    from utils import get_model_module
                    model_module = get_model_module(self.model)
                    model_module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session, wandb_logger=self.wandb)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print(f'The new best test acc of base session={self.trlog["max_acc"][session]:.3f}')

                # Dummy classifiers保存
                from utils import get_model_module
                model_module = get_model_module(self.model)
                self.dummy_classifiers = deepcopy(model_module.fc.weight.detach())
                self.dummy_classifiers = F.normalize(
                    self.dummy_classifiers[self.args.base_class:, :], p=2, dim=-1
                )
                self.old_classifiers = self.dummy_classifiers[:self.args.base_class, :]

                # 強化学習コンポーネントの初期化
                self.initialize_rl_components()
                self.freeze_encoder()

            else:
                # ==================== Incremental Sessions: 強化学習 ====================
                print('=' * 50)
                print(f'Incremental Session {session}: Reinforcement Learning with DQN')
                print('=' * 50)
                print(f"Training session: [{session}]")

                # 強化学習で訓練
                optimizer_rl = self.get_optimizer_rl()

                tl, ta, tr = rl_train_session(
                    model=self.model,
                    dqn_head=self.dqn_head,
                    target_head=self.target_head,
                    trainloader=trainloader,
                    replay_buffer=self.replay_buffer,
                    optimizer=optimizer_rl,
                    args=args,
                    session=session,
                    reward_calculator=self.reward_calculator,
                )

                print(f'Session {session} RL Training: loss={tl:.4f}, acc={ta:.4f}, avg_reward={tr:.4f}')

                # テスト
                tsl, tsa = rl_test(
                    model=self.model,
                    dqn_head=self.dqn_head,
                    testloader=testloader,
                    args=args,
                    session=session,
                    validation=False,
                    wandb_logger=self.wandb
                )

                # モデル保存
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, f'session{session}_rl_max_acc.pth')
                torch.save(
                    {
                        'model_params': self.model.state_dict(),
                        'dqn_head_params': self.dqn_head.state_dict(),
                    },
                    save_model_dir
                )
                print('Saving RL model to :%s' % save_model_dir)
                print(f'  test acc={self.trlog["max_acc"][session]:.3f}')

                result_list.append(
                    f'Session {session}, test Acc {self.trlog["max_acc"][session]:.3f}\n'
                )

                # Wandb logging
                self.global_step += 1
                self.wandb.log_metrics(
                    {
                        'session': session,
                        'epoch': 0,
                        'train/rl_loss': tl,
                        'train/rl_acc': ta,
                        'train/rl_reward': tr,
                        'test/loss': tsl,
                        'test/acc': tsa,
                    },
                    step=self.global_step
                )

        # 最終結果
        result_list.append(f'Base Session Best Epoch {self.trlog["max_acc_epoch"]}\n')
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print(f'Total time used {total_time:.2f} mins')

        # Wandb summary
        summary_payload = {
            f'session_{idx}_best_acc': acc for idx, acc in enumerate(self.trlog['max_acc'])
        }
        summary_payload['base_best_epoch'] = self.trlog['max_acc_epoch']
        summary_payload['total_time_min'] = total_time
        self.wandb.set_summary(**summary_payload)
        self.finalize()

    def set_save_path(self):
        """保存パスの設定"""
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        mode = mode + '-RL'  # 強化学習モード

        self.args.save_path = f'{self.args.dataset}/'
        self.args.save_path = self.args.save_path + f'{self.args.project}/'

        self.args.save_path = self.args.save_path + f'{mode}-start_{self.args.start_session}/'

        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + (
                f'Epo_{self.args.epochs_base}-Lr_{self.args.lr_base:.4f}-MS_{mile_stone}-'
                f'Gam_{self.args.gamma:.2f}-Bs_{self.args.batch_size_base}-Mom_{self.args.momentum:.2f}'
            )
            self.args.save_path = self.args.save_path + f'Bal{self.args.balance:.2f}-LossIter{self.args.loss_iter}'
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + (
                f'Epo_{self.args.epochs_base}-Lr_{self.args.lr_base:.4f}-Step_{self.args.step}-'
                f'Gam_{self.args.gamma:.2f}-Bs_{self.args.batch_size_base}-Mom_{self.args.momentum:.2f}'
            )
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + (
                f'Cosine-Epo_{self.args.epochs_base}-Lr_{self.args.lr_base:.4f}'
            )
            self.args.save_path = self.args.save_path + f'Bal{self.args.balance:.2f}-LossIter{self.args.loss_iter}'

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + f'-T_{self.args.temperature:.2f}'

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + (
                f'-ftLR_{self.args.lr_new:.3f}-ftEpoch_{self.args.epochs_new}'
            )

        # RL parameters
        rl_reward_type = getattr(self.args, 'rl_reward_type', 'simple')
        self.args.save_path = self.args.save_path + f'-RLReward_{rl_reward_type}'

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
