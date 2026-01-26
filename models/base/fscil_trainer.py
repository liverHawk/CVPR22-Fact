from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import torch

from .helper import *
from utils import *
from dataloader.data_utils import *


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        if self.args.num_gpu > 0 and torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
            self.model = self.model.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, weights_only=False)['params']
            #self.best_model_dict = torch.load(self.args.model_dir)['state_dict']
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
        
        # データセットキャッシュ（重複読み込みを防ぐ）
        self.dataset_cache = {}

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)

        return optimizer, scheduler

    def get_dataloader(self, session):
        # キャッシュキーを生成
        cache_key = f"session_{session}"
        
        # キャッシュに存在する場合は再利用
        if cache_key in self.dataset_cache:
            print(f"Using cached dataset for session {session}")
            return self.dataset_cache[cache_key]
        
        # データセットを読み込む
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        
        # キャッシュに保存
        self.dataset_cache[cache_key] = (trainset, trainloader, testloader)
        
        # ベースセッションのテストデータをキャッシュ（全セッションで再利用）
        if session == 0:
            self.base_testloader = testloader
        
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    dataset_transform = getattr(testloader.dataset, "transform", None)
                    self.model = replace_base_fc(train_set, dataset_transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    model_module.mode = 'avg_cos'
                    # ベースセッションの最終評価（validation=Falseで実行）
                    tsl_final, tsa_final = test(self.model, testloader, 0, args, session, validation=False)
                    if (tsa_final * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa_final * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                    
                    # ベースセッションの最終評価結果を記録
                    result_list.append('Session {} Final Test, loss={:.4f}, acc={:.4f}\n'.format(
                        session, tsl_final, tsa_final * 100))
                    print('Session {} Final Test: loss={:.4f}, acc={:.4f}'.format(
                        session, tsl_final, tsa_final * 100))


            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                model_module.mode = self.args.new_mode
                self.model.eval()
                if hasattr(testloader.dataset, "transform"):
                    trainloader.dataset.transform = testloader.dataset.transform
                # 新規セッションでは、新しいクラスのみを処理
                new_class_start = args.base_class + args.way * (session - 1)
                new_class_end = args.base_class + args.way * session
                new_classes = np.arange(new_class_start, new_class_end)
                # 訓練セット内の新しいクラスのみを抽出
                all_classes = np.unique(train_set.targets)
                new_classes_in_data = np.intersect1d(all_classes, new_classes)
                if len(new_classes_in_data) == 0:
                    # より詳細なエラーメッセージを提供
                    session_file = f"data/index_list/{args.dataset_name}/session_{session + 1}.txt"
                    error_msg = (
                        f"Session {session}: No new classes found in training data.\n"
                        f"  Expected classes: {new_classes.tolist()}\n"
                        f"  Found classes in training data: {all_classes.tolist()}\n"
                        f"  Training samples loaded: {len(train_set)}\n"
                        f"  Session file: {session_file}\n"
                        f"  This usually means the session file contains wrong data indices.\n"
                        f"  Please regenerate session files using: uv run create_session_files.py"
                    )
                    raise ValueError(error_msg)
                model_module.update_fc(trainloader, new_classes_in_data, session)

                tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset_name
        
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
