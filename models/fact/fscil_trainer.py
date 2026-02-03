from .base import Trainer
import torch.nn as nn
from copy import deepcopy
import torch

from .helper import base_train, test, replace_base_fc
from utils import Averager, count_acc, confmatrix, get_dataset_label_names
from utils import init_comet_experiment, log_metrics_to_comet, log_confusion_matrix_to_comet
from dataloader.data_utils import get_base_dataloader, get_new_dataloader, set_up_datasets
import numpy as np
import time
import os
from utils import ensure_path, save_list_to_txt
import torch.nn.functional as F
from utils import count_acc_topk
from models.fact.Network import MYNET
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from db import init_db, Experiment, EpochMetric, SessionMetric
import math


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        # Comet ML実験を初期化
        self.comet_exp = init_comet_experiment(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        if self.args.num_gpu > 0 and torch.cuda.is_available():
            # DataParallel は全パラメータが device_ids[0] (cuda:0) にあることを要求する
            device_ids = list(range(self.args.num_gpu))
            self.model = self.model.to(f'cuda:{device_ids[0]}')
            self.model = nn.DataParallel(self.model, device_ids=device_ids)
            self.device = torch.device(f'cuda:{device_ids[0]}')
        else:
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, weights_only=False)['params']
            
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
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

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
        
        # キャッシュに保存（ベースセッションのテストデータは全セッションで再利用可能）
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

        # PeeWee: one experiment row per run, all params as columns
        init_db()
        db_exp = Experiment.from_args(args)
        self.db_experiment_id = db_exp.id

        #gen_mask
        masknum=3
        # masknumがbase_classより大きい場合はbase_classに制限
        actual_masknum = min(masknum, args.base_class)
        mask=np.zeros((args.base_class,args.num_classes))
        for i in range(args.num_classes-args.base_class):
            picked_dummy=np.random.choice(args.base_class, actual_masknum, replace=False)
            mask[:,i+args.base_class][picked_dummy]=1
        mask=torch.tensor(mask).to(self.device)



        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args,mask)
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
                    
                    # Comet MLにメトリクスをログ
                    log_metrics_to_comet(
                        self.comet_exp,
                        {
                            'train_loss': tl,
                            'train_acc': ta,
                            'test_loss': tsl,
                            'test_acc': tsa,
                            'learning_rate': lrc
                        },
                        epoch=epoch,
                        session=session
                    )
                    
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    # None値とnan値のチェックとデフォルト値の設定
                    def safe_float(value, default=0.0):
                        """Noneまたはnanの場合はデフォルト値を返す"""
                        if value is None:
                            return default
                        try:
                            fval = float(value)
                            if math.isnan(fval) or math.isinf(fval):
                                return default
                            return fval
                        except (ValueError, TypeError):
                            return default
                    
                    train_loss = safe_float(tl, 0.0)
                    train_acc = safe_float(ta, 0.0)
                    test_loss = safe_float(tsl, 0.0)
                    test_acc = safe_float(tsa, 0.0)
                    learning_rate = safe_float(lrc, 0.0)
                    EpochMetric.create(
                        experiment=self.db_experiment_id,
                        session=session,
                        epoch=epoch,
                        train_loss=train_loss,
                        train_acc=train_acc,
                        test_loss=test_loss,
                        test_acc=test_acc,
                        learning_rate=learning_rate,
                    )
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                
                # Comet MLにセッションごとの最高精度をログ
                log_metrics_to_comet(
                    self.comet_exp,
                    {
                        'max_acc': self.trlog['max_acc'][session] / 100.0,
                        'max_acc_epoch': self.trlog['max_acc_epoch']
                    },
                    session=session
                )

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
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                #save dummy classifiers
                model_module = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                self.dummy_classifiers=deepcopy(model_module.fc.weight.detach())
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]
                model_module.mode = 'avg_cos'
                # ベースセッションの最終評価（validation=Falseで実行）
                test_result = test(self.model, testloader, 0, args, session, validation=False)
                if len(test_result) == 3:
                    tsl_final, tsa_final, metrics_dict = test_result
                    SessionMetric.create(
                        experiment=self.db_experiment_id,
                        session=session,
                        max_acc=self.trlog['max_acc'][session],
                        test_loss=metrics_dict['test_loss'],
                        accuracy=metrics_dict['accuracy'],
                        precision_macro=metrics_dict['precision_macro'],
                        recall_macro=metrics_dict['recall_macro'],
                        f1_macro=metrics_dict['f1_macro'],
                        seen_acc=metrics_dict['seen_acc'],
                        unseen_acc=metrics_dict['unseen_acc'],
                    )
                else:
                    tsl_final, tsa_final = test_result
                
                # セッション0の最終テスト結果をComet MLにログ
                log_metrics_to_comet(
                    self.comet_exp,
                    {
                        'final_test_loss': tsl_final,
                        'final_test_acc': tsa_final
                    },
                    session=session
                )
                
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

                # 訓練セット内の新しいクラスのみを抽出
                all_classes = np.unique(train_set.targets)
                print(f"{all_classes.tolist()} vs {args.base_labels}")
                # new_classes_in_data = np.intersect1d(all_classes, new_classes)
                if all_classes.tolist() in args.base_labels:
                    # より詳細なエラーメッセージを提供
                    session_file = f"data/index_list/{args.dataset_name}/session_{session + 1}.txt"
                    error_msg = (
                        f"Session {session}: No new classes found in training data.\n"
                        f"  Expected classes: except base classes: {args.base_class.tolist()}\n"
                        f"  Expected classes: except base classes: {args.base_class.tolist()}\n"
                        f"  Found classes in training data: {all_classes.tolist()}\n"
                        f"  Training samples loaded: {len(train_set)}\n"
                        f"  Session file: {session_file}\n"
                        f"  This usually means the session file contains wrong data indices.\n"
                        f"  Please regenerate session files using: uv run create_session_files.py"
                    )
                    raise ValueError(error_msg)
                # new_classes_in_data = data of all_classes
                model_module.update_fc(trainloader, train_set.targets, session)

                #tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)
                #tsl, tsa = test_withfc(self.model, testloader, 0, args, session,validation=False)
                test_result = self.test_intergrate(self.model, testloader, 0, args, session, validation=False)
                if len(test_result) == 3:
                    tsl, tsa, metrics_dict = test_result
                    SessionMetric.create(
                        experiment=self.db_experiment_id,
                        session=session,
                        max_acc=self.trlog['max_acc'][session],
                        test_loss=metrics_dict['test_loss'],
                        accuracy=metrics_dict['accuracy'],
                        precision_macro=metrics_dict['precision_macro'],
                        recall_macro=metrics_dict['recall_macro'],
                        f1_macro=metrics_dict['f1_macro'],
                        seen_acc=metrics_dict['seen_acc'],
                        unseen_acc=metrics_dict['unseen_acc'],
                    )
                else:
                    tsl, tsa = test_result
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                # Comet MLにセッションごとの精度をログ
                log_metrics_to_comet(
                    self.comet_exp,
                    {
                        'test_acc': tsa,
                        'test_loss': tsl,
                        'max_acc': self.trlog['max_acc'][session] / 100.0
                    },
                    session=session
                )

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        # 最終結果をComet MLにログ
        log_metrics_to_comet(
            self.comet_exp,
            {
                'total_time_minutes': (time.time() - t_start_time) / 60,
                'best_epoch': self.trlog['max_acc_epoch']
            }
        )
        
        # 全セッションの最高精度をログ
        for sess_idx, max_acc in enumerate(self.trlog['max_acc']):
            log_metrics_to_comet(
                self.comet_exp,
                {'session_max_acc': max_acc / 100.0},
                session=sess_idx
            )

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        
        # Comet ML実験を終了
        if self.comet_exp is not None:
            self.comet_exp.end()
            print('Comet ML experiment ended')


    def test_intergrate(self, model, testloader, epoch,args, session,validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        # DataParallelの場合はmodule、そうでない場合はmodel自体を使用
        model_module = model.module if isinstance(model, nn.DataParallel) else model
        vl = Averager()
        va = Averager()
        va5= Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model_module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        
        eta=args.eta
        
        softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.to(self.device) for _ in batch]
                
                emb=model_module.encode(data)
            
                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                k = min(40, proj.size(1))
                topk, indices = torch.topk(proj, k)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1=torch.mm(res_logit,proj_matrix)
                logits2 = model_module.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logits1,dim=1)+(1-eta)*F.softmax(logits2,dim=1)
            
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc=count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)
                # CPU環境では既にCPU上にあるため、.cpu()は不要
                if self.device.type == 'cuda':
                    lgt=torch.cat([lgt,logits.cpu()])
                    lbs=torch.cat([lbs,test_label.cpu()])
                else:
                    lgt=torch.cat([lgt,logits])
                    lbs=torch.cat([lbs,test_label])
            vl = vl.item()
            va = va.item()
            va5= va5.item()
            print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va,va5))

        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            label_names = get_dataset_label_names(testloader.dataset)
            if label_names:
                label_names = label_names[:test_class]
            cm=confmatrix(lgt,lbs,save_model_dir, label_names=label_names)
            perclassacc=cm.diagonal()
            seen_slice = perclassacc[:args.base_class] if len(perclassacc) >= args.base_class else perclassacc
            seenac=np.mean(seen_slice) if len(seen_slice) > 0 else float('nan')
            unseen_slice = perclassacc[args.base_class:]
            unseenac=np.mean(unseen_slice) if len(unseen_slice) > 0 else float('nan')
            print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
            
            # 予測ラベルを取得
            pred = torch.argmax(lgt, dim=1)
            
            # y_trueとy_predをnumpy配列に変換
            y_true_np = lbs.numpy() if isinstance(lbs, torch.Tensor) else lbs
            y_pred_np = pred.numpy() if isinstance(pred, torch.Tensor) else pred
            
            # accuracy, precision, recall, f1-scoreを計算
            accuracy = accuracy_score(y_true_np, y_pred_np)
            precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
            
            # クラスごとのprecision, recall, f1-scoreも計算
            precision_per_class = precision_score(y_true_np, y_pred_np, average=None, zero_division=0)
            recall_per_class = recall_score(y_true_np, y_pred_np, average=None, zero_division=0)
            f1_per_class = f1_score(y_true_np, y_pred_np, average=None, zero_division=0)
            
            print(f'Session {session} Metrics:')
            print(f'  Accuracy: {accuracy:.4f}')
            print(f'  Precision (macro): {precision:.4f}')
            print(f'  Recall (macro): {recall:.4f}')
            print(f'  F1-score (macro): {f1:.4f}')
            
            # メトリクスをファイルに保存
            metrics_file = os.path.join(args.save_path, f'session_{session}_metrics.txt')
            with open(metrics_file, 'w') as f:
                f.write(f'Session {session} Evaluation Metrics\n')
                f.write('=' * 50 + '\n\n')
                f.write(f'Overall Metrics:\n')
                f.write(f'  Accuracy: {accuracy:.4f}\n')
                f.write(f'  Precision (macro): {precision:.4f}\n')
                f.write(f'  Recall (macro): {recall:.4f}\n')
                f.write(f'  F1-score (macro): {f1:.4f}\n\n')
                f.write(f'Per-class Metrics:\n')
                for i, (p, r, f1_val) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                    class_name = label_names[i] if label_names and i < len(label_names) else f'Class {i}'
                    f.write(f'  {class_name}:\n')
                    f.write(f'    Precision: {p:.4f}\n')
                    f.write(f'    Recall: {r:.4f}\n')
                    f.write(f'    F1-score: {f1_val:.4f}\n')
                f.write(f'\nSeen Classes Accuracy: {seenac:.4f}\n')
                f.write(f'Unseen Classes Accuracy: {unseenac:.4f}\n')
            
            # Comet MLに混同行列をログ（log_confusion_matrix APIを使用）
            log_confusion_matrix_to_comet(
                self.comet_exp,
                y_true=lbs,
                y_pred=pred,
                labels=label_names,
                session=session,
                title=f"Session {session} Confusion Matrix"
            )
            
            # Comet MLにメトリクスをログ
            log_metrics_to_comet(
                self.comet_exp,
                {
                    'accuracy': accuracy,
                    'precision_macro': precision,
                    'recall_macro': recall,
                    'f1_score_macro': f1,
                    'seen_acc': seenac if not np.isnan(seenac) else 0.0,
                    'unseen_acc': unseenac if not np.isnan(unseenac) else 0.0
                },
                session=session
            )

            metrics_dict = {
                'test_loss': vl,
                'accuracy': float(accuracy),
                'precision_macro': float(precision),
                'recall_macro': float(recall),
                'f1_macro': float(f1),
                'seen_acc': float(seenac) if not np.isnan(seenac) else 0.0,
                'unseen_acc': float(unseenac) if not np.isnan(unseenac) else 0.0,
            }
            return vl, va, metrics_dict
        return vl, va

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset_name
        
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
