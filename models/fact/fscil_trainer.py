from .base import Trainer
import os.path as osp
import torch.nn as nn

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
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        self.wandb.watch(self.model)

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            # state_dict()は既に新しい辞書を返すので、deepcopyは不要（高速化）
            self.best_model_dict = dict(self.model.state_dict())

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
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        #gen_mask
        masknum=3
        mask=np.zeros((args.base_class,args.num_classes))
        for i in range(args.num_classes-args.base_class):
            picked_dummy=np.random.choice(args.base_class,masknum,replace=False)
            mask[:,i+args.base_class][picked_dummy]=1
        mask=torch.tensor(mask).cuda()



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
                    test_result = test(self.model, testloader, epoch, args, session, 
                                     wandb_logger=self.wandb,
                                     enable_unknown_detection=getattr(args, 'enable_unknown_detection', False),
                                     distance_type=getattr(args, 'distance_type', 'cosine'),
                                     distance_threshold=getattr(args, 'distance_threshold', None))
                    if isinstance(test_result, tuple) and len(test_result) == 3:
                        tsl, tsa, _ = test_result  # unknown_statsは無視（base sessionでは不要）
                    else:
                        tsl, tsa = test_result

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        # state_dict()は既に新しい辞書を返すので、deepcopyは不要（高速化）
                        self.best_model_dict = dict(self.model.state_dict())
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
                    self.global_step += 1
                    self.wandb.log_metrics({
                        'session': session,
                        'epoch': epoch,
                        'lr': lrc,
                        'train/loss': tl,
                        'train/acc': ta,
                        'test/loss': tsl,
                        'test/acc': tsa,
                    }, step=self.global_step)
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    # For CICIDS2017_improved (tabular data), transform is not needed
                    transform = getattr(testloader.dataset, 'transform', None) if args.dataset != 'CICIDS2017_improved' else None
                    self.model = replace_base_fc(train_set, transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    # state_dict()は既に新しい辞書を返すので、deepcopyは不要（高速化）
                    self.best_model_dict = dict(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    test_result = test(self.model, testloader, 0, args, session, 
                                      wandb_logger=self.wandb,
                                      enable_unknown_detection=getattr(args, 'enable_unknown_detection', False),
                                      distance_type=getattr(args, 'distance_type', 'cosine'),
                                      distance_threshold=getattr(args, 'distance_threshold', None))
                    if isinstance(test_result, tuple) and len(test_result) == 3:
                        tsl, tsa, _ = test_result  # unknown_statsは無視（base sessionでは不要）
                    else:
                        tsl, tsa = test_result
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                #save dummy classifiers
                # テンソルの場合はclone()を使用（deepcopyより高速）
                self.dummy_classifiers=self.model.module.fc.weight.detach().clone()
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                # Only set transform for image datasets (CICIDS2017_improved doesn't have transform attribute)
                if hasattr(trainloader.dataset, 'transform') and hasattr(testloader.dataset, 'transform'):
                    trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                #tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)
                #tsl, tsa = test_withfc(self.model, testloader, 0, args, session,validation=False)
                test_result = self.test_intergrate(
                    self.model, testloader, 0, args, session, validation=False, 
                    wandb_logger=self.wandb,
                    enable_unknown_detection=getattr(args, 'enable_unknown_detection', False),
                    distance_type=getattr(args, 'distance_type', 'cosine'),
                    distance_threshold=getattr(args, 'distance_threshold', None)
                )
                
                if isinstance(test_result, tuple) and len(test_result) == 3:
                    tsl, tsa, unknown_stats = test_result
                else:
                    tsl, tsa = test_result
                    unknown_stats = None
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                # state_dict()は既に新しい辞書を返すので、deepcopyは不要（高速化）
                self.best_model_dict = dict(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))
                
                # 未知クラス検出の結果をunseen accのように表示
                if getattr(args, 'enable_unknown_detection', False) and unknown_stats and unknown_stats['total_samples'] > 0:
                    unknown_rate = 100 * unknown_stats['total_unknown_detected'] / unknown_stats['total_samples']
                    avg_distance = sum(unknown_stats['unknown_distances']) / len(unknown_stats['unknown_distances']) if unknown_stats['unknown_distances'] else 0.0
                    print('  Unknown Detection Rate: {:.2f}% (avg_distance={:.4f})'.format(unknown_rate, avg_distance))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
                self.global_step += 1
                self.wandb.log_metrics({
                    'session': session,
                    'epoch': 0,
                    'test/loss': tsl,
                    'test/acc': tsa,
                }, step=self.global_step)

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        summary_payload = {f'session_{idx}_best_acc': acc for idx, acc in enumerate(self.trlog['max_acc'])}
        summary_payload['base_best_epoch'] = self.trlog['max_acc_epoch']
        summary_payload['total_time_min'] = total_time
        self.wandb.set_summary(**summary_payload)
        self.finalize()


    def test_intergrate(self, model, testloader, epoch, args, session, validation=True, wandb_logger=None,
                       enable_unknown_detection=False, distance_type='cosine', distance_threshold=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5= Averager()
        # リストに蓄積してから最後にまとめてcat（高速化）
        lgt_list = []
        lbs_list = []

        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(model.module.fc.weight[:test_class, :],1,0),p=2,dim=-1))
        
        eta=args.eta
        
        softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)

        # 未知クラス検出用の統計
        unknown_stats = {
            'total_unknown_detected': 0,
            'total_samples': 0,
            'unknown_distances': [],
        } if enable_unknown_detection else None

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                
                emb=model.module.encode(data)
            
                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                # Adjust k based on actual dimension (for datasets with fewer classes like CICIDS2017_improved)
                k = min(40, proj.size(1))
                topk, indices = torch.topk(proj, k)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1=torch.mm(res_logit,proj_matrix)
                logits2 = model.module.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logits1,dim=1)+(1-eta)*F.softmax(logits2,dim=1)
            
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc=count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)
                
                # 未知クラス検出が有効な場合
                if enable_unknown_detection:
                    known_class_indices = list(range(test_class))
                    is_unknown, distances, nearest_class = model.module.detect_unknown_by_distance(
                        data,
                        known_class_indices=known_class_indices,
                        distance_threshold=distance_threshold,
                        distance_type=distance_type
                    )
                    unknown_stats['total_samples'] += data.size(0)
                    unknown_stats['total_unknown_detected'] += is_unknown.sum().item()
                    unknown_stats['unknown_distances'].extend(distances[is_unknown].cpu().tolist())
                
                lgt_list.append(logits.cpu())
                lbs_list.append(test_label.cpu())
            vl = vl.item()
            va = va.item()
            va5= va5.item()
            
            # 未知クラス検出の結果を表示
            if enable_unknown_detection and unknown_stats['total_samples'] > 0:
                unknown_rate = 100 * unknown_stats['total_unknown_detected'] / unknown_stats['total_samples']
                avg_distance = sum(unknown_stats['unknown_distances']) / len(unknown_stats['unknown_distances']) if unknown_stats['unknown_distances'] else 0.0
                print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}, unknown_detected={:.2f}% (avg_dist={:.4f})'.format(
                    epoch, vl, va, va5, unknown_rate, avg_distance))
            else:
                print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch, vl, va,va5))
            
            # 最後にまとめてcat（メモリ効率と速度が向上）
            lgt = torch.cat(lgt_list, dim=0)
            lbs = torch.cat(lbs_list, dim=0)

            lgt=lgt.view(-1,test_class)
            lbs=lbs.view(-1)
            if validation is not True:
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
                # ラベル名を取得（CICIDS2017_improvedの場合）
                label_names = None
                if args.dataset == 'CICIDS2017_improved' and hasattr(testloader.dataset, 'label_encoder'):
                    label_names = list(testloader.dataset.label_encoder.classes_)
                cm=confmatrix(lgt,lbs,save_model_dir, label_names=label_names)
                perclassacc=cm.diagonal()
                seenac=np.mean(perclassacc[:args.base_class])
                unseenac=np.mean(perclassacc[args.base_class:])
                print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
                # Classification reportを保存
                from utils import save_classification_report
                save_classification_report(lgt, lbs, save_model_dir)
                if wandb_logger is not None:
                    wandb_logger.log_image(f'session_{session}_confusion_matrix', save_model_dir + '.png')
            
        # 未知クラス検出の統計を返す
        return vl, va, unknown_stats if enable_unknown_detection else None

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)

        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
