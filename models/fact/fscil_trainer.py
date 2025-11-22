from .base import Trainer
import torch
import torch.nn as nn
import time
import numpy as np
import os
from copy import deepcopy
import torch.nn.functional as F
from .helper import base_train, test, replace_base_fc
from utils import ensure_path, save_list_to_txt, Averager, count_acc, count_acc_topk
from dataloader.data_utils import set_up_datasets, get_base_dataloader, get_new_dataloader
from models.fact.Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        self.wandb.watch(self.model)

        if self.args.model_dir is not None and os.path.exists(self.args.model_dir):
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, weights_only=False)['params']
            
        else:
            if self.args.model_dir is not None:
                print('Model file not found: %s, using random init params' % self.args.model_dir)
            else:
                print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

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
        mask=torch.tensor(mask).to(self.device)

        print(f"session: {args.start_session} to {args.sessions-1}")
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args, mask, self.device)
                    # test model with all seen class
                    # 最終エポックの場合のみ混同行列を作成
                    is_final_epoch = (epoch == args.epochs_base - 1)
                    tsl, tsa = test(self.model, testloader, epoch, args, session, validation=not is_final_epoch, wandb_logger=self.wandb)

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
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session, validation=False, wandb_logger=self.wandb)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                #save dummy classifiers
                # Handle both DataParallel and single GPU/CPU models
                if hasattr(self.model, 'module'):
                    self.dummy_classifiers=deepcopy(self.model.module.fc.weight.detach())
                else:
                    self.dummy_classifiers=deepcopy(self.model.fc.weight.detach())
                
                self.dummy_classifiers=F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]

            else:  # incremental learning sessions
                print("=" * 100)
                print("training session: [%d]" % session)

                # Handle both DataParallel and single GPU/CPU models
                if hasattr(self.model, 'module'):
                    self.model.module.mode = self.args.new_mode
                else:
                    self.model.mode = self.args.new_mode
                self.model.eval()
                # Only set transform for image datasets (CICIDS2017_improved doesn't have transform attribute)
                if hasattr(trainloader.dataset, 'transform') and hasattr(testloader.dataset, 'transform'):
                    trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                #tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)
                #tsl, tsa = test_withfc(self.model, testloader, 0, args, session,validation=False)
                tsl, tsa = self.test_intergrate(self.model, testloader, 0,args, session,validation=False, wandb_logger=self.wandb)
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

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


    def test_intergrate(self, model, testloader, epoch,args, session,validation=True, wandb_logger=None):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5= Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        # Handle both DataParallel and single GPU/CPU models
        if hasattr(model, 'module'):
            fc_weight = model.module.fc.weight[:test_class, :]
        else:
            fc_weight = model.fc.weight[:test_class, :]
        proj_matrix=torch.mm(self.dummy_classifiers,F.normalize(torch.transpose(fc_weight,1,0),p=2,dim=-1))
        
        eta=args.eta
        
        # softmaxed_proj_matrix=F.softmax(proj_matrix,dim=1)

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.to(self.device) for _ in batch]
                
                # Handle both DataParallel and single GPU/CPU models
                if hasattr(model, 'module'):
                    emb=model.module.encode(data)
                else:
                    emb=model.encode(data)
            
                proj=torch.mm(F.normalize(emb,p=2,dim=-1),torch.transpose(self.dummy_classifiers,1,0))
                # Adjust k based on actual dimension (for datasets with fewer classes like CICIDS2017_improved)
                k = min(40, proj.size(1))
                topk, indices = torch.topk(proj, k)
                res = (torch.zeros_like(proj))
                res_logit = res.scatter(1, indices, topk)

                logits1=torch.mm(res_logit,proj_matrix)
                # Handle both DataParallel and single GPU/CPU models
                if hasattr(model, 'module'):
                    logits2 = model.module.forpass_fc(data)[:, :test_class]
                else:
                    logits2 = model.forpass_fc(data)[:, :test_class] 
                logits=eta*F.softmax(logits1,dim=1)+(1-eta)*F.softmax(logits2,dim=1)
                
                # Ensure logits has the correct number of classes
                if logits.size(1) != test_class:
                    # Pad or truncate logits to match test_class
                    if logits.size(1) < test_class:
                        # Pad with zeros
                        padding = torch.zeros(logits.size(0), test_class - logits.size(1), device=logits.device)
                        logits = torch.cat([logits, padding], dim=1)
                    else:
                        # Truncate
                        logits = logits[:, :test_class]
                
            
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                top5acc=count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            va = va.item()
            va5= va5.item()
            print('epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}'.format(epoch+1, vl, va,va5))

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
            
        return vl, va

    def set_save_path(self):
        self.args.save_path = os.path.join("checkpoint", self.args.dataset)
        ensure_path(self.args.save_path)
        return None
