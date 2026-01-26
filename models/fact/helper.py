# import new Network name here and add in model_class args
from utils import Averager, count_acc, confmatrix, get_dataset_label_names
from tqdm import tqdm
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def _get_label_names_for_loader(loader, num_classes):
    if loader is None or num_classes is None:
        return None
    label_names = get_dataset_label_names(loader.dataset)
    if label_names:
        return label_names[:num_classes]
    return None


def base_train(model, trainloader, optimizer, scheduler, epoch, args,mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    device = next(model.parameters()).device
    # DataParallelの場合はmodule、そうでない場合はmodel自体を使用
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        beta=torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.to(device) for _ in batch]
        
        logits = model(data)
        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)
        
        acc = count_acc(logits_, train_label)
        
        
        if epoch>=args.loss_iter:
            logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model_module.pre_allocate) == 1, -1e9)
            logits_masked_chosen= logits_masked * mask[train_label]
            pseudo_label = torch.argmax(logits_masked_chosen[:,args.base_class:], dim=-1) + args.base_class
            #pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
            loss2 = F.cross_entropy(logits_masked, pseudo_label)

            index = torch.randperm(data.size(0)).to(device)
            if args.dataset == 'CICIDS2017_improved':
                pre_emb1 = model_module.encode(data)
            else:
                pre_emb1 = model_module.pre_encode(data)
            mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
            mixed_logits = model_module.post_encode(mixed_data)

            newys=train_label[index]
            idx_chosen=newys!=train_label
            mixed_logits=mixed_logits[idx_chosen]

            pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], dim=-1) + args.base_class # new class label
            pseudo_label2 = torch.argmax(mixed_logits[:,:args.base_class], dim=-1)  # old class label
            loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
            novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model_module.pre_allocate) == 1, -1e9)
            loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
            total_loss = loss+args.balance*(loss2+loss3+loss4)
        else:
            total_loss = loss


        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        #loss.backward()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()
    # DataParallelの場合はmodule、そうでない場合はmodel自体を使用
    model_module = model.module if isinstance(model, nn.DataParallel) else model

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=getattr(args, 'num_workers', 4), pin_memory=getattr(args, 'pin_memory', False), shuffle=False)
    if transform is not None:
        trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.to(device) for _ in batch]
            model_module.mode = 'encoder'
            embedding = model(data)

            # CPU環境では既にCPU上にあるため、.cpu()は不要
            if device.type == 'cuda':
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
            else:
                embedding_list.append(embedding)
                label_list.append(label)
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model_module.fc.weight.data[:args.base_class] = proto_list

    return model



def test(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(device) for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            # CPU環境では既にCPU上にあるため、.cpu()は不要
            if device.type == 'cuda':
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            else:
                lgt=torch.cat([lgt,logits])
                lbs=torch.cat([lbs,test_label])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            label_names = _get_label_names_for_loader(testloader, test_class)
            cm=confmatrix(lgt,lbs,save_model_dir, label_names=label_names)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
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
    return vl, va



def test_withfc(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    # DataParallelの場合はmodule、そうでない場合はmodel自体を使用
    model_module = model.module if isinstance(model, nn.DataParallel) else model
    vl = Averager()
    va = Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(device) for _ in batch]
            logits = model_module.forpass_fc(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            # CPU環境では既にCPU上にあるため、.cpu()は不要
            if device.type == 'cuda':
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            else:
                lgt=torch.cat([lgt,logits])
                lbs=torch.cat([lbs,test_label])
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        
        lgt=lgt.view(-1,test_class)
        lbs=lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + 'confusion_matrix')
            label_names = _get_label_names_for_loader(testloader, test_class)
            cm=confmatrix(lgt,lbs,save_model_dir, label_names=label_names)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
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
            
            print(f'Session {session} Metrics (test_withfc):')
            print(f'  Accuracy: {accuracy:.4f}')
            print(f'  Precision (macro): {precision:.4f}')
            print(f'  Recall (macro): {recall:.4f}')
            print(f'  F1-score (macro): {f1:.4f}')
            
            # メトリクスをファイルに保存
            metrics_file = os.path.join(args.save_path, f'session_{session}_metrics_withfc.txt')
            with open(metrics_file, 'w') as f:
                f.write(f'Session {session} Evaluation Metrics (test_withfc)\n')
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
    return vl, va
