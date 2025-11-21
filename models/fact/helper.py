# import new Network name here and add in model_class args
from .Network import MYNET
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np

def base_train(model, trainloader, optimizer, scheduler, epoch, args,mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):

        beta=torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.cuda() for _ in batch]
        
        embeddings=model.module.encode(data)

        logits = model(data)
        logits_ = logits[:, :args.base_class]
        loss = F.cross_entropy(logits_, train_label)
        
        acc = count_acc(logits_, train_label)
        
        
        if epoch>=args.loss_iter:
            logits_masked = logits.masked_fill(F.one_hot(train_label, num_classes=model.module.pre_allocate) == 1, -1e9)
            logits_masked_chosen= logits_masked * mask[train_label]
            pseudo_label = torch.argmax(logits_masked_chosen[:,args.base_class:], dim=-1) + args.base_class
            #pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
            loss2 = F.cross_entropy(logits_masked, pseudo_label)

            index = torch.randperm(data.size(0)).cuda()
            pre_emb1=model.module.pre_encode(data)
            mixed_data=beta*pre_emb1+(1-beta)*pre_emb1[index]
            mixed_logits=model.module.post_encode(mixed_data)

            newys=train_label[index]
            idx_chosen=newys!=train_label
            mixed_logits=mixed_logits[idx_chosen]

            pseudo_label1 = torch.argmax(mixed_logits[:,args.base_class:], dim=-1) + args.base_class # new class label
            pseudo_label2 = torch.argmax(mixed_logits[:,:args.base_class], dim=-1)  # old class label
            loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
            novel_logits_masked = mixed_logits.masked_fill(F.one_hot(pseudo_label1, num_classes=model.module.pre_allocate) == 1, -1e9)
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

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    # Only set transform for image datasets (CICIDS2017_improved doesn't have transform attribute)
    if hasattr(trainloader.dataset, 'transform') and transform is not None:
        trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = 'encoder'
            embedding = model(data)
            # GPU上で保持してから最後にまとめてCPU転送（高速化）
            embedding_list.append(embedding)
            label_list.append(label)
    # GPU上でcatしてから一度だけCPU転送（メモリ効率と速度が向上）
    embedding_list = torch.cat(embedding_list, dim=0).cpu()
    label_list = torch.cat(label_list, dim=0).cpu()

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    model.module.fc.weight.data[:args.base_class] = proto_list

    return model



def test(model, testloader, epoch,args, session,validation=True, wandb_logger=None, 
         enable_unknown_detection=False, distance_type='cosine', distance_threshold=None):
    """
    テスト関数（未知クラス検出オプション付き）
    
    Args:
        enable_unknown_detection: 未知クラス検出を有効にするか
        distance_type: 距離の種類
        distance_threshold: 距離の閾値（Noneの場合は自動計算）
    """
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    # リストに蓄積してから最後にまとめてcat（高速化）
    lgt_list = []
    lbs_list = []
    
    # 未知クラス検出用の統計
    unknown_stats = {
        'total_unknown_detected': 0,
        'total_samples': 0,
        'unknown_distances': [],
    } if enable_unknown_detection else None
    
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            # 未知クラス検出が有効な場合
            if enable_unknown_detection:
                known_class_indices = list(range(test_class))
                result = model(data, enable_unknown_detection=True, 
                              known_class_indices=known_class_indices,
                              distance_threshold=distance_threshold,
                              distance_type=distance_type)
                logits, is_unknown, distances, nearest_class = result
                
                # 未知クラス検出の統計を更新
                unknown_stats['total_samples'] += data.size(0)
                unknown_stats['total_unknown_detected'] += is_unknown.sum().item()
                unknown_stats['unknown_distances'].extend(distances[is_unknown].cpu().tolist())
            else:
                logits = model(data)
            
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt_list.append(logits.cpu())
            lbs_list.append(test_label.cpu())
        vl = vl.item()
        va = va.item()
        
        # 未知クラス検出の結果を表示
        if enable_unknown_detection and unknown_stats['total_samples'] > 0:
            unknown_rate = 100 * unknown_stats['total_unknown_detected'] / unknown_stats['total_samples']
            avg_distance = sum(unknown_stats['unknown_distances']) / len(unknown_stats['unknown_distances']) if unknown_stats['unknown_distances'] else 0.0
            print('epo {}, test, loss={:.4f} acc={:.4f}, unknown_detected={:.2f}% (avg_dist={:.4f})'.format(
                epoch, vl, va, unknown_rate, avg_distance))
        else:
            print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
        
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



def test_withfc(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    # リストに蓄積してから最後にまとめてcat（高速化）
    lgt_list = []
    lbs_list = []
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model.module.forpass_fc(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt_list.append(logits.cpu())
            lbs_list.append(test_label.cpu())
        vl = vl.item()
        va = va.item()
        print('epo {}, test, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))
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
    return vl, va
