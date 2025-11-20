# import new Network name here and add in model_class args
from utils import Averager, count_acc, count_acc_topk, confmatrix
from tqdm import tqdm
import torch.nn.functional as F
import torch
import numpy as np
import os


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        device = next(model.parameters()).device
        data, train_label = [_.to(device) for _ in batch]

        print(f"base_train: イテレーション {i}, data.shape={data.shape}, train_label.shape={train_label.shape}")
        print(f"base_train: train_label={train_label}")

        logits = model(data)
        logits = logits[:, :args.base_class]
        print(f"base_train: logits.shape={logits.shape}, logits={logits}")

        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        print(f"base_train: loss={loss.item():.4f}, acc={acc:.4f}")

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        print(f"base_train: lrc (学習率)={lrc:.4f}")
        tqdm_gen.set_description(
            'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch+1, lrc, total_loss.item(), acc))
        tl.add(total_loss.item())
        ta.add(acc)

        print(f"base_train: tl (累積平均損失)={tl.item():.4f}, ta (累積平均精度)={ta.item():.4f}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    print(f"base_train: 最終値 - tl={tl:.4f}, ta={ta:.4f}")
    print(f"base_train: 返り値 - tl={tl}, ta={ta}")
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128,
                                              num_workers=8, pin_memory=True, shuffle=False)
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            device = next(model.parameters()).device
            data, label = [_.to(device) for _ in batch]
            if hasattr(model, 'module'):
                model.module.mode = 'encoder'
            else:
                model.mode = 'encoder'
            embedding = model(data)

            embedding_list.append(embedding.cpu())
            label_list.append(label.cpu())
    embedding_list = torch.cat(embedding_list, dim=0)
    label_list = torch.cat(label_list, dim=0)

    proto_list = []

    for class_index in range(args.base_class):
        data_index = (label_list == class_index).nonzero()
        embedding_this = embedding_list[data_index.squeeze(-1)]
        embedding_this = embedding_this.mean(0)
        proto_list.append(embedding_this)

    proto_list = torch.stack(proto_list, dim=0)

    if hasattr(model, 'module'):
        model.module.fc.weight.data[:args.base_class] = proto_list
    else:
        model.fc.weight.data[:args.base_class] = proto_list

    return model




def test(model, testloader, epoch,args, session,validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5= Averager()
    lgt=torch.tensor([])
    lbs=torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            device = next(model.parameters()).device
            data, test_label = [_.to(device) for _ in batch]
            logits = model(data)
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
            cm=confmatrix(lgt,lbs,save_model_dir)
            perclassacc=cm.diagonal()
            seenac=np.mean(perclassacc[:args.base_class])
            unseenac=np.mean(perclassacc[args.base_class:])
            print('Seen Acc:',seenac, 'Unseen ACC:', unseenac)
    return vl, va
