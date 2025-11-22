# import new Network name here and add in model_class args
# from .Network import MYNET
import torch.nn.functional as F
from tqdm import tqdm

from utils import Averager, confmatrix, count_acc, count_acc_topk, np, os, torch


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]

        logits = model(data)
        logits = logits[:, : args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            "Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}".format(
                epoch + 1, lrc, total_loss.item(), acc
            )
        )
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    return tl, ta


def replace_base_fc(trainset, transform, model, args):
    # replace fc.weight with the embedding average of train data
    model = model.eval()

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=128, num_workers=8, pin_memory=True, shuffle=False
    )
    # Only set transform for image datasets (CICIDS2017_improved doesn't have transform attribute)
    if hasattr(trainloader.dataset, "transform") and transform is not None:
        trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.cuda() for _ in batch]
            model.module.mode = "encoder"
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

    model.module.fc.weight.data[: args.base_class] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True, wandb_logger=None):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print(
            "epo {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}".format(
                epoch + 1, vl, va, va5
            )
        )

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(
                args.save_path, "session" + str(session) + "confusion_matrix"
            )
            # ラベル名を取得（CICIDS2017_improvedの場合）
            label_names = None
            if args.dataset == "CICIDS2017_improved" and hasattr(
                testloader.dataset, "label_encoder"
            ):
                label_names = list(testloader.dataset.label_encoder.classes_)
            cm = confmatrix(lgt, lbs, save_model_dir, label_names=label_names)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[: args.base_class])
            unseenac = np.mean(perclassacc[args.base_class :])
            print("Seen Acc:", seenac, "Unseen ACC:", unseenac)
            # Classification reportを保存
            from utils import save_classification_report

            save_classification_report(lgt, lbs, save_model_dir)
            if wandb_logger is not None:
                wandb_logger.log_image(
                    f"session_{session}_confusion_matrix", save_model_dir + ".png"
                )
    return vl, va
