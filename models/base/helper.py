# import new Network name here and add in model_class args
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import (
    Averager,
    confmatrix,
    count_acc,
    count_acc_topk,
    log_wandb_cm_image,
    should_log_wandb_cm,
    wandb_step,
)


def base_train(model, trainloader, optimizer, scheduler, epoch, args):
    tl = Averager()
    ta = Averager()
    model = model.train()
    # standard classification for pretrain
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.to(args.device) for _ in batch]

        logits = model(data)
        logits = logits[:, : args.base_class]
        loss = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)

        total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            f"Session 0, epo {epoch}, lrc={lrc:.4f},total loss={total_loss.item():.4f} acc={acc:.4f}"
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
        dataset=trainset,
        batch_size=512,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )
    trainloader.dataset.transform = transform
    embedding_list = []
    label_list = []
    # data_list=[]
    with torch.no_grad():
        for i, batch in enumerate(trainloader):
            data, label = [_.to(args.device) for _ in batch]
            (model.module if hasattr(model, "module") else model).mode = "encoder"
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

    (model.module if hasattr(model, "module") else model).fc.weight.data[
        : args.base_class
    ] = proto_list

    return model


def test(model, testloader, epoch, args, session, validation=True, log_wandb_cm=False):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt_list = []
    lbs_list = []
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(args.device) for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            top5acc = count_acc_topk(logits, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)

            lgt_list.append(logits.cpu())
            lbs_list.append(test_label.cpu())
        vl = vl.item()
        va = va.item()
        va5 = va5.item()
        print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}, acc@5={va5:.4f}")

        lgt = torch.cat(lgt_list, dim=0).view(-1, test_class)
        lbs = torch.cat(lbs_list, dim=0).view(-1)
        if validation is not True:
            save_model_dir = os.path.join(
                args.save_path, "session" + str(session) + "confusion_matrix"
            )
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[: args.base_class])
            unseenac = np.mean(perclassacc[args.base_class :])
            print("Seen Acc:", seenac, "Unseen ACC:", unseenac)

    # Log confusion matrix to wandb if enabled (lightweight aggregated image).
    # Only for the session's final evaluation, never mid-training epoch checks.
    if (
        log_wandb_cm
        and hasattr(args, "use_wandb")
        and args.use_wandb
        and should_log_wandb_cm(args, session, epoch)
    ):
        try:
            log_wandb_cm_image(
                lbs.numpy().astype(int),
                torch.argmax(lgt, dim=1).numpy().astype(int),
                test_class,
                step=wandb_step(args, session, epoch),
                key=f"confusion_matrix_session_{session}",
            )
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to wandb: {e}")

    return vl, va
