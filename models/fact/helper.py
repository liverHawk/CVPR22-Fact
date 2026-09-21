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
    log_wandb_cm_image,
    should_log_wandb_cm,
)


def base_train(model, trainloader, optimizer, scheduler, epoch, args, mask):
    tl = Averager()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.to(args.device) for _ in batch]

        embeddings = (model.module if hasattr(model, "module") else model).encode(data)

        logits = model(data)
        logits_ = logits[:, : args.base_class]
        loss = F.cross_entropy(logits_, train_label)

        acc = count_acc(logits_, train_label)

        if epoch >= args.loss_iter:
            logits_masked = logits.masked_fill(
                F.one_hot(
                    train_label,
                    num_classes=(
                        model.module if hasattr(model, "module") else model
                    ).pre_allocate,
                )
                == 1,
                -1e9,
            )
            logits_masked_chosen = logits_masked * mask[train_label]
            pseudo_label = (
                torch.argmax(logits_masked_chosen[:, args.base_class :], dim=-1)
                + args.base_class
            )
            # pseudo_label = torch.argmax(logits_masked[:,args.base_class:], dim=-1) + args.base_class
            loss2 = F.cross_entropy(logits_masked, pseudo_label)

            index = torch.randperm(data.size(0)).to(args.device)
            pre_emb1 = (model.module if hasattr(model, "module") else model).pre_encode(
                data
            )
            mixed_data = beta * pre_emb1 + (1 - beta) * pre_emb1[index]
            mixed_logits = (
                model.module if hasattr(model, "module") else model
            ).post_encode(mixed_data)

            newys = train_label[index]
            idx_chosen = newys != train_label
            mixed_logits = mixed_logits[idx_chosen]

            pseudo_label1 = (
                torch.argmax(mixed_logits[:, args.base_class :], dim=-1)
                + args.base_class
            )  # new class label
            pseudo_label2 = torch.argmax(
                mixed_logits[:, : args.base_class], dim=-1
            )  # old class label
            loss3 = F.cross_entropy(mixed_logits, pseudo_label1)
            novel_logits_masked = mixed_logits.masked_fill(
                F.one_hot(
                    pseudo_label1,
                    num_classes=(
                        model.module if hasattr(model, "module") else model
                    ).pre_allocate,
                )
                == 1,
                -1e9,
            )
            loss4 = F.cross_entropy(novel_logits_masked, pseudo_label2)
            total_loss = loss + args.balance * (loss2 + loss3 + loss4)
        else:
            total_loss = loss

        lrc = scheduler.get_last_lr()[0]
        tqdm_gen.set_description(
            f"Session 0, epo {epoch}, lrc={lrc:.4f},total loss={total_loss.item():.4f} acc={acc:.4f}"
        )
        tl.add(total_loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
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


def test(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(args.device) for _ in batch]
            logits = model(data)
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}")

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(
                args.save_path, "session" + str(session) + "confusion_matrix"
            )
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[: args.base_class])
            unseenac = np.mean(perclassacc[args.base_class :])
            print("Seen Acc:", seenac, "Unseen ACC:", unseenac)

    # Log confusion matrix to wandb if enabled (lightweight aggregated image)
    if (
        hasattr(args, "use_wandb")
        and args.use_wandb
        and should_log_wandb_cm(args, session, epoch)
    ):
        try:
            log_wandb_cm_image(
                lbs.numpy().astype(int),
                torch.argmax(lgt, dim=1).numpy().astype(int),
                test_class,
                step=epoch if session == 0 else session,
            )
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to wandb: {e}")

    return vl, va


def test_withfc(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(args.device) for _ in batch]
            logits = (model.module if hasattr(model, "module") else model).forpass_fc(
                data
            )
            logits = logits[:, :test_class]
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            vl.add(loss.item())
            va.add(acc)
            lgt = torch.cat([lgt, logits.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])
        vl = vl.item()
        va = va.item()
        print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}")

        lgt = lgt.view(-1, test_class)
        lbs = lbs.view(-1)
        if validation is not True:
            save_model_dir = os.path.join(
                args.save_path, "session" + str(session) + "confusion_matrix"
            )
            cm = confmatrix(lgt, lbs, save_model_dir)
            perclassacc = cm.diagonal()
            seenac = np.mean(perclassacc[: args.base_class])
            unseenac = np.mean(perclassacc[args.base_class :])
            print("Seen Acc:", seenac, "Unseen ACC:", unseenac)

    # Log confusion matrix to wandb if enabled (lightweight aggregated image)
    if (
        hasattr(args, "use_wandb")
        and args.use_wandb
        and should_log_wandb_cm(args, session, epoch)
    ):
        try:
            log_wandb_cm_image(
                lbs.numpy().astype(int),
                torch.argmax(lgt, dim=1).numpy().astype(int),
                test_class,
                step=epoch if session == 0 else session,
            )
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to wandb: {e}")

    return vl, va
