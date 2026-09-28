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
    eval_stats,
    class_names,
    log_wandb_cm_image,
    should_log_wandb_cm,
)


# refresh the tqdm loss/acc text every N steps: each refresh is a GPU sync
TQDM_DESC_EVERY = 50


class _GpuMean:
    """Mean of per-batch scalars kept on device, synced once in item().

    Replaces Averager in the step loop, where a .item() per value stalled the
    GPU queue every step. float64 so the mean matches Averager to ~1e-15.
    """

    def __init__(self):
        self.sum = None
        self.n = 0

    def add(self, x):
        x = x.detach().double()
        self.sum = x if self.sum is None else self.sum + x
        self.n += 1

    def item(self):
        return self.sum.item() / self.n


def base_train(model, trainloader, optimizer, scheduler, epoch, args, mask):
    tl = _GpuMean()
    ta = _GpuMean()
    mix_conf = _GpuMean()  # mean max-prob of mixed z over KNOWN classes (want: low)
    mix_ent = _GpuMean()  # mean entropy of mixed z over KNOWN classes (want: high)
    # per-term losses (all but L1 only filled once epoch >= loss_iter)
    loss_avg = {
        k: _GpuMean()
        for k in ("L1", "L2", "L3", "L4", "Lv", "Lf")
    }
    model = model.train()
    # single-GPU DataParallel only adds scatter/gather per step; call the
    # wrapped module directly (state_dict keys keep their "module." prefix)
    net = (
        model.module
        if isinstance(model, torch.nn.DataParallel) and len(model.device_ids) == 1
        else model
    )
    tqdm_gen = tqdm(trainloader)

    for i, batch in enumerate(tqdm_gen, 1):
        beta = torch.distributions.beta.Beta(args.alpha, args.alpha).sample([]).item()
        data, train_label = [_.to(args.device) for _ in batch]

        logits = net(data)
        logits_ = logits[:, : args.base_class]
        loss = F.cross_entropy(logits_, train_label)

        acc = (torch.argmax(logits_, dim=1) == train_label).float().mean()
        loss_avg["L1"].add(loss)

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

            if mixed_logits.size(0) > 0:
                # verification: mixed z should NOT confidently match known classes
                known_probs = F.softmax(
                    mixed_logits[:, : args.base_class], dim=-1
                )
                mix_conf.add(known_probs.max(dim=-1).values.mean())
                mix_ent.add(
                    -(known_probs * known_probs.clamp_min(1e-12).log()).sum(
                        dim=-1
                    ).mean()
                )

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
            # L = Lv + Lf  (Lv: real data, Lf: mixed instances)
            loss_v = loss + args.balance * loss2
            loss_f = loss3 + args.balance * loss4
            total_loss = loss_v + loss_f
            for k, v in (
                ("L2", loss2),
                ("L3", loss3),
                ("L4", loss4),
                ("Lv", loss_v),
                ("Lf", loss_f),
            ):
                loss_avg[k].add(v)
        else:
            total_loss = loss

        if i % TQDM_DESC_EVERY == 1:
            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                f"Session 0, epo {epoch}, lrc={lrc:.4f},total loss={total_loss.item():.4f} acc={acc.item():.4f}"
            )
        tl.add(total_loss)
        ta.add(acc)

        optimizer.zero_grad()
        # loss.backward()
        total_loss.backward()
        optimizer.step()
    tl = tl.item()
    ta = ta.item()
    if getattr(args, "use_wandb", False):
        try:
            import wandb

            wandb.log(
                {
                    **{f"train/{k}": a.item() for k, a in loss_avg.items() if a.n > 0},
                    "train/L": tl,
                    "epoch": epoch,
                },
            )
        except Exception as e:
            print(f"Warning: Could not log loss terms to wandb: {e}")
    if mix_conf.n > 0:
        print(
            f"epo {epoch}, mixup vs known classes: "
            f"mean-max-prob={mix_conf.item():.4f} (lower is better), "
            f"entropy={mix_ent.item():.4f} (higher is better, "
            f"uniform={np.log(args.base_class):.4f})"
        )
        if getattr(args, "use_wandb", False):
            try:
                import wandb

                wandb.log(
                    {
                        "mixup_known_maxprob": mix_conf.item(),
                        "mixup_known_entropy": mix_ent.item(),
                        "epoch": epoch,
                    },
                )
            except Exception as e:
                print(f"Warning: Could not log mixup stats to wandb: {e}")
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


def test(
    model, testloader, epoch, args, session, validation=True, return_stats=False,
    log_cm=True,
):
    # log_cm: per-epoch CM (x-axis epoch). Session-final evals pass False; the
    # trainer logs their CM as session/confusion_matrix (x-axis session).
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt_list = []
    lbs_list = []
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(args.device) for _ in batch]
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
        print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}")

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

    # Log confusion matrix to wandb if enabled (lightweight aggregated image)
    if (
        log_cm
        and hasattr(args, "use_wandb")
        and args.use_wandb
        and should_log_wandb_cm(args, session, epoch)
    ):
        try:
            log_wandb_cm_image(
                lbs.numpy().astype(int),
                torch.argmax(lgt, dim=1).numpy().astype(int),
                test_class,
                key="test_confusion_matrix",
                names=class_names(args),
                extra={"epoch": epoch},
            )
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to wandb: {e}")

    if return_stats:
        return vl, va, eval_stats(lgt, lbs, test_class, args.base_class)
    return vl, va


def test_withfc(model, testloader, epoch, args, session, validation=True):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager()
    va = Averager()
    lgt_list = []
    lbs_list = []
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
            lgt_list.append(logits.cpu())
            lbs_list.append(test_label.cpu())
        vl = vl.item()
        va = va.item()
        print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}")

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
                names=class_names(args),
            )
        except Exception as e:
            print(f"Warning: Could not log confusion matrix to wandb: {e}")

    return vl, va
