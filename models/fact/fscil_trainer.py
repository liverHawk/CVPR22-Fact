import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataloader.data_utils import (
    get_base_dataloader,
    get_new_dataloader,
    set_up_datasets,
)
from utils import (
    Averager,
    count_acc,
    count_acc_topk,
    ensure_path,
    log_wandb_cm_image,
    save_list_to_txt,
)

from .base import Trainer
from .helper import base_train, replace_base_fc, test
from .Network import MYNET


# --- async checkpoint saving (P1) -------------------------------------------
# torch.save is synchronous file I/O that stalls the training loop. Each save
# is snapshotted (deepcopy, so the optimizer can keep stepping while the write
# runs) and executed on a single background worker; a new save awaits the
# previous one, so two writers can never target the same path.
# flush_pending_saves() blocks until the last write is on disk and is called
# before train() returns.
_SAVE_POOL = ThreadPoolExecutor(max_workers=1)
_SAVE_PENDING = None


def save_async(obj, path):
    global _SAVE_PENDING
    if _SAVE_PENDING is not None:
        _SAVE_PENDING.result()  # previous write finished -> path is free
        _SAVE_PENDING = None
    snapshot = deepcopy(obj)
    _SAVE_PENDING = _SAVE_POOL.submit(torch.save, snapshot, path)


def flush_pending_saves():
    global _SAVE_PENDING
    if _SAVE_PENDING is not None:
        _SAVE_PENDING.result()
        _SAVE_PENDING = None


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        # Initialize wandb if enabled
        if hasattr(args, "use_wandb") and args.use_wandb:
            import wandb

            self.wandb = wandb

        self.model = MYNET(self.args, mode=self.args.base_mode)
        if self.args.num_gpu > 0 and torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
            self.model = self.model.to(args.device)

        if self.args.model_dir is not None:
            print("Loading init parameters from: %s" % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)["params"]

        else:
            print("random init params")
            if args.start_session > 0:
                print("WARING: Random init weights for new sessions!")
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.args.lr_base,
            momentum=0.9,
            nesterov=True,
            weight_decay=self.args.decay,
        )
        if self.args.schedule == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.args.step, gamma=self.args.gamma
            )
        elif self.args.schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.args.milestones, gamma=self.args.gamma
            )
        elif self.args.schedule == "Cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.epochs_base
            )

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
        no_eval = getattr(args, "no_eval", False)
        if no_eval:
            print(
                "no_eval mode: skipping in-loop evaluation, final models will be saved per session"
            )

        # init train statistics
        result_list = [args]

        # gen_mask
        masknum = 3
        mask = np.zeros((args.base_class, args.num_classes))
        for i in range(args.num_classes - args.base_class):
            picked_dummy = np.random.choice(args.base_class, masknum, replace=False)
            mask[:, i + args.base_class][picked_dummy] = 1
        mask = torch.tensor(mask).to(args.device)

        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                print("new classes for this session:\n", np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(
                        self.model, trainloader, optimizer, scheduler, epoch, args, mask
                    )
                    # test model with all seen class (skipped in no_eval mode;
                    # use test.py for the main test run)
                    if not no_eval:
                        tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # Log metrics to wandb (test_* only exist when in-loop
                    # eval ran; they are undefined under no_eval)
                    if hasattr(args, "use_wandb") and args.use_wandb:
                        payload = {
                            "train_loss": tl,
                            "train_acc": ta,
                            "lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                            "session": session,
                        }
                        if not no_eval:
                            payload["test_loss"] = tsl
                            payload["test_acc"] = tsa
                        self.wandb.log(payload, step=epoch)

                    # save better model
                    if not no_eval and (tsa * 100) >= self.trlog["max_acc"][session]:
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        self.trlog["max_acc_epoch"] = epoch
                        save_model_dir = os.path.join(
                            args.save_path, "session" + str(session) + "_max_acc.pth"
                        )
                        save_async(dict(params=self.model.state_dict()), save_model_dir)
                        save_async(
                            optimizer.state_dict(),
                            os.path.join(args.save_path, "optimizer_best.pth"),
                        )
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print("********A better model is found!!**********")
                        print("Saving model to :%s" % save_model_dir)
                    print(
                        "best epoch {}, best test acc={:.3f}".format(
                            self.trlog["max_acc_epoch"], self.trlog["max_acc"][session]
                        )
                    )

                    self.trlog["train_loss"].append(tl)
                    self.trlog["train_acc"].append(ta)
                    if not no_eval:
                        self.trlog["test_loss"].append(tsl)
                        self.trlog["test_acc"].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            "epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f"
                            % (epoch, lrc, tl, ta, tsl, tsa)
                        )
                    else:
                        lrc = scheduler.get_last_lr()[0]
                        result_list.append(
                            "epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_skipped(no_eval)"
                            % (epoch, lrc, tl, ta)
                        )
                    print(
                        "This epoch takes %d seconds" % (time.time() - start_time),
                        "\nstill need around %.2f mins to finish this session"
                        % (
                            (time.time() - start_time) * (args.epochs_base - epoch) / 60
                        ),
                    )
                    scheduler.step()

                if no_eval:
                    # no validation reference: persist the final model for the main test run
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    save_model_dir = os.path.join(
                        args.save_path, "session" + str(session) + "_max_acc.pth"
                    )
                    save_async(dict(params=self.model.state_dict()), save_model_dir)
                    print("no_eval: saved final model to :%s" % save_model_dir)
                    if args.not_data_init:
                        # no data_init eval below: measure the final model once
                        # so max_acc/metrics reflect the saved checkpoint
                        tsl, tsa = test(
                            self.model,
                            testloader,
                            args.epochs_base - 1,
                            args,
                            session,
                        )
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        self.trlog["max_acc_epoch"] = args.epochs_base - 1
                        self.trlog["test_loss"].append(tsl)
                        self.trlog["test_acc"].append(tsa)
                        print(
                            "no_eval: final-model test acc={:.3f}".format(tsa * 100)
                        )

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(
                        train_set, testloader.dataset.transform, self.model, args
                    )
                    best_model_dir = os.path.join(
                        args.save_path, "session" + str(session) + "_max_acc.pth"
                    )
                    print(
                        "Replace the fc with average embedding, and save it to :%s"
                        % best_model_dir
                    )
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    save_async(dict(params=self.model.state_dict()), best_model_dir)

                    (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    ).mode = "avg_cos"
                    # measured in both modes: this scores the exact data_init
                    # checkpoint test.py evaluates (no_eval skips only the
                    # per-epoch evals above, so metrics stay meaningful)
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog["max_acc"][session]:
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        print(
                            "The new best test acc of base session={:.3f}".format(
                                self.trlog["max_acc"][session]
                            )
                        )

                # logged after the data_init eval so it matches the checkpoint
                # that was actually saved
                result_list.append(
                    "Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n".format(
                        session,
                        self.trlog["max_acc_epoch"],
                        self.trlog["max_acc"][session],
                    )
                )

                # save dummy classifiers
                self.dummy_classifiers = deepcopy(
                    (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    ).fc.weight.detach()
                )

                self.dummy_classifiers = F.normalize(
                    self.dummy_classifiers[self.args.base_class :, :], p=2, dim=-1
                )
                self.old_classifiers = self.dummy_classifiers[: self.args.base_class, :]

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                (
                    self.model.module if hasattr(self.model, "module") else self.model
                ).mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                (
                    self.model.module if hasattr(self.model, "module") else self.model
                ).update_fc(trainloader, np.unique(train_set.targets), session)

                # tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)
                # tsl, tsa = test_withfc(self.model, testloader, 0, args, session,validation=False)
                # evaluated in both modes: there is no epoch loop here (cost
                # is one eval per session) and it populates max_acc for the
                # metrics json / tune objective
                tsl, tsa = self.test_intergrate(
                    self.model, testloader, 0, args, session, validation=True
                )

                # Log metrics to wandb
                if hasattr(args, "use_wandb") and args.use_wandb:
                    self.wandb.log(
                        {
                            "session_test_loss": tsl,
                            "session_test_acc": tsa,
                            "session": session,
                        },
                        step=session,
                    )

                    # Confusion matrix logging will be handled in test function

                # save model (always persist so test.py can run the main test run)
                self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                save_model_dir = os.path.join(
                    args.save_path, "session" + str(session) + "_max_acc.pth"
                )
                save_async(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print("Saving model to :%s" % save_model_dir)
                print("  test acc={:.3f}".format(self.trlog["max_acc"][session]))
                result_list.append(
                    "Session {}, test Acc {:.3f}\n".format(
                        session, self.trlog["max_acc"][session]
                    )
                )

        flush_pending_saves()  # all checkpoints on disk before train() returns

        result_list.append(
            "Base Session Best Epoch {}\n".format(self.trlog["max_acc_epoch"])
        )
        result_list.append(self.trlog["max_acc"])
        print(self.trlog["max_acc"])
        save_list_to_txt(os.path.join(args.save_path, "results.txt"), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print("Base Session Best epoch:", self.trlog["max_acc_epoch"])
        print("Total time used %.2f mins" % total_time)

    def test_intergrate(self, model, testloader, epoch, args, session, validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        va5 = Averager()
        lgt_list = []
        lbs_list = []

        proj_matrix = torch.mm(
            self.dummy_classifiers,
            F.normalize(
                torch.transpose(
                    (model.module if hasattr(model, "module") else model).fc.weight[
                        :test_class, :
                    ],
                    1,
                    0,
                ),
                p=2,
                dim=-1,
            ),
        )

        eta = args.eta

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.to(args.device) for _ in batch]

                emb = (model.module if hasattr(model, "module") else model).encode(data)

                proj = torch.mm(
                    F.normalize(emb, p=2, dim=-1),
                    torch.transpose(self.dummy_classifiers, 1, 0),
                )
                # top-40 over novel prototypes assumes >=40 novel classes
                # (CIFAR100/CUB200); clamp for small CIC session setups.
                topk, indices = torch.topk(proj, min(40, proj.size(1)))
                res = torch.zeros_like(proj)
                res_logit = res.scatter(1, indices, topk)

                logits1 = torch.mm(res_logit, proj_matrix)
                # reuse the emb computed above (forpass_fc re-encoded data,
                # i.e. ran the backbone twice per test batch)
                logits2 = (
                    model.module if hasattr(model, "module") else model
                ).forpass_fc_emb(emb)[:, :test_class]
                logits = eta * F.softmax(logits1, dim=1) + (1 - eta) * F.softmax(
                    logits2, dim=1
                )

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
            lgt = torch.cat(lgt_list, dim=0)
            lbs = torch.cat(lbs_list, dim=0)
            print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}, acc@5={va5:.4f}")

        # Log confusion matrix to wandb if enabled (lightweight aggregated image;
        # test_intergrate runs ~once per incremental session, so always log here)
        if hasattr(args, "use_wandb") and args.use_wandb:
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

    def set_save_path(self):
        mode = self.args.base_mode + "-" + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + "-" + "data_init"

        self.args.save_path = "%s/" % self.args.dataset
        self.args.save_path = self.args.save_path + "%s/" % self.args.project

        self.args.save_path = self.args.save_path + "%s-start_%d/" % (
            mode,
            self.args.start_session,
        )
        if self.args.schedule == "Milestone":
            mile_stone = (
                str(self.args.milestones).replace(" ", "").replace(",", "_")[1:-1]
            )
            self.args.save_path = (
                self.args.save_path
                + "Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f"
                % (
                    self.args.epochs_base,
                    self.args.lr_base,
                    mile_stone,
                    self.args.gamma,
                    self.args.batch_size_base,
                    self.args.momentum,
                )
            )
            self.args.save_path = self.args.save_path + "Bal%.2f-LossIter%d" % (
                self.args.balance,
                self.args.loss_iter,
            )
        elif self.args.schedule == "Step":
            self.args.save_path = (
                self.args.save_path
                + "Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f"
                % (
                    self.args.epochs_base,
                    self.args.lr_base,
                    self.args.step,
                    self.args.gamma,
                    self.args.batch_size_base,
                    self.args.momentum,
                )
            )
        elif self.args.schedule == "Cosine":
            self.args.save_path = self.args.save_path + "Cosine-Epo_%d-Lr_%.4f" % (
                self.args.epochs_base,
                self.args.lr_base,
            )
            self.args.save_path = self.args.save_path + "Bal%.2f-LossIter%d" % (
                self.args.balance,
                self.args.loss_iter,
            )

        if "cos" in mode:
            self.args.save_path = self.args.save_path + "-T_%.2f" % (
                self.args.temperature
            )

        if "ft" in self.args.new_mode:
            self.args.save_path = self.args.save_path + "-ftLR_%.3f-ftEpoch_%d" % (
                self.args.lr_new,
                self.args.epochs_new,
            )

        if self.args.debug:
            self.args.save_path = os.path.join("debug", self.args.save_path)

        self.args.save_path = os.path.join("checkpoint", self.args.save_path)
        ensure_path(self.args.save_path)
