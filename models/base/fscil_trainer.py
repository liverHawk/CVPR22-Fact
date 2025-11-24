from .base import Trainer
# import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import base_train, test, replace_base_fc
from utils import ensure_path, save_list_to_txt, torch, time, np, os
from dataloader.data_utils import set_up_datasets, get_base_dataloader, get_new_dataloader
from .Network import MYNET


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        if args.device != "cpu":
            self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.to(self.args.device)
        self.wandb.watch(self.model)

        if self.args.model_dir is not None:
            print("Loading init parameters from: %s" % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir, weights_only=False)[
                "params"
            ]
            # self.best_model_dict = torch.load(self.args.model_dir)['state_dict']
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
                        self.model, trainloader, optimizer, scheduler, epoch, args
                    )
                    args.comet.log_metrics(
                        dic={
                            "train/base": {
                                "loss": tl,
                                "acc": ta,
                            }
                        },
                        step=epoch
                    )
                    # test model with all seen class
                    tsl, tsa, acc_dict = test(
                        self.model,
                        testloader,
                        epoch,
                        args,
                        session,
                        wandb_logger=self.wandb,
                        name="train"
                    )
                    args.comet.log_metrics(
                        dic={
                            "test/base": {
                                "loss": tsl,
                                "acc": tsa,
                                "seenac": acc_dict["seenac"],
                                "unseenac": acc_dict["unseenac"],
                            }
                        },
                        step=epoch
                    )

                    # save better model
                    if (tsa * 100) >= self.trlog["max_acc"][session]:
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        self.trlog["max_acc_epoch"] = epoch
                        save_model_dir = os.path.join(
                            args.save_path, "session" + str(session) + "_max_acc.pth"
                        )
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(
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
                    self.trlog["test_loss"].append(tsl)
                    self.trlog["test_acc"].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        "epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f"
                        % (epoch, lrc, tl, ta, tsl, tsa)
                    )
                    self.global_step += 1
                    self.wandb.log_metrics(
                        {
                            "session": session,
                            "epoch": epoch,
                            "lr": lrc,
                            "train/loss": tl,
                            "train/acc": ta,
                            "test/loss": tsl,
                            "test/acc": tsa,
                        },
                        step=self.global_step,
                    )
                    print(
                        "This epoch takes %d seconds" % (time.time() - start_time),
                        "\nstill need around %.2f mins to finish this session"
                        % (
                            (time.time() - start_time) * (args.epochs_base - epoch) / 60
                        ),
                    )
                    scheduler.step()

                result_list.append(
                    "Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n".format(
                        session,
                        self.trlog["max_acc_epoch"],
                        self.trlog["max_acc"][session],
                    )
                )

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    # For CICIDS2017_improved (tabular data), transform is not needed
                    transform = (
                        getattr(testloader.dataset, "transform", None)
                        if args.dataset != "CICIDS2017_improved"
                        else None
                    )
                    self.model = replace_base_fc(train_set, transform, self.model, args)
                    best_model_dir = os.path.join(
                        args.save_path, "session" + str(session) + "_max_acc.pth"
                    )
                    print(
                        "Replace the fc with average embedding, and save it to :%s"
                        % best_model_dir
                    )
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = "avg_cos"
                    tsl, tsa, acc_dict = test(
                        self.model,
                        testloader,
                        0,
                        args,
                        session,
                        wandb_logger=self.wandb,
                        name="session"
                    )
                    args.comet.log_metrics(
                        dic={
                            "test": {
                                "loss": tsl,
                                "acc": tsa,
                                "seenac": acc_dict["seenac"],
                                "unseenac": acc_dict["unseenac"],
                            }
                        },
                        step=0
                    )
                    if (tsa * 100) >= self.trlog["max_acc"][session]:
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        print(
                            "The new best test acc of base session={:.3f}".format(
                                self.trlog["max_acc"][session]
                            )
                        )

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                # Only set transform for image datasets (CICIDS2017_improved doesn't have transform attribute)
                if hasattr(trainloader.dataset, "transform") and hasattr(
                    testloader.dataset, "transform"
                ):
                    trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(
                    trainloader, np.unique(train_set.targets), session
                )

                tsl, tsa, acc_dict = test(
                    self.model,
                    testloader,
                    0,
                    args,
                    session,
                    validation=False,
                    wandb_logger=self.wandb,
                    name="session"
                )
                args.comet.log_metrics(
                    dic={
                        "test": {
                            "loss": tsl,
                            "acc": tsa,
                            "seenac": acc_dict["seenac"],
                            "unseenac": acc_dict["unseenac"],
                        }
                    },
                    step=session
                )

                # save model
                self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                save_model_dir = os.path.join(
                    args.save_path, "session" + str(session) + "_max_acc.pth"
                )
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print("Saving model to :%s" % save_model_dir)
                print("  test acc={:.3f}".format(self.trlog["max_acc"][session]))

                result_list.append(
                    "Session {}, test Acc {:.3f}\n".format(
                        session, self.trlog["max_acc"][session]
                    )
                )
                self.global_step += 1
                self.wandb.log_metrics(
                    {
                        "session": session,
                        "epoch": 0,
                        "test/loss": tsl,
                        "test/acc": tsa,
                    },
                    step=self.global_step,
                )

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
        summary_payload = {
            f"session_{idx}_best_acc": acc
            for idx, acc in enumerate(self.trlog["max_acc"])
        }
        summary_payload["base_best_epoch"] = self.trlog["max_acc_epoch"]
        summary_payload["total_time_min"] = total_time
        self.wandb.set_summary(**summary_payload)
        self.finalize()

    def set_save_path(self):
        self.args.save_path = os.path.join("checkpoint", self.args.dataset)
        ensure_path(self.args.save_path)
        return None
