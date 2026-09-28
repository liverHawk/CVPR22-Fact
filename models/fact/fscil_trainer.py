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
    eval_stats,
    class_names,
    cm_counts,
    cm_figure,
    SPLIT_METRICS,
    stratified_indices,
    tsne_embed,
    tsne_figure,
    save_list_to_txt,
)

from .base import Trainer
from .helper import _GpuMean, base_train, replace_base_fc, test
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

        self.session_cm = {}  # session -> (y_true, y_pred, n_class)
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
                        tsl, tsa, tss = test(
                            self.model, testloader, epoch, args, session, return_stats=True
                        )

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
                        self.wandb.log(payload)

                    # save better model
                    if not no_eval and (tsa * 100) >= self.trlog["max_acc"][session]:
                        self._record_session(session, tsa, tss)
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
                    if not no_eval:
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
                        tsl, tsa, tss = test(
                            self.model,
                            testloader,
                            args.epochs_base - 1,
                            args,
                            session,
                            return_stats=True,
                            log_cm=False,
                        )
                        self._record_session(session, tsa, tss)
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
                    tsl, tsa, tss = test(
                        self.model, testloader, 0, args, session, return_stats=True,
                        log_cm=False,
                    )
                    if (tsa * 100) >= self.trlog["max_acc"][session]:
                        self._record_session(session, tsa, tss)
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
                tsl, tsa, tss = self.test_intergrate(
                    self.model, testloader, 0, args, session, validation=True,
                    return_stats=True,
                )

                # save model (always persist so test.py can run the main test run)
                self._record_session(session, tsa, tss)
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

            tsne_log = self._session_tsne(session, testloader) if args.tsne else {}
            cm_table = self._session_cm_counts(session)

            # one point per session (x-axis "session", see train.py define_metric)
            if hasattr(args, "use_wandb") and args.use_wandb:
                cm_img = self._session_cm_image(session)
                self.wandb.log(
                    {
                        **tsne_log,
                        **(
                            {"session/confusion_table": cm_table}
                            if cm_table is not None
                            else {}
                        ),
                        **(
                            {"session/confusion_matrix": cm_img}
                            if cm_img is not None
                            else {}
                        ),
                        "session": session,
                        "session/acc": self.trlog["max_acc"][session],
                        "session/f1": self.trlog["max_f1"][session],
                        **{
                            f"session/{k}": self.trlog[k][session]
                            for k in SPLIT_METRICS
                            if self.trlog[k][session] is not None
                        },
                    }
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

    def _record_session(self, session, acc, stats):
        """Store one session's best-checkpoint metrics (acc is a 0-1 fraction)."""
        self.trlog["max_acc"][session] = float("%.3f" % (acc * 100))
        self.trlog["max_f1"][session] = stats["f1"]
        for k in SPLIT_METRICS:
            self.trlog[k][session] = stats[k]
        # predictions of the recorded checkpoint, for session/confusion_matrix
        self.session_cm[session] = (stats["y_true"], stats["y_pred"], stats["n_class"])

    def _session_tsne(self, session, testloader):
        """t-SNE of the current (= recorded) model's test features, with FACT
        mixup points and classifier confidence.

        tsne_layer: pre (mixup point) | emb (final embedding) | both | all
        (every encoder block, pre1..preN + post1..postM, however many
        mlp_pre_layers / mlp_post_layers there are; encoders without
        layer_outputs fall back to pre + emb).

        Mixup points are built like base_train builds them: pre features of
        two test samples with different labels, mixed with lam ~
        Beta(alpha, alpha), then pushed through post. They exist from the
        mixup point on (preN / pre, post*, emb) and share each of those
        layers' t-SNE fit with the real points. Confidence is the max class
        probability of that session's classifier (same as its evaluation).

        Saves <save_path>/tsne/session<N>_<layer>.png plus session<N>.json and
        returns the wandb payload for the session log (images + mean
        confidences; empty without wandb or on failure; t-SNE never stops
        training).
        """
        args = self.args
        try:
            import json

            import matplotlib.pyplot as plt

            ds = testloader.dataset
            idx = stratified_indices(ds.targets, args.tsne_samples, args.seed + session)
            xs, ys = zip(*(ds[int(i)] for i in idx))
            x = torch.stack([torch.as_tensor(v) for v in xs]).to(args.device)
            y = np.asarray(ys, dtype=int)
            net = self.model.module if hasattr(self.model, "module") else self.model
            net.eval()
            mode = args.tsne_layer
            if mode == "all" and not hasattr(net.encoder, "layer_outputs"):
                mode = "both"
            can_mix = hasattr(net.encoder, "post_outputs")

            # mixup pairs over the sampled points (different labels only)
            rng = np.random.default_rng(args.seed + 1000 + session)
            n_mix = min(args.tsne_mixup, len(y) * 4)
            a = rng.integers(0, len(y), n_mix * 4)
            b = rng.integers(0, len(y), n_mix * 4)
            keep = y[a] != y[b]
            a, b = a[keep][:n_mix], b[keep][:n_mix]
            lam = rng.beta(args.alpha, args.alpha, len(a)).astype(np.float32)
            if not can_mix or len(a) == 0:
                a = b = np.zeros(0, dtype=int)
                lam = np.zeros(0, dtype=np.float32)

            with torch.no_grad():
                real, h_list, emb_list = {}, [], []
                for start in range(0, len(x), 1024):
                    xb = x[start : start + 1024]
                    h_list.append(net.pre_encode(xb))
                    emb_list.append(net.encode(xb))
                    if mode == "all":
                        outs = net.encoder.layer_outputs(xb)
                    else:
                        outs = []
                        if mode in ("pre", "both"):
                            outs.append(("pre", h_list[-1]))
                        if mode in ("emb", "both"):
                            outs.append(("emb", emb_list[-1]))
                    for name, f in outs:
                        real.setdefault(name, []).append(f.flatten(1).cpu())
                real = {k: torch.cat(v) for k, v in real.items()}
                h = torch.cat(h_list)
                emb = torch.cat(emb_list)
                probs = self._session_probs(net, emb, session)

                mix = {}
                if len(a):
                    lam_t = torch.as_tensor(lam, device=h.device)[:, None]
                    h_mix = lam_t * h[a] + (1 - lam_t) * h[b]
                    post = net.encoder.post_outputs(h_mix)
                    mix_emb = post[-1][1]
                    mix_probs = self._session_probs(net, mix_emb, session)
                    last_pre = [k for k in real if k.startswith("pre")]
                    if last_pre:
                        mix[last_pre[-1]] = h_mix.cpu()
                    for name, f in post:
                        if name in real:
                            mix[name] = f.cpu()
                    if "emb" in real:
                        mix["emb"] = mix_emb.cpu()

            conf, pred = probs.max(dim=1)
            out_dir = getattr(args, "tsne_dir", None) or os.path.join(args.save_path, "tsne")
            ensure_path(out_dir)
            names = class_names(args)
            rnd = lambda t: [round(float(v), 4) for v in np.asarray(t).ravel()]
            record = {
                "session": session,
                "names": names,
                "base_class": args.base_class,
                "labels": y.tolist(),
                "conf": rnd(conf.cpu()),
                "pred": pred.cpu().tolist(),
                "metrics": {
                    k: self.trlog[t][session]
                    for k, t in (
                        ("acc", "max_acc"),
                        ("f1", "max_f1"),
                        *((m, m) for m in SPLIT_METRICS),
                    )
                },
                "layers": [],
            }
            if session in self.session_cm:
                yt, yp, k = self.session_cm[session]
                record["cm"] = cm_counts(yt, yp, k).tolist()  # full test set
            payload = {"session/tsne_conf_test": float(conf.mean())}
            if len(a):
                mc, mp = mix_probs.max(dim=1)
                record["mix"] = {
                    "a": a.tolist(),
                    "b": b.tolist(),
                    "lam": rnd(lam),
                    "conf": rnd(mc.cpu()),
                    "pred": mp.cpu().tolist(),
                }
                payload["session/tsne_conf_mixup"] = float(mc.mean())

            for name, f in real.items():  # insertion order = network order
                f = f.numpy()
                m = mix[name].numpy() if name in mix else None
                both = f if m is None else np.concatenate([f, m.reshape(len(m), -1)])
                xy = tsne_embed(both, seed=args.seed)
                if xy is None:
                    continue
                span = np.ptp(xy, axis=0)
                span[span == 0] = 1
                norm = (xy - xy.min(axis=0)) / span  # [0, 1] per axis, for viewers
                layer = {"name": name, "dim": int(f.shape[1]), "xy": rnd(norm[: len(f)])}
                if m is not None:
                    layer["mix_xy"] = rnd(norm[len(f) :])
                record["layers"].append(layer)
                fig = tsne_figure(
                    xy[: len(f)],
                    y,
                    names=names,
                    base_class=args.base_class,
                    title=f"t-SNE ({name}), session {session}",
                    mix_xy=None if m is None else xy[len(f) :],
                )
                fig.savefig(os.path.join(out_dir, f"session{session}_{name}.png"))
                if hasattr(args, "use_wandb") and args.use_wandb:
                    payload[f"session/tsne_{name}"] = self.wandb.Image(fig)
                plt.close(fig)
            with open(os.path.join(out_dir, f"session{session}.json"), "w") as fp:
                json.dump(record, fp, separators=(",", ":"))
            return payload
        except Exception as e:
            print(f"Warning: Could not build t-SNE for session {session}: {e}")
            return {}

    def _session_cm_counts(self, session):
        """Raw confusion-matrix counts of the recorded checkpoint.

        Always writes <save_path>/confusion/session<N>.csv (rows = true class,
        columns = predicted, plus a total column); returns the same as a
        wandb.Table when wandb is on, else None. Never stops training.
        """
        if session not in self.session_cm:
            return None
        try:
            import csv

            args = self.args
            y_true, y_pred, n_class = self.session_cm[session]
            counts = cm_counts(y_true, y_pred, n_class)
            names = class_names(args) or [str(i) for i in range(n_class)]
            header = ["true \\ pred"] + list(names[:n_class]) + ["total"]
            rows = [
                [names[i]] + [int(v) for v in counts[i]] + [int(counts[i].sum())]
                for i in range(n_class)
            ]
            out_dir = os.path.join(args.save_path, "confusion")
            ensure_path(out_dir)
            with open(os.path.join(out_dir, f"session{session}.csv"), "w", newline="") as f:
                csv.writer(f).writerows([header] + rows)
            if hasattr(args, "use_wandb") and args.use_wandb:
                return self.wandb.Table(columns=header, data=rows)
            return None
        except Exception as e:
            print(f"Warning: Could not write confusion counts for session {session}: {e}")
            return None

    def _session_cm_image(self, session):
        """wandb.Image of the recorded checkpoint's CM, or None if unavailable."""
        if session not in self.session_cm:
            return None
        try:
            import matplotlib.pyplot as plt

            y_true, y_pred, n_class = self.session_cm[session]
            fig = cm_figure(y_true, y_pred, n_class, class_names(self.args))
            if fig is None:
                return None
            img = self.wandb.Image(fig)
            plt.close(fig)
            return img
        except Exception as e:
            print(f"Warning: Could not build confusion matrix image: {e}")
            return None

    def _proj_matrix(self, net, test_class):
        return torch.mm(
            self.dummy_classifiers,
            F.normalize(
                torch.transpose(net.fc.weight[:test_class, :], 1, 0),
                p=2,
                dim=-1,
            ),
        )

    def _integrated_probs(self, net, emb, proj_matrix, test_class):
        """Class probabilities of the incremental-session classifier:
        eta * (dummy-prototype projection) + (1 - eta) * (cosine fc head)."""
        eta = self.args.eta
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
        logits2 = net.forpass_fc_emb(emb)[:, :test_class]
        return eta * F.softmax(logits1, dim=1) + (1 - eta) * F.softmax(logits2, dim=1)

    def _session_probs(self, net, emb, session):
        """Probabilities over the seen classes, computed like that session's
        evaluation (test() for session 0, test_intergrate() afterwards)."""
        test_class = self.args.base_class + session * self.args.way
        if session == 0:
            return F.softmax(net.forpass_fc_emb(emb)[:, :test_class], dim=1)
        proj_matrix = self._proj_matrix(net, test_class)
        return self._integrated_probs(net, emb, proj_matrix, test_class)

    def test_intergrate(
        self, model, testloader, epoch, args, session, validation=True, return_stats=False
    ):
        test_class = args.base_class + session * args.way
        model = model.eval()
        # per-batch means kept on device (one sync at the end, same batch-mean
        # averaging as Averager); logits/labels moved to CPU once
        vl = _GpuMean()
        va = _GpuMean()
        va5 = _GpuMean()
        lgt_list = []
        lbs_list = []

        net = model.module if hasattr(model, "module") else model
        proj_matrix = self._proj_matrix(net, test_class)
        k5 = min(5, test_class)

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.to(args.device, non_blocking=True) for _ in batch]

                emb = net.encode(data)
                logits = self._integrated_probs(net, emb, proj_matrix, test_class)

                loss = F.cross_entropy(logits, test_label)
                vl.add(loss)
                va.add((torch.argmax(logits, dim=1) == test_label).float().mean())
                # same as count_acc_topk: hits among the top-k / batch size
                top5 = torch.topk(logits, k5, dim=-1).indices
                va5.add((top5 == test_label.view(-1, 1)).sum().double() / test_label.size(0))
                lgt_list.append(logits)
                lbs_list.append(test_label)
            vl = vl.item()
            va = va.item()
            va5 = va5.item()
            lgt = torch.cat(lgt_list, dim=0).cpu()
            lbs = torch.cat(lbs_list, dim=0).cpu()
            print(f"epo {epoch}, test, loss={vl:.4f} acc={va:.4f}, acc@5={va5:.4f}")

        if return_stats:
            return vl, va, eval_stats(lgt, lbs, test_class, args.base_class)
        return vl, va

    def set_save_path(self):
        # one flat dir, tracked as the DVC train output; the run's settings
        # live in its config.yaml, not in the directory name
        self.args.save_path = "checkpoint_debug" if self.args.debug else "checkpoint"
        ensure_path(self.args.save_path)
