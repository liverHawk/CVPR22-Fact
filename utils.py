import os
import pprint as pprint
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def set_seed(seed):
    if seed == 0:
        print(" random seed")
        torch.backends.cudnn.benchmark = True
    else:
        print("manual seed:", seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(",")]
    print("use gpu:", gpu_list)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print("create folder:", path)
        os.makedirs(path)


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return f"{x / 3600:.1f}h"
        if x >= 60:
            return f"{round(x / 60)}m"
        return f"{x}s"


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()


def count_acc_topk(x, y, k=5):
    k = min(k, x.size(-1))
    _, maxk = torch.topk(x, k, dim=-1)
    total = y.size(0)
    test_labels = y.view(-1, 1)
    # top1=(test_labels == maxk[:,0:1]).sum().item()
    topk = (test_labels == maxk).sum().item()
    return float(topk / total)


def count_acc_taskIL(logits, label, args):
    basenum = args.base_class
    incrementnum = (args.num_classes - args.base_class) / args.way
    for i in range(len(label)):
        currentlabel = label[i]
        if currentlabel < basenum:
            logits[i, basenum:] = -1e9
        else:
            space = int((currentlabel - basenum) / args.way)
            low = basenum + space * args.way
            high = low + args.way
            logits[i, :low] = -1e9
            logits[i, high:] = -1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def confmatrix(logits, label, filename):

    font = {"family": "FreeSerif", "size": 18}
    matplotlib.rc("font", **font)
    matplotlib.rcParams.update({"font.family": "FreeSerif", "font.size": 18})
    plt.rcParams["font.family"] = "FreeSerif"

    pred = torch.argmax(logits, dim=1)
    cm = confusion_matrix(label, pred, normalize="true")
    # print(cm)
    clss = len(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks(
            [0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16
        )
        plt.xticks(
            [0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16
        )

    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + ".pdf", bbox_inches="tight")
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(cm, cmap=plt.cm.jet)
    cbar = plt.colorbar(cax)  # This line includes the color bar
    cbar.ax.tick_params(labelsize=16)
    if clss <= 100:
        plt.yticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
        plt.xticks([0, 19, 39, 59, 79, 99], [0, 20, 40, 60, 80, 100], fontsize=16)
    elif clss <= 200:
        plt.yticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
        plt.xticks([0, 39, 79, 119, 159, 199], [0, 40, 80, 120, 160, 200], fontsize=16)
    else:
        plt.yticks(
            [0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16
        )
        plt.xticks(
            [0, 199, 399, 599, 799, 999], [0, 200, 400, 600, 800, 1000], fontsize=16
        )
    plt.xlabel("Predicted Label", fontsize=20)
    plt.ylabel("True Label", fontsize=20)
    plt.tight_layout()
    plt.savefig(filename + "_cbar.pdf", bbox_inches="tight")
    plt.close()

    return cm


def log_wandb_cm_image(y_true, y_pred, n_class, step=None, key="confusion_matrix"):
    """Lightweight confusion-matrix logging: aggregated KxK image, no per-sample table.

    Replaces wandb.plot.confusion_matrix (which uploads one table row per
    test sample every call) with a single small wandb.Image.
    """
    import wandb

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n_class = int(n_class)
    if y_true.size == 0:
        return
    # Clip out-of-range labels (can happen with partial test_class slices)
    valid = (y_true >= 0) & (y_true < n_class) & (y_pred >= 0) & (y_pred < n_class)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_class)))
    # Row-normalize for readability; keep raw counts out of the payload
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_n = np.divide(
            cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0
        )

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=80)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (K={n_class})")
    # Sparse ticks only: full tick labels for K=200 are unreadable and heavy
    if n_class <= 20:
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
    else:
        ticks = np.linspace(0, n_class - 1, 6).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if step is None:
        wandb.log({key: wandb.Image(fig)})
    else:
        wandb.log({key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def should_log_wandb_cm(args, session, epoch):
    """Throttle CM logging: incremental sessions log once; base session logs sparsely."""
    freq = getattr(args, "wandb_cm_freq", 20)
    if freq is not None and freq <= 0:  # 0 / negative disables CM logging
        return False
    if session != 0:
        return True  # called ~once per incremental session
    try:
        total = getattr(args, "epochs_base", None)
        if total is not None and int(epoch) >= int(total) - 1:
            return True  # always log final base epoch
    except Exception:
        pass
    return int(epoch) % int(freq) == 0


def save_list_to_txt(name, input_list):
    f = open(name, mode="w")
    f.writelines(str(item) + "\n" for item in input_list)
    f.close()
