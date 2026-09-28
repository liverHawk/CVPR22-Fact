import os
import pprint as pprint
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

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


def old_new_acc(logits, labels, base_class):
    """Top-1 accuracy (percent) on old (< base_class) / new samples and their
    harmonic mean; a split with no test samples (e.g. new in session 0) is None."""
    correct = (torch.argmax(logits, dim=1) == labels).float()
    old = labels < base_class
    old_acc = float(correct[old].mean() * 100) if old.any() else None
    new_acc = float(correct[~old].mean() * 100) if (~old).any() else None
    if old_acc is None or new_acc is None:
        hm = None
    elif old_acc + new_acc == 0:
        hm = 0.0
    else:
        hm = 2 * old_acc * new_acc / (old_acc + new_acc)
    return old_acc, new_acc, hm


# per-session old/new split metrics (percent, None where a split is empty);
# trlog, metrics json, wandb session/* and the t-SNE export all use this list
SPLIT_METRICS = ("old_acc", "new_acc", "hm", "old_f1", "new_f1", "hm_f1")


def _hm(a, b):
    if a is None or b is None:
        return None
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def old_new_f1(logits, labels, n_class, base_class):
    """Macro F1 (percent) over old classes (< base_class) and over new ones
    (base_class..n_class-1), from predictions over all n_class seen classes,
    plus their harmonic mean; a split with no classes is None."""
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    y = labels.cpu().numpy()

    def split(ids):
        if not ids:
            return None
        return float(
            f1_score(y, preds, labels=ids, average="macro", zero_division=0) * 100
        )

    old_f1 = split(list(range(min(base_class, n_class))))
    new_f1 = split(list(range(base_class, n_class)))
    return old_f1, new_f1, _hm(old_f1, new_f1)


def eval_stats(logits, labels, n_class, base_class):
    """Per-session metrics recorded next to top-1 acc: macro F1 + SPLIT_METRICS."""
    old_acc, new_acc, hm = old_new_acc(logits, labels, base_class)
    old_f1, new_f1, hm_f1 = old_new_f1(logits, labels, n_class, base_class)
    return {
        "f1": macro_f1(logits, labels, n_class),
        "old_acc": old_acc,
        "new_acc": new_acc,
        "hm": hm,
        "old_f1": old_f1,
        "new_f1": new_f1,
        "hm_f1": hm_f1,
        # for the per-session confusion matrix
        "y_true": labels.cpu().numpy().astype(int),
        "y_pred": torch.argmax(logits, dim=1).cpu().numpy().astype(int),
        "n_class": n_class,
    }


def macro_f1(logits, labels, n_class):
    """Macro F1 (percent) over the n_class seen classes."""
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    return float(
        f1_score(
            labels.cpu().numpy(),
            preds,
            labels=list(range(n_class)),
            average="macro",
            zero_division=0,
        )
        * 100
    )


def class_names(args):
    """Class-id -> name list from <dataroot>/<dataset>/label_map.json, or None
    (image datasets / no label map), in which case plots fall back to ids."""
    import json

    path = os.path.join(
        os.path.expanduser(str(args.dataroot)), args.dataset, "label_map.json"
    )
    try:
        with open(path) as f:
            label_map = json.load(f)
    except (OSError, ValueError):
        return None
    names = [None] * len(label_map)
    for name, i in label_map.items():
        names[int(i)] = name
    return names if None not in names else None


def stratified_indices(labels, n_total, seed):
    """Up to n_total indices with an equal cap per class, so rare classes
    still show up in plots (classes smaller than the cap are taken whole)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    cap = max(1, int(np.ceil(n_total / max(len(classes), 1))))
    rng = np.random.default_rng(seed)
    picked = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        picked.append(idx if len(idx) <= cap else rng.choice(idx, cap, replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def tsne_embed(feats, seed=1):
    """[N, D] features -> [N, 2] t-SNE coords, or None with fewer than 3 points."""
    from sklearn.manifold import TSNE

    feats = np.asarray(feats, dtype=np.float32).reshape(len(feats), -1)
    n = len(feats)
    if n < 3:
        return None
    return TSNE(
        n_components=2,
        init="pca",
        perplexity=min(30.0, (n - 1) / 3),
        random_state=seed,
    ).fit_transform(feats)


def tsne_figure(xy, labels, names=None, base_class=None, title="t-SNE", mix_xy=None):
    """Scatter of t-SNE coords xy [N, 2] colored by class; classes >=
    base_class are drawn as triangles, mixup points (mix_xy) as grey x.
    Caller closes it (plt.close)."""
    labels = np.asarray(labels, dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=80)
    # tab20 comes in dark/light pairs; use the 10 dark ones before any light
    # one so neighbouring class ids don't get near-identical colors
    tab20 = plt.get_cmap("tab20").colors
    palette = tab20[0::2] + tab20[1::2]
    for c in np.unique(labels):
        m = labels == c
        new = base_class is not None and c >= base_class
        name = names[c] if names is not None and c < len(names) else str(c)
        ax.scatter(
            xy[m, 0], xy[m, 1], s=6, alpha=0.7, color=palette[c % 20],
            marker="^" if new else "o", label=name + (" (new)" if new else ""),
        )
    if mix_xy is not None and len(mix_xy):
        ax.scatter(
            mix_xy[:, 0], mix_xy[:, 1], s=10, marker="x", linewidths=0.8,
            color="0.35", alpha=0.6, label="mixup",
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(
        loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
        markerscale=2.5, frameon=False,
    )
    fig.tight_layout()
    return fig


def cm_counts(y_true, y_pred, n_class):
    """Raw K x K confusion-matrix counts (rows = true, cols = predicted);
    out-of-range ids are dropped, as in cm_figure."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    valid = (y_true >= 0) & (y_true < n_class) & (y_pred >= 0) & (y_pred < n_class)
    return confusion_matrix(y_true[valid], y_pred[valid], labels=list(range(n_class)))


def log_wandb_cm_image(
    y_true, y_pred, n_class, step=None, key="confusion_matrix", names=None, extra=None
):
    """Lightweight confusion-matrix logging: aggregated KxK image, no per-sample table.

    Replaces wandb.plot.confusion_matrix (which uploads one table row per
    test sample every call) with a single small wandb.Image. names (class-id
    -> label, e.g. from class_names) replaces the numeric tick labels; extra
    is merged into the same log call (e.g. {"epoch": e} for the x-axis).
    """
    import wandb

    fig = cm_figure(y_true, y_pred, n_class, names)
    if fig is None:
        return
    payload = {key: wandb.Image(fig), **(extra or {})}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)
    plt.close(fig)


def cm_figure(y_true, y_pred, n_class, names=None):
    """Row-normalized KxK confusion-matrix figure, or None with no valid samples.
    Caller closes it (plt.close)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    n_class = int(n_class)
    if y_true.size == 0:
        return None
    # Clip out-of-range labels (can happen with partial test_class slices)
    valid = (y_true >= 0) & (y_true < n_class) & (y_pred >= 0) & (y_pred < n_class)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if y_true.size == 0:
        return None

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_class)))
    # Row-normalize for readability; keep raw counts out of the payload
    with np.errstate(invalid="ignore", divide="ignore"):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_n = np.divide(
            cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0
        )

    named = names is not None and len(names) >= n_class and n_class <= 20
    # named ticks need room for strings like "Web Attack - Sql Injection"
    figsize = (0.45 * n_class + 4, 0.45 * n_class + 3.5) if named else (4, 3.5)
    fig, ax = plt.subplots(figsize=figsize, dpi=80)
    im = ax.imshow(cm_n, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (K={n_class})")
    # Sparse ticks only: full tick labels for K=200 are unreadable and heavy
    if named:
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
        ax.set_xticklabels(names[:n_class], rotation=90, fontsize=8)
        ax.set_yticklabels(names[:n_class], fontsize=8)
    elif n_class <= 20:
        ax.set_xticks(range(n_class))
        ax.set_yticks(range(n_class))
    else:
        ticks = np.linspace(0, n_class - 1, 6).astype(int)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


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
