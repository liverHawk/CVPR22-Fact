import random
import torch
import os
import time
import numpy as np
import pprint as pprint
from typing import Any, Dict, Optional
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import matplotlib

try:
    import yaml
except ImportError:
    yaml = None
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
            return "{:.1f}h".format(x / 3600)
        if x >= 60:
            return "{}m".format(round(x / 60))
        return "{}s".format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def count_acc_topk(x, y, k=5):
    _, maxk = torch.topk(x, k, dim=-1)
    total = y.size(0)
    test_labels = y.view(-1, 1)
    # top1=(test_labels == maxk[:,0:1]).sum().item()
    topk = (test_labels == maxk).sum().item()
    return float(topk / total)


def count_acc_taskIL(logits, label, args):
    basenum = args.base_class
    # incrementnum = (args.num_classes - args.base_class) / args.way
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


# def cal_macro_recall(logits, labels, num_classes):
#     """
#     マクロリコールを計算する関数

#     Args:
#         logits (torch.Tensor): モデルの出力ロジット (バッチサイズ x クラス数)
#         label (torch.Tensor): 真のラベル (バッチサイズ)
#         num_classes (int): クラス数

#     Returns:
#         float: マクロリコール
#     """
#     preds = torch.argmax(logits, dim=1)
    
#     # クラスごとのTPとFNを計算
#     recall_per_class = []
#     for class_idx in range(num_classes):
#         # 現在のクラスに対するマスク
#         true_positive = ((preds == class_idx) & (labels == class_idx)).sum().item()
#         false_negative = ((preds != class_idx) & (labels == class_idx)).sum().item()

#         # リコールを計算（ゼロ除算を防ぐ）
#         recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
#         recall_per_class.append(recall)

#     # マクロリコールを計算
#     macro_recall = sum(recall_per_class) / num_classes
#     return macro_recall


def confmatrix(logits, label, filename, args, label_names=None, name=None, step=0):
    font = {"family": "DejaVu Serif", "size": 18}
    matplotlib.rc("font", **font)
    matplotlib.rcParams.update({"font.family": "DejaVu Serif", "font.size": 18})
    plt.rcParams["font.family"] = "DejaVu Serif"

    pred = torch.argmax(logits, dim=1)
    # label, pred -> int array
    label_array = label.cpu().numpy().astype(int)
    pred_array = pred.cpu().numpy().astype(int)
    args.comet.log_confusion_matrix(
        label_array, pred_array, class_names=label_names, file_name=f"{name}.json", step=step
    )
    # 正規化された混同行列（割合）
    cm_normalized = confusion_matrix(label, pred, normalize="true")
    # 生の混同行列（カウント）
    # cm_counts = confusion_matrix(label, pred, normalize=None)
    clss = len(cm_normalized)

    # ラベル名を取得
    if label_names is None:
        # ラベル名が提供されていない場合は数値ラベルを使用
        label_names = [str(i) for i in range(clss)]
    else:
        # ラベル名が提供されている場合、必要な数だけ使用
        if len(label_names) < clss:
            # 不足している場合は数値で補完
            label_names = label_names + [str(i) for i in range(len(label_names), clss)]
        else:
            # 必要な数だけ使用
            label_names = label_names[:clss]

    # すべてのクラスにチックを配置（ラベル名を全て表示）
    tick_positions = list(range(clss))
    tick_labels = [label_names[i] for i in range(clss)]

    # クラス数に応じてフォントサイズを調整
    if clss <= 10:
        tick_fontsize = 12
        cell_fontsize = 10
    elif clss <= 20:
        tick_fontsize = 10
        cell_fontsize = 8
    elif clss <= 50:
        tick_fontsize = 8
        cell_fontsize = 6
    else:
        tick_fontsize = 6
        cell_fontsize = 4

    # 混同行列を1つだけ作成（カラーバー付き、割合を表示）
    # クラス数に応じて図のサイズを調整
    fig_size = max(10, clss * 0.5)
    fig = plt.figure(figsize=(fig_size, fig_size * 0.8))
    ax = fig.add_subplot(111)
    # extentで正確な範囲を指定: [left, right, bottom, top]
    # 0からclss-1までの範囲を正確に表示
    cax = ax.imshow(
        cm_normalized,
        cmap=plt.cm.Blues,
        extent=[-0.5, clss - 0.5, clss - 0.5, -0.5],
        aspect="auto",
    )
    ax.set_xlim(-0.5, clss - 0.5)
    ax.set_ylim(clss - 0.5, -0.5)

    # 各セルに割合（0~1の範囲）を表示（クラス数が多い場合は省略）
    if clss <= 30:
        thresh = cm_normalized.max() / 2.0
        for i in range(clss):
            for j in range(clss):
                text = f"{cm_normalized[i, j]:.2f}"
                text_obj = ax.text(
                    j,
                    i,
                    text,
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=cell_fontsize,
                )
                # 対角要素（正しく分類された要素）に下線を引く
                if i == j:
                    text_obj.set_fontweight("bold")

    cbar = plt.colorbar(cax)  # This line includes the color bar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Normalized Count", fontsize=14)
    
    # 縦軸ラベル
    plt.yticks(tick_positions, tick_labels, fontsize=tick_fontsize)
    # 横軸ラベルを45度回転させて重なりを防ぐ
    plt.xticks(tick_positions, tick_labels, fontsize=tick_fontsize, rotation=45, ha="right")
    plt.xlabel("Predicted Label", fontsize=16)
    plt.ylabel("True Label", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename + ".png", bbox_inches="tight", dpi=300)
    plt.close()

    return cm_normalized


def save_classification_report(logits, label, filename):
    """
    Classification reportをファイルに保存
    """
    pred = torch.argmax(logits, dim=1)

    # NumPy配列に変換
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(label, torch.Tensor):
        label = label.cpu().numpy()

    # Classification reportを生成
    report = classification_report(label, pred, output_dict=False, zero_division=0)

    # ファイルに保存
    with open(filename + "_classification_report.txt", "w") as f:
        f.write(report)

    return report


class WandbLogger:
    """
    Weights & Biasesへのロギングをラップする軽量ユーティリティ.
    """

    def __init__(self, args):
        self.enabled = bool(getattr(args, "use_wandb", False))
        self.run = None
        self._wandb = None
        self._watch_mode = getattr(args, "wandb_watch", "gradients")
        self._watch_freq = int(getattr(args, "wandb_watch_freq", 100))
        if not self.enabled:
            return
        try:
            import wandb
        except ImportError as exc:
            print(f"wandbのインポートに失敗したため、ロギングを無効化します: {exc}")
            self.enabled = False
            return

        self._wandb = wandb
        tags = getattr(args, "wandb_tags", None)
        if tags == []:
            tags = None

        project = (
            getattr(args, "wandb_project", None)
            or f"CVPR22-Fact-{getattr(args, 'dataset', 'default')}"
        )
        entity = getattr(args, "wandb_entity", None)
        group = getattr(args, "wandb_group", None)
        run_name = getattr(args, "wandb_run_name", None)
        mode = getattr(args, "wandb_mode", "online")
        config = self._build_config(vars(args))

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            mode=mode,
            config=config,
        )

    def _coerce_value(self, value: Any):
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._coerce_value(v) for v in value]
        if isinstance(value, set):
            return [
                self._coerce_value(v) for v in sorted(value, key=lambda item: str(item))
            ]
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)

    def _build_config(self, args_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: self._coerce_value(val)
            for key, val in args_dict.items()
            if not key.startswith("_")
        }

    def log_metrics(
        self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True
    ):
        if not self.enabled or not metrics:
            return
        self._wandb.log(metrics, step=step, commit=commit)

    def watch(self, model):
        if not self.enabled or self._watch_mode == "none" or model is None:
            return
        log_target = "all" if self._watch_mode == "all" else self._watch_mode
        self._wandb.watch(model, log=log_target, log_freq=self._watch_freq)

    def log_image(self, label: str, image_path: str):
        if not self.enabled or not image_path or not os.path.exists(image_path):
            return
        self._wandb.log({label: self._wandb.Image(image_path, caption=label)})

    def set_summary(self, **kwargs):
        if not self.enabled or self.run is None:
            return
        for key, value in kwargs.items():
            self.run.summary[key] = value

    def finish(self):
        if not self.enabled or self.run is None:
            return
        self.run.finish()


def save_list_to_txt(name, input_list):
    f = open(name, mode="w")
    for item in input_list:
        f.write(str(item) + "\n")
    f.close()


def load_params_yaml(yaml_path="params.yaml"):
    """
    params.yamlファイルを読み込む

    Args:
        yaml_path: YAMLファイルのパス

    Returns:
        dict: パラメータの辞書
    """
    if yaml is None:
        raise ImportError(
            "PyYAMLが必要です。`uv add pyyaml`でインストールしてください。"
        )

    if not os.path.exists(yaml_path):
        print(f"Warning: {yaml_path} not found, returning empty dict")
        return {}

    with open(yaml_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    return params if params else {}
