try:
    import comet_ml
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
import random
import torch
import os
import time
import numpy as np
import pprint as pprint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import matplotlib
import seaborn as sns
_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def get_device():
    """利用可能なデバイスを返す（GPUが利用可能な場合はGPU、そうでない場合はCPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_module(model):
    """モデルから実際のモデルオブジェクトを取得（DataParallelの場合はmodule、そうでない場合はmodel自体）"""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def set_seed(seed):
    if seed == 0:
        print(' random seed')
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def set_gpu(args):
    """GPU設定を行い、利用可能なGPU数を返す（CPUの場合は0を返す）"""
    if not torch.cuda.is_available():
        print('CUDA is not available. Using CPU.')
        return 0
    
    if args.gpu.lower() == 'cpu' or args.gpu == '':
        print('Using CPU (specified by user).')
        return 0
    
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def count_acc_topk(x,y,k=5):
    _,maxk = torch.topk(x,k,dim=-1)
    total = y.size(0)
    test_labels = y.view(-1,1) 
    #top1=(test_labels == maxk[:,0:1]).sum().item()
    topk=(test_labels == maxk).sum().item()
    return float(topk/total)

def count_acc_taskIL(logits, label,args):
    basenum=args.base_class
    incrementnum=(args.num_classes-args.base_class)/args.way
    for i in range(len(label)):
        currentlabel=label[i]
        if currentlabel<basenum:
            logits[i,basenum:]=-1e9
        else:
            space=int((currentlabel-basenum)/args.way)
            low=basenum+space*args.way
            high=low+args.way
            logits[i,:low]=-1e9
            logits[i,high:]=-1e9

    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()

def _resolve_dataset(dataset):
    """Unwraps Subset/DataLoader wrappers to the base dataset."""
    if hasattr(dataset, 'dataset'):
        return _resolve_dataset(dataset.dataset)
    return dataset


def get_dataset_label_names(dataset):
    """Returns a list of label names in index order if available."""
    if dataset is None:
        return None
    base_dataset = _resolve_dataset(dataset)

    if hasattr(base_dataset, 'idx_to_label'):
        label_map = base_dataset.idx_to_label
        sorted_indices = sorted(label_map.keys())
        return [label_map[idx] for idx in sorted_indices]
    if hasattr(base_dataset, 'classes'):
        return list(base_dataset.classes)
    if hasattr(base_dataset, 'class_to_idx'):
        sorted_items = sorted(base_dataset.class_to_idx.items(), key=lambda item: item[1])
        return [item[0] for item in sorted_items]
    return None


def confmatrix(logits, label, filename, label_names=None):
    font={'family':'DejaVu Serif','size':18}
    matplotlib.rc('font',**font)
    matplotlib.rcParams.update({'font.family':'DejaVu Serif','font.size':18})
    plt.rcParams["font.family"]="DejaVu Serif"

    pred = torch.argmax(logits, dim=1)
    cm=confusion_matrix(label, pred, normalize='true')
    clss=len(cm)

    # アノテーションの準備（0以外の値のみ表示）
    annot = np.full(cm.shape, '', dtype=object)
    for i in range(clss):
        for j in range(clss):
            value = cm[i, j]
            if value != 0:
                annot[i, j] = f'{value:.2f}'

    # ラベルの準備
    if label_names and len(label_names) >= clss:
        tick_labels = label_names[:clss]
    else:
        tick_labels = [str(i) for i in range(clss)]
    
    # クラス数に応じたサイズとフォント調整
    if clss <= 10:
        figsize = (10, 8)
        label_fontsize = 10
        annot_fontsize = 8
        rotation = 45
    elif clss <= 20:
        figsize = (14, 12)
        label_fontsize = 8
        annot_fontsize = 6
        rotation = 45
    else:
        figsize = (max(16, clss * 0.5), max(14, clss * 0.45))
        label_fontsize = 6
        annot_fontsize = 4
        rotation = 90

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap='Blues', annot=annot, fmt='', cbar=True, ax=ax, square=True,
                linewidths=0.5, linecolor='white',
                xticklabels=tick_labels, yticklabels=tick_labels,
                annot_kws={'fontsize': annot_fontsize})
    
    if ax.collections:
        ax.collections[0].colorbar.ax.tick_params(labelsize=12)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_ylabel('True Label', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename+'_cbar.pdf', bbox_inches='tight', dpi=150)
    plt.close()

    return cm





def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()


# Comet ML統合用のユーティリティ関数

def init_comet_experiment(args):
    """
    Comet ML実験を初期化する
    
    Args:
        args: コマンドライン引数オブジェクト
        
    Returns:
        comet_ml.Experiment: Comet実験オブジェクト、またはNone（Cometが無効な場合）
    """
    if not COMET_AVAILABLE:
        print("Comet ML is not available. Install with: pip install comet-ml")
        return None
    
    # Cometが無効化されている場合はNoneを返す
    if hasattr(args, 'comet_disabled') and args.comet_disabled:
        return None
    
    try:
        # 初回実行時にインタラクティブログイン
        try:
            comet_ml.login()
        except Exception:
            # 既にログイン済みの場合はスキップ
            pass
        
        # プロジェクト名と実験名を設定
        project_name = getattr(args, 'comet_project', args.dataset)
        experiment_name = f"{args.project}_{args.dataset}_{args.base_mode}_{args.new_mode}"
        
        # 実験を開始
        exp = comet_ml.Experiment(
            project_name=project_name,
            experiment_name=experiment_name,
            auto_param_logging=False,  # 手動でパラメータをログ
            auto_metric_logging=False,  # 手動でメトリクスをログ
            disabled=getattr(args, 'comet_disabled', False)
        )
        
        # ハイパーパラメータをログ
        params_dict = {}
        for key, value in vars(args).items():
            # オブジェクトや関数は除外
            if not key.startswith('_') and not callable(value):
                try:
                    # シリアライズ可能な値のみログ
                    if isinstance(value, (str, int, float, bool, type(None))):
                        params_dict[key] = value
                    elif isinstance(value, (list, tuple)):
                        params_dict[key] = list(value)
                except Exception:
                    pass
        
        exp.log_parameters(params_dict)
        
        print(f"Comet ML experiment started: {experiment_name}")
        return exp
        
    except Exception as e:
        print(f"Failed to initialize Comet ML: {e}")
        return None


def log_metrics_to_comet(exp, metrics, epoch=None, session=None, step=None):
    """
    Comet MLにメトリクスをログする
    
    Args:
        exp: Comet実験オブジェクト（Noneの場合は何もしない）
        metrics: メトリクスの辞書
        epoch: エポック番号（オプション）
        session: セッション番号（オプション）
        step: ステップ番号（オプション、epochとsessionから自動計算される）
    """
    if exp is None:
        return
    
    try:
        # ステップ番号を決定
        if step is None:
            if epoch is not None and session is not None:
                step = session * 1000 + epoch  # セッションごとに1000エポック分のオフセット
            elif epoch is not None:
                step = epoch
            elif session is not None:
                step = session * 1000
            else:
                step = None
        
        # メトリクスをログ
        if step is not None:
            exp.log_metrics(metrics, step=step)
        else:
            exp.log_metrics(metrics)
            
        # コンテキスト情報を追加
        if epoch is not None:
            exp.log_metrics({f"epoch": epoch}, step=step if step is not None else epoch)
        if session is not None:
            exp.log_metrics({f"session": session}, step=step if step is not None else session * 1000)
            
    except Exception as e:
        print(f"Failed to log metrics to Comet ML: {e}")


def log_confusion_matrix_to_comet(exp, y_true, y_pred, labels=None, session=None, title=None):
    """
    Comet MLに混同行列をログする（log_confusion_matrix APIを使用）
    
    Args:
        exp: Comet実験オブジェクト（Noneの場合は何もしない）
        y_true: 真のラベルのリストまたはnumpy配列
        y_pred: 予測ラベルのリストまたはnumpy配列
        labels: ラベル名のリスト（オプション）
        session: セッション番号（オプション）
        title: 混同行列のタイトル（オプション）
    """
    if exp is None:
        return
    
    try:
        # numpy配列またはTensorを整数のリストに変換
        # Comet MLは整数のラベルを要求するため、明示的に整数型に変換
        if isinstance(y_true, torch.Tensor):
            # Tensorから直接整数型のリストに変換
            y_true = y_true.cpu().long().numpy().astype(np.int64).tolist()
        elif isinstance(y_true, np.ndarray):
            y_true = y_true.astype(np.int64).tolist()
        else:
            # リストやその他のイテラブルの場合
            y_true = [int(float(x)) for x in y_true]
            
        if isinstance(y_pred, torch.Tensor):
            # Tensorから直接整数型のリストに変換
            y_pred = y_pred.cpu().long().numpy().astype(np.int64).tolist()
        elif isinstance(y_pred, np.ndarray):
            y_pred = y_pred.astype(np.int64).tolist()
        else:
            # リストやその他のイテラブルの場合
            y_pred = [int(float(x)) for x in y_pred]
        
        # タイトルを設定
        if title is None:
            if session is not None:
                title = f"Session {session} Confusion Matrix"
            else:
                title = "Confusion Matrix"
        
        # Comet MLのlog_confusion_matrix APIを使用
        exp.log_confusion_matrix(
            y_true=y_true,
            y_predicted=y_pred,
            labels=labels,
            title=title
        )
        
    except Exception as e:
        print(f"Failed to log confusion matrix to Comet ML: {e}")


