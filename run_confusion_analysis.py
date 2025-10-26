#!/usr/bin/env python3
"""
混同行列分析の実行スクリプト
既存のモデルチェックポイントを使用して混同行列を生成・可視化
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # seabornが利用できない場合はコメントアウト
import argparse
import json

# プロジェクトのルートディレクトリをパスに追加
sys.path.append('/home/hawk/Documents/school/test/CVPR22-Fact')

from utils import confmatrix
from models.fact.Network import MYNET as FACT_MYNET
from models.base.Network import MYNET as BASE_MYNET
from dataloader.data_utils import set_up_datasets, get_dataloader

# CICIDS2017のクラス名マッピング
CICIDS2017_CLASS_NAMES = {
    0: 'BENIGN',
    1: 'Botnet', 
    2: 'DDoS',
    3: 'DoS',
    4: 'FTP-Patator',
    5: 'Heartbleed',
    6: 'Infiltration',
    7: 'Portscan',
    8: 'SSH-Patator',
    9: 'Web Attack'
}

def analyze_session(session, args, checkpoint_dir, output_dir):
    """指定されたセッションを分析"""
    print(f"Session {session} を分析中...")
    
    # モデルのセットアップ（チェックポイントディレクトリからモデルタイプを判定）
    if 'fact' in checkpoint_dir:
        model = FACT_MYNET(args, mode='ft_cos')
    else:
        model = BASE_MYNET(args, mode='ft_cos')
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()
    
    # チェックポイントの読み込み
    checkpoint_path = os.path.join(checkpoint_dir, f'session{session}_max_acc.pth')
    if not os.path.exists(checkpoint_path):
        print(f"チェックポイントが見つかりません: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    
    # チェックポイントの読み込み（不足キーを許可）
    try:
        model.load_state_dict(checkpoint['params'], strict=True)
        print(f"モデルを読み込みました: {checkpoint_path}")
    except RuntimeError as e:
        if "Missing key(s)" in str(e):
            print("警告: 一部のキーが不足しています。strict=Falseで読み込みます。")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['params'], strict=False)
            if missing_keys:
                print(f"不足キー: {missing_keys}")
            if unexpected_keys:
                print(f"予期しないキー: {unexpected_keys}")
            print(f"モデルを読み込みました（部分読み込み）: {checkpoint_path}")
        else:
            raise e
    
    # データローダーの準備
    train_set, trainloader, testloader = get_dataloader(args, session)
    
    # テストクラス数の計算
    test_class = args.base_class + session * args.way
    
    # モデルを評価モードに設定
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            data, labels = [_.cuda() for _ in batch]
            
            # 予測の生成
            if session == 0:
                # ベースセッション
                logits = model(data)
            else:
                # インクリメンタルセッション
                if 'fact' in checkpoint_dir:
                    logits = model.module.forpass_fc(data)
                else:
                    # baseモデルの場合
                    logits = model(data)
            
            # 現在のセッションまでのクラスのみ使用
            logits = logits[:, :test_class]
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 混同行列の生成
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions, normalize='true')
    
    # 結果の保存
    session_dir = os.path.join(output_dir, f'session_{session}')
    os.makedirs(session_dir, exist_ok=True)
    
    # 1. 混同行列の可視化
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(label='Normalized Count')
    
    # クラス名の設定
    if args.dataset == 'cicids2017_improved':
        class_labels = [CICIDS2017_CLASS_NAMES.get(i, f'Class {i}') for i in range(test_class)]
        plt.xticks(range(test_class), class_labels, rotation=45, ha='right')
        plt.yticks(range(test_class), class_labels)
    else:
        plt.xticks(range(test_class))
        plt.yticks(range(test_class))
    
    plt.title(f'Confusion Matrix - Session {session}\n'
             f'Classes: {test_class}, Overall Accuracy: {np.mean(np.array(all_labels) == np.array(all_predictions)):.3f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, 'confusion_matrix.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. クラス別精度の分析
    per_class_acc = cm.diagonal()
    
    # ベースクラスと新規クラスの分離
    base_classes = per_class_acc[:args.base_class]
    new_classes = per_class_acc[args.base_class:] if session > 0 else []
    
    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ベースクラスの精度
    ax1.plot(range(len(base_classes)), base_classes, 'b-', marker='o', markersize=3)
    ax1.set_title(f'Base Classes Accuracy (Session {session})')
    ax1.set_xlabel('Class Index')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 新規クラスの精度（存在する場合）
    if len(new_classes) > 0:
        ax2.plot(range(args.base_class, args.base_class + len(new_classes)), 
                new_classes, 'r-', marker='s', markersize=3)
        ax2.set_title(f'New Classes Accuracy (Session {session})')
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No new classes in base session', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('New Classes Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(session_dir, 'per_class_accuracy.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 統計情報の保存
    overall_accuracy = np.mean(np.array(all_labels) == np.array(all_predictions))
    base_mean = np.mean(base_classes)
    base_std = np.std(base_classes)
    
    stats = {
        'session': session,
        'overall_accuracy': float(overall_accuracy),
        'base_class_accuracy_mean': float(base_mean),
        'base_class_accuracy_std': float(base_std),
        'base_class_accuracy_min': float(np.min(base_classes)),
        'base_class_accuracy_max': float(np.max(base_classes)),
    }
    
    if len(new_classes) > 0:
        stats.update({
            'new_class_accuracy_mean': float(np.mean(new_classes)),
            'new_class_accuracy_std': float(np.std(new_classes)),
            'new_class_accuracy_min': float(np.min(new_classes)),
            'new_class_accuracy_max': float(np.max(new_classes)),
        })
    
    # 統計情報をファイルに保存
    with open(os.path.join(session_dir, 'accuracy_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # テキスト形式でも保存
    with open(os.path.join(session_dir, 'accuracy_summary.txt'), 'w') as f:
        f.write(f"Session {session} Analysis Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Base Class Accuracy: {base_mean:.4f} ± {base_std:.4f}\n")
        f.write(f"Base Class Range: [{np.min(base_classes):.4f}, {np.max(base_classes):.4f}]\n")
        
        if len(new_classes) > 0:
            f.write(f"New Class Accuracy: {np.mean(new_classes):.4f} ± {np.std(new_classes):.4f}\n")
            f.write(f"New Class Range: [{np.min(new_classes):.4f}, {np.max(new_classes):.4f}]\n")
    
    print(f"Session {session} の分析完了: 全体精度 {overall_accuracy:.3f}")
    return stats

def compare_sessions(all_stats, output_dir):
    """セッション間の比較"""
    if len(all_stats) < 2:
        return
    
    print("セッション間の比較分析を実行中...")
    
    sessions = sorted(all_stats.keys())
    overall_accuracies = [all_stats[s]['overall_accuracy'] for s in sessions]
    base_accuracies = [all_stats[s]['base_class_accuracy_mean'] for s in sessions]
    
    # 1. 全体精度の推移
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, overall_accuracies, 'b-o', linewidth=2, markersize=8)
    plt.title('Overall Accuracy Across Sessions')
    plt.xlabel('Session')
    plt.ylabel('Overall Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(sessions)
    
    # 数値を点の上に表示
    for i, (s, acc) in enumerate(zip(sessions, overall_accuracies)):
        plt.text(s, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_accuracy_trend.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ベースクラス精度の推移
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, base_accuracies, 'r-s', linewidth=2, markersize=8)
    plt.title('Base Class Accuracy Across Sessions')
    plt.xlabel('Session')
    plt.ylabel('Base Class Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(sessions)
    
    # 数値を点の上に表示
    for i, (s, acc) in enumerate(zip(sessions, base_accuracies)):
        plt.text(s, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'base_accuracy_trend.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 比較サマリーの保存
    with open(os.path.join(output_dir, 'comparison_summary.txt'), 'w') as f:
        f.write("FACT Model Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Session-wise Performance:\n")
        f.write("-" * 30 + "\n")
        for session in sessions:
            stats = all_stats[session]
            f.write(f"Session {session}:\n")
            f.write(f"  Overall Accuracy: {stats['overall_accuracy']:.4f}\n")
            f.write(f"  Base Classes: {stats['base_class_accuracy_mean']:.4f} ± {stats['base_class_accuracy_std']:.4f}\n")
            
            if 'new_class_accuracy_mean' in stats:
                f.write(f"  New Classes: {stats['new_class_accuracy_mean']:.4f} ± {stats['new_class_accuracy_std']:.4f}\n")
            f.write("\n")
        
        # トレンド分析
        f.write("Trend Analysis:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Overall Accuracy: {overall_accuracies[0]:.4f} → {overall_accuracies[-1]:.4f} "
               f"(Change: {overall_accuracies[-1] - overall_accuracies[0]:+.4f})\n")
        f.write(f"Base Class Accuracy: {base_accuracies[0]:.4f} → {base_accuracies[-1]:.4f} "
               f"(Change: {base_accuracies[-1] - base_accuracies[0]:+.4f})\n")
        
        # Catastrophic Forgettingの評価
        forgetting_rate = base_accuracies[0] - base_accuracies[-1]
        f.write(f"\nCatastrophic Forgetting Rate: {forgetting_rate:.4f}\n")
        if forgetting_rate > 0.1:
            f.write("⚠️  Significant catastrophic forgetting detected!\n")
        elif forgetting_rate > 0.05:
            f.write("⚠️  Moderate catastrophic forgetting detected.\n")
        else:
            f.write("✅ Catastrophic forgetting is well controlled.\n")

def main():
    parser = argparse.ArgumentParser(description='混同行列分析スクリプト')
    parser.add_argument('-dataset', type=str, default='cifar100', 
                       choices=['cifar100', 'cub200', 'mini_imagenet', 'cicids2017_improved'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-checkpoint_dir', type=str, 
                       default='checkpoint/cicids2017_improved/fact/ft_cos-avg_cos-data_init-start_0/Epo_1-Lr_0.1000-Step_20-Gam_0.10-Bs_128-Mom_0.90-T_16.00',
                       help='チェックポイントディレクトリ')
    parser.add_argument('-output_dir', type=str, default='confusion_analysis',
                       help='出力ディレクトリ')
    parser.add_argument('-sessions', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5],
                       help='分析するセッション')
    parser.add_argument('-batch_size_base', type=int, default=128,
                       help='ベースセッションのバッチサイズ')
    parser.add_argument('-batch_size_new', type=int, default=0,
                       help='新規セッションのバッチサイズ（0の場合は全データを使用）')
    parser.add_argument('-test_batch_size', type=int, default=100,
                       help='テストのバッチサイズ')
    parser.add_argument('-num_workers', type=int, default=8,
                       help='データローダーのワーカー数')
    parser.add_argument('-temperature', type=float, default=16.0,
                       help='温度パラメータ')
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                       help='ベースモード')
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                       help='新規モード')
    
    args = parser.parse_args()
    
    # セッションリストを保存
    sessions_to_analyze = args.sessions.copy()
    
    # データセット設定
    args = set_up_datasets(args)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 各セッションを分析
    all_stats = {}
    for session in sessions_to_analyze:
        stats = analyze_session(session, args, args.checkpoint_dir, args.output_dir)
        if stats:
            all_stats[session] = stats
    
    # セッション間の比較
    if len(all_stats) > 1:
        compare_sessions(all_stats, args.output_dir)
    
    print(f"\n分析完了！結果は {args.output_dir} に保存されました。")
    print(f"分析したセッション: {list(all_stats.keys())}")

if __name__ == '__main__':
    main()
