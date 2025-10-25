#!/usr/bin/env python3
"""
Session 0のモデルを使用して全セッションを評価するスクリプト
FACTの前方互換性をテストする
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

# プロジェクトのルートディレクトリをパスに追加
sys.path.append('/home/hawk/Documents/school/test/CVPR22-Fact')

from utils import confmatrix
from models.fact.Network import MYNET
from dataloader.data_utils import set_up_datasets, get_dataloader

def evaluate_session_with_session0_model(session, args, checkpoint_path, output_dir):
    """Session 0のモデルを使用して指定されたセッションを評価"""
    print(f"Session {session} をSession 0のモデルで評価中...")
    
    # モデルのセットアップ
    model = MYNET(args, mode='ft_cos')
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.cuda()
    
    # Session 0のチェックポイントを読み込み
    if not os.path.exists(checkpoint_path):
        print(f"チェックポイントが見つかりません: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['params'])
    print(f"Session 0のモデルを読み込みました: {checkpoint_path}")
    
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
            
            # Session 0のモデルで予測
            logits = model(data)
            
            # 現在のセッションまでのクラスのみ使用
            logits = logits[:, :test_class]
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 混同行列の生成
    cm = confmatrix(all_labels, all_predictions, normalize='true')
    
    # 結果の保存
    session_dir = os.path.join(output_dir, f'session_{session}')
    os.makedirs(session_dir, exist_ok=True)
    
    # 1. 混同行列の可視化
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap='Blues', aspect='auto')
    plt.colorbar(label='Normalized Count')
    plt.title(f'Confusion Matrix - Session {session} (Evaluated with Session 0 Model)\n'
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
        'evaluated_with': 'session0_model'
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
        f.write(f"Session {session} Analysis Summary (Evaluated with Session 0 Model)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Base Class Accuracy: {base_mean:.4f} ± {base_std:.4f}\n")
        f.write(f"Base Class Range: [{np.min(base_classes):.4f}, {np.max(base_classes):.4f}]\n")
        
        if len(new_classes) > 0:
            f.write(f"New Class Accuracy: {np.mean(new_classes):.4f} ± {np.std(new_classes):.4f}\n")
            f.write(f"New Class Range: [{np.min(new_classes):.4f}, {np.max(new_classes):.4f}]\n")
        
        f.write("\nForward Compatibility Test:\n")
        if session > 0:
            if np.mean(new_classes) > 0.1:  # 10%以上の精度
                f.write("✅ Good forward compatibility for new classes\n")
            elif np.mean(new_classes) > 0.05:  # 5%以上の精度
                f.write("⚠️  Moderate forward compatibility for new classes\n")
            else:
                f.write("❌ Poor forward compatibility for new classes\n")
    
    print(f"Session {session} の評価完了: 全体精度 {overall_accuracy:.3f}")
    return stats

def compare_forward_compatibility(all_stats, output_dir):
    """前方互換性の比較分析"""
    if len(all_stats) < 2:
        return
    
    print("前方互換性の比較分析を実行中...")
    
    sessions = sorted(all_stats.keys())
    overall_accuracies = [all_stats[s]['overall_accuracy'] for s in sessions]
    base_accuracies = [all_stats[s]['base_class_accuracy_mean'] for s in sessions]
    
    # 新規クラスの精度（存在する場合）
    new_accuracies = []
    for s in sessions:
        if 'new_class_accuracy_mean' in all_stats[s]:
            new_accuracies.append(all_stats[s]['new_class_accuracy_mean'])
        else:
            new_accuracies.append(None)
    
    # 1. 全体精度の推移
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, overall_accuracies, 'b-o', linewidth=2, markersize=8, label='Overall Accuracy')
    plt.title('Forward Compatibility Test: Overall Accuracy Across Sessions\n(Evaluated with Session 0 Model)')
    plt.xlabel('Session')
    plt.ylabel('Overall Accuracy')
    plt.grid(True, alpha=0.3)
    plt.xticks(sessions)
    plt.legend()
    
    # 数値を点の上に表示
    for i, (s, acc) in enumerate(zip(sessions, overall_accuracies)):
        plt.text(s, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forward_compatibility_overall.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ベースクラス vs 新規クラスの精度比較
    plt.figure(figsize=(12, 8))
    plt.plot(sessions, base_accuracies, 'b-o', linewidth=2, markersize=8, label='Base Classes')
    
    # 新規クラスがあるセッションのみプロット
    new_sessions = [s for s, acc in zip(sessions, new_accuracies) if acc is not None]
    new_accs = [acc for acc in new_accuracies if acc is not None]
    if new_accs:
        plt.plot(new_sessions, new_accs, 'r-s', linewidth=2, markersize=8, label='New Classes')
    
    plt.title('Forward Compatibility Test: Base vs New Classes\n(Evaluated with Session 0 Model)')
    plt.xlabel('Session')
    plt.ylabel('Mean Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(sessions)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'forward_compatibility_base_vs_new.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 前方互換性サマリーの保存
    with open(os.path.join(output_dir, 'forward_compatibility_summary.txt'), 'w') as f:
        f.write("FACT Forward Compatibility Analysis\n")
        f.write("=" * 50 + "\n")
        f.write("Evaluated with Session 0 Model\n\n")
        
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
        
        # 前方互換性の評価
        f.write("Forward Compatibility Assessment:\n")
        f.write("-" * 40 + "\n")
        
        if len(new_accs) > 0:
            avg_new_accuracy = np.mean(new_accs)
            f.write(f"Average New Class Accuracy: {avg_new_accuracy:.4f}\n")
            
            if avg_new_accuracy > 0.2:
                f.write("✅ Excellent forward compatibility\n")
            elif avg_new_accuracy > 0.1:
                f.write("✅ Good forward compatibility\n")
            elif avg_new_accuracy > 0.05:
                f.write("⚠️  Moderate forward compatibility\n")
            else:
                f.write("❌ Poor forward compatibility\n")
        
        # ベースクラスの安定性
        base_stability = np.std(base_accuracies)
        f.write(f"Base Class Stability (std): {base_stability:.4f}\n")
        
        if base_stability < 0.05:
            f.write("✅ Base classes are stable across sessions\n")
        elif base_stability < 0.1:
            f.write("⚠️  Base classes show moderate variation\n")
        else:
            f.write("❌ Base classes show high variation\n")

def main():
    parser = argparse.ArgumentParser(description='Session 0モデルでの前方互換性テスト')
    parser.add_argument('-dataset', type=str, default='cifar100', 
                       choices=['cifar100', 'cub200', 'mini_imagenet'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-checkpoint_dir', type=str, 
                       default='checkpoint/cifar100/fact/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_3-Lr_0.1000Bal0.00-LossIter0-T_16.00',
                       help='チェックポイントディレクトリ')
    parser.add_argument('-output_dir', type=str, default='forward_compatibility_test',
                       help='出力ディレクトリ')
    parser.add_argument('-sessions', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       help='評価するセッション')
    parser.add_argument('-batch_size_base', type=int, default=128,
                       help='ベースセッションのバッチサイズ')
    parser.add_argument('-test_batch_size', type=int, default=100,
                       help='テストのバッチサイズ')
    parser.add_argument('-temperature', type=float, default=16.0,
                       help='温度パラメータ')
    parser.add_argument('-base_mode', type=str, default='ft_cos',
                       help='ベースモード')
    parser.add_argument('-new_mode', type=str, default='avg_cos',
                       help='新規モード')
    parser.add_argument('-batch_size_new', type=int, default=0,
                       help='新規セッションのバッチサイズ')
    parser.add_argument('-num_workers', type=int, default=8,
                       help='データローダーのワーカー数')
    
    args = parser.parse_args()
    
    # セッションリストを保存
    sessions_to_analyze = args.sessions.copy()
    
    # データセット設定
    args = set_up_datasets(args)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Session 0のチェックポイントパス
    session0_checkpoint = os.path.join(args.checkpoint_dir, 'session0_max_acc.pth')
    
    # 各セッションをSession 0のモデルで評価
    all_stats = {}
    for session in sessions_to_analyze:
        stats = evaluate_session_with_session0_model(session, args, session0_checkpoint, args.output_dir)
        if stats:
            all_stats[session] = stats
    
    # 前方互換性の比較
    if len(all_stats) > 1:
        compare_forward_compatibility(all_stats, args.output_dir)
    
    print(f"\n前方互換性テスト完了！結果は {args.output_dir} に保存されました。")
    print(f"評価したセッション: {list(all_stats.keys())}")

if __name__ == '__main__':
    main()
