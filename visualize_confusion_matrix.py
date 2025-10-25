#!/usr/bin/env python3
"""
混同行列可視化スクリプト
FACTのテスト結果から混同行列を生成し、詳細な分析を行う
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import argparse
from models.fact.Network import MYNET
from dataloader.data_utils import set_up_datasets, get_dataloader


class ConfusionMatrixAnalyzer:
    def __init__(self, args):
        self.args = args
        self.setup_model()
        
    def setup_model(self):
        """モデルのセットアップ"""
        self.model = MYNET(self.args, mode='ft_cos')
        self.model = torch.nn.DataParallel(self.model, device_ids=range(torch.cuda.device_count()))
        self.model = self.model.cuda()
        
    def load_model_checkpoint(self, checkpoint_path):
        """モデルのチェックポイントを読み込み"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['params'])
            print(f"モデルを読み込みました: {checkpoint_path}")
        else:
            print(f"チェックポイントが見つかりません: {checkpoint_path}")
            
    def generate_confusion_matrix(self, session, save_dir=None):
        """指定されたセッションの混同行列を生成"""
        print(f"Session {session} の混同行列を生成中...")
        
        # データローダーの準備
        train_set, trainloader, testloader = get_dataloader(self.args, session)
        
        # テストクラス数の計算
        test_class = self.args.base_class + session * self.args.way
        
        # モデルを評価モードに設定
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, labels = [_.cuda() for _ in batch]
                
                # 予測の生成
                if session == 0:
                    # ベースセッション
                    logits = self.model(data)
                else:
                    # インクリメンタルセッション
                    logits = self.model.module.forpass_fc(data)
                
                # 現在のセッションまでのクラスのみ使用
                logits = logits[:, :test_class]
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 混同行列の生成
        cm = confusion_matrix(all_labels, all_predictions, normalize='true')
        
        # 結果の保存
        if save_dir:
            self.save_confusion_matrix(cm, session, save_dir, all_labels, all_predictions)
            
        return cm, all_labels, all_predictions
    
    def save_confusion_matrix(self, cm, session, save_dir, true_labels, predictions):
        """混同行列を可視化して保存"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. 基本の混同行列
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='.3f', 
                   cbar_kws={'label': 'Normalized Count'})
        plt.title(f'Confusion Matrix - Session {session}\n'
                 f'Classes: {self.args.base_class + session * self.args.way}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 軸の設定
        num_classes = len(cm)
        if num_classes <= 100:
            step = 20
            ticks = list(range(0, num_classes, step))
            tick_labels = [str(i) for i in ticks]
        else:
            step = 50
            ticks = list(range(0, num_classes, step))
            tick_labels = [str(i) for i in ticks]
            
        plt.xticks(ticks, tick_labels)
        plt.yticks(ticks, tick_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_session_{session}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 詳細な混同行列（数値付き）
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', 
                   cbar_kws={'label': 'Normalized Count'},
                   annot_kws={'size': 8})
        plt.title(f'Detailed Confusion Matrix - Session {session}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'detailed_confusion_matrix_session_{session}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. クラス別精度の分析
        self.analyze_per_class_accuracy(cm, session, save_dir)
        
        # 4. 分類レポートの保存
        self.save_classification_report(true_labels, predictions, session, save_dir)
        
    def analyze_per_class_accuracy(self, cm, session, save_dir):
        """クラス別精度の分析"""
        per_class_acc = cm.diagonal()
        
        # ベースクラスと新規クラスの分離
        base_classes = per_class_acc[:self.args.base_class]
        new_classes = per_class_acc[self.args.base_class:] if session > 0 else []
        
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
            ax2.plot(range(self.args.base_class, self.args.base_class + len(new_classes)), 
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
        plt.savefig(os.path.join(save_dir, f'per_class_accuracy_session_{session}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 統計情報の保存
        stats = {
            'session': session,
            'base_class_accuracy_mean': np.mean(base_classes),
            'base_class_accuracy_std': np.std(base_classes),
            'base_class_accuracy_min': np.min(base_classes),
            'base_class_accuracy_max': np.max(base_classes),
        }
        
        if len(new_classes) > 0:
            stats.update({
                'new_class_accuracy_mean': np.mean(new_classes),
                'new_class_accuracy_std': np.std(new_classes),
                'new_class_accuracy_min': np.min(new_classes),
                'new_class_accuracy_max': np.max(new_classes),
            })
        
        # 統計情報をファイルに保存
        with open(os.path.join(save_dir, f'accuracy_stats_session_{session}.txt'), 'w') as f:
            for key, value in stats.items():
                f.write(f'{key}: {value:.4f}\n')
    
    def save_classification_report(self, true_labels, predictions, session, save_dir):
        """分類レポートの保存"""
        report = classification_report(
            true_labels, predictions, 
            target_names=[f'Class_{i}' for i in range(len(set(true_labels)))],
            output_dict=True
        )
        
        # テキスト形式で保存
        with open(os.path.join(save_dir, f'classification_report_session_{session}.txt'), 'w') as f:
            f.write(f"Classification Report - Session {session}\n")
            f.write("=" * 50 + "\n")
            f.write(report)
    
    def compare_sessions(self, sessions, save_dir):
        """複数セッションの比較"""
        print("複数セッションの比較分析を実行中...")
        
        session_accuracies = []
        session_names = []
        
        for session in sessions:
            cm, _, _ = self.generate_confusion_matrix(session)
            per_class_acc = cm.diagonal()
            session_accuracies.append(per_class_acc)
            session_names.append(f'Session {session}')
        
        # セッション別精度の比較
        plt.figure(figsize=(12, 8))
        for i, (acc, name) in enumerate(zip(session_accuracies, session_names)):
            plt.plot(range(len(acc)), acc, marker='o', markersize=2, 
                    label=name, alpha=0.7)
        
        plt.title('Per-Class Accuracy Comparison Across Sessions')
        plt.xlabel('Class Index')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'session_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 平均精度の比較
        mean_accuracies = [np.mean(acc) for acc in session_accuracies]
        plt.figure(figsize=(10, 6))
        plt.bar(session_names, mean_accuracies, color='skyblue', alpha=0.7)
        plt.title('Mean Accuracy Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('Mean Accuracy')
        plt.xticks(rotation=45)
        
        # 数値をバーの上に表示
        for i, v in enumerate(mean_accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mean_accuracy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='混同行列可視化スクリプト')
    parser.add_argument('-dataset', type=str, default='cifar100', 
                       choices=['cifar100', 'cub200', 'mini_imagenet'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-session', type=int, default=None, 
                       help='特定のセッションを分析（指定しない場合は全セッション）')
    parser.add_argument('-checkpoint_dir', type=str, 
                       default='checkpoint/cifar100/fact/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_3-Lr_0.1000Bal0.00-LossIter0-T_16.00',
                       help='チェックポイントディレクトリ')
    parser.add_argument('-output_dir', type=str, default='confusion_matrix_analysis',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # データセット設定
    args = set_up_datasets(args)
    
    # アナライザーの初期化
    analyzer = ConfusionMatrixAnalyzer(args)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.session is not None:
        # 特定のセッションのみ分析
        checkpoint_path = os.path.join(args.checkpoint_dir, f'session{args.session}_max_acc.pth')
        analyzer.load_model_checkpoint(checkpoint_path)
        analyzer.generate_confusion_matrix(args.session, args.output_dir)
    else:
        # 全セッションを分析
        sessions_to_analyze = list(range(args.sessions))
        
        for session in sessions_to_analyze:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'session{session}_max_acc.pth')
            if os.path.exists(checkpoint_path):
                analyzer.load_model_checkpoint(checkpoint_path)
                analyzer.generate_confusion_matrix(session, args.output_dir)
            else:
                print(f"Session {session} のチェックポイントが見つかりません: {checkpoint_path}")
        
        # セッション間の比較
        available_sessions = [s for s in sessions_to_analyze 
                            if os.path.exists(os.path.join(args.checkpoint_dir, f'session{s}_max_acc.pth'))]
        if len(available_sessions) > 1:
            analyzer.compare_sessions(available_sessions, args.output_dir)
    
    print(f"分析完了！結果は {args.output_dir} に保存されました。")

if __name__ == '__main__':
    main()
