#!/usr/bin/env python3
"""
既存のテスト結果から混同行列を生成・分析するスクリプト
保存されたモデルを使用して詳細な分析を行う
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from utils import confmatrix
from models.fact.Network import MYNET
from dataloader.data_utils import set_up_datasets, get_dataloader

class ExistingResultsAnalyzer:
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
            return True
        else:
            print(f"チェックポイントが見つかりません: {checkpoint_path}")
            return False
    
    def analyze_session_with_model(self, session, checkpoint_path, save_dir):
        """モデルを使用してセッションを分析"""
        if not self.load_model_checkpoint(checkpoint_path):
            return None
            
        print("=" * 100)
        print(f"Session {session} を分析中...")
        
        # データローダーの準備
        train_set, trainloader, testloader = get_dataloader(self.args, session)
        
        # テストクラス数の計算
        test_class = self.args.base_class + session * self.args.way
        
        # モデルを評価モードに設定
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        
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
                all_logits.extend(logits.cpu().numpy())
        
        # 混同行列の生成
        cm = confmatrix(all_labels, all_predictions, normalize='true')
        
        # 詳細分析
        analysis_results = self.detailed_analysis(cm, all_labels, all_predictions, 
                                                all_logits, session, test_class)
        
        # 結果の保存
        self.save_analysis_results(analysis_results, session, save_dir)
        
        return analysis_results
    
    def detailed_analysis(self, cm, true_labels, predictions, logits, session, test_class):
        """詳細な分析を実行"""
        results = {
            'session': session,
            'test_class_count': test_class,
            'confusion_matrix': cm,
            'true_labels': true_labels,
            'predictions': predictions,
            'logits': logits
        }
        
        # 全体の精度
        overall_accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        results['overall_accuracy'] = overall_accuracy
        
        # クラス別精度
        per_class_acc = cm.diagonal()
        results['per_class_accuracy'] = per_class_acc
        
        # ベースクラスと新規クラスの分析
        base_classes = per_class_acc[:self.args.base_class]
        results['base_class_accuracy'] = {
            'mean': np.mean(base_classes),
            'std': np.std(base_classes),
            'min': np.min(base_classes),
            'max': np.max(base_classes),
            'values': base_classes
        }
        
        if session > 0:
            new_classes = per_class_acc[self.args.base_class:]
            results['new_class_accuracy'] = {
                'mean': np.mean(new_classes),
                'std': np.std(new_classes),
                'min': np.min(new_classes),
                'max': np.max(new_classes),
                'values': new_classes
            }
        
        # 混同パターンの分析
        confusion_patterns = self.analyze_confusion_patterns(cm, session)
        results['confusion_patterns'] = confusion_patterns
        
        # 信頼度の分析
        confidence_analysis = self.analyze_confidence(logits, true_labels, predictions)
        results['confidence_analysis'] = confidence_analysis
        
        return results
    
    def analyze_confusion_patterns(self, cm, session):
        """混同パターンの分析"""
        patterns = {}
        
        # 最も混同しやすいクラスペア
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)  # 対角成分を0に
        
        # 上位10の混同パターン
        flat_indices = np.argsort(cm_copy.flatten())[-10:]
        top_confusions = []
        
        for idx in flat_indices:
            true_class, pred_class = np.unravel_index(idx, cm_copy.shape)
            confusion_rate = cm_copy[true_class, pred_class]
            if confusion_rate > 0.01:  # 1%以上の混同
                top_confusions.append({
                    'true_class': int(true_class),
                    'pred_class': int(pred_class),
                    'confusion_rate': float(confusion_rate)
                })
        
        patterns['top_confusions'] = top_confusions
        
        # クラス別の混同度
        class_confusion_rates = np.sum(cm_copy, axis=1)
        patterns['class_confusion_rates'] = class_confusion_rates.tolist()
        
        return patterns
    
    def analyze_confidence(self, logits, true_labels, predictions):
        """信頼度の分析"""
        logits_array = np.array(logits)
        true_labels_array = np.array(true_labels)
        predictions_array = np.array(predictions)
        
        # 予測の信頼度（最大確率）
        max_probs = np.max(torch.softmax(torch.tensor(logits_array), dim=1).numpy(), axis=1)
        
        # 正解・不正解別の信頼度
        correct_mask = true_labels_array == predictions_array
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]
        
        return {
            'mean_confidence_correct': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0,
            'mean_confidence_incorrect': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0,
            'confidence_std_correct': float(np.std(correct_confidences)) if len(correct_confidences) > 0 else 0,
            'confidence_std_incorrect': float(np.std(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0,
        }
    
    def save_analysis_results(self, results, session, save_dir):
        """分析結果を保存"""
        session_dir = os.path.join(save_dir, f'session_{session}')
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. 混同行列の可視化
        self.visualize_confusion_matrix(results['confusion_matrix'], session, session_dir)
        
        # 2. クラス別精度の可視化
        self.visualize_per_class_accuracy(results, session, session_dir)
        
        # 3. 混同パターンの可視化
        self.visualize_confusion_patterns(results['confusion_patterns'], session, session_dir)
        
        # 4. 信頼度の可視化
        self.visualize_confidence(results['confidence_analysis'], session, session_dir)
        
        # 5. 数値結果の保存
        self.save_numerical_results(results, session, session_dir)
    
    def visualize_confusion_matrix(self, cm, session, save_dir):
        """混同行列の可視化"""
        # 基本の混同行列
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='.3f', 
                   cbar_kws={'label': 'Normalized Count'})
        plt.title(f'Confusion Matrix - Session {session}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 詳細な混同行列（数値付き、小さい場合のみ）
        if len(cm) <= 20:
            plt.figure(figsize=(15, 12))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', 
                       cbar_kws={'label': 'Normalized Count'},
                       annot_kws={'size': 10})
            plt.title(f'Detailed Confusion Matrix - Session {session}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'detailed_confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_per_class_accuracy(self, results, session, save_dir):
        """クラス別精度の可視化"""
        per_class_acc = results['per_class_accuracy']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # 全体のクラス別精度
        ax1.plot(range(len(per_class_acc)), per_class_acc, 'b-', marker='o', markersize=2)
        ax1.axvline(x=self.args.base_class-0.5, color='r', linestyle='--', alpha=0.7, label='Base/New Boundary')
        ax1.set_title(f'Per-Class Accuracy - Session {session}')
        ax1.set_xlabel('Class Index')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # ベースクラスと新規クラスの比較
        base_acc = results['base_class_accuracy']['values']
        ax2.plot(range(len(base_acc)), base_acc, 'b-', marker='o', markersize=3, label='Base Classes')
        
        if 'new_class_accuracy' in results:
            new_acc = results['new_class_accuracy']['values']
            new_indices = range(self.args.base_class, self.args.base_class + len(new_acc))
            ax2.plot(new_indices, new_acc, 'r-', marker='s', markersize=3, label='New Classes')
        
        ax2.set_title('Base vs New Classes Accuracy')
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_confusion_patterns(self, patterns, session, save_dir):
        """混同パターンの可視化"""
        if not patterns['top_confusions']:
            return
            
        # 上位混同パターンの可視化
        confusions = patterns['top_confusions'][:10]  # 上位10
        true_classes = [c['true_class'] for c in confusions]
        pred_classes = [c['pred_class'] for c in confusions]
        rates = [c['confusion_rate'] for c in confusions]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(confusions)), rates, color='red', alpha=0.7)
        plt.title(f'Top Confusion Patterns - Session {session}')
        plt.xlabel('Confusion Pattern')
        plt.ylabel('Confusion Rate')
        plt.xticks(range(len(confusions)), 
                  [f'{tc}→{pc}' for tc, pc in zip(true_classes, pred_classes)], 
                  rotation=45)
        
        # 数値をバーの上に表示
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{rate:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_patterns.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_confidence(self, confidence_analysis, session, save_dir):
        """信頼度の可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 正解・不正解別の信頼度比較
        categories = ['Correct', 'Incorrect']
        means = [confidence_analysis['mean_confidence_correct'], 
                confidence_analysis['mean_confidence_incorrect']]
        stds = [confidence_analysis['confidence_std_correct'], 
               confidence_analysis['confidence_std_incorrect']]
        
        bars = ax1.bar(categories, means, yerr=stds, capsize=5, 
                      color=['green', 'red'], alpha=0.7)
        ax1.set_title(f'Confidence Analysis - Session {session}')
        ax1.set_ylabel('Mean Confidence')
        ax1.set_ylim(0, 1)
        
        # 数値をバーの上に表示
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean:.3f}', ha='center', va='bottom')
        
        # 信頼度の分布（簡易版）
        ax2.text(0.5, 0.5, f'Correct: {means[0]:.3f} ± {stds[0]:.3f}\n'
                          f'Incorrect: {means[1]:.3f} ± {stds[1]:.3f}', 
                ha='center', va='center', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax2.set_title('Confidence Statistics')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_numerical_results(self, results, session, save_dir):
        """数値結果の保存"""
        # JSON形式で保存
        json_results = {
            'session': results['session'],
            'overall_accuracy': results['overall_accuracy'],
            'base_class_accuracy': results['base_class_accuracy'],
            'confusion_patterns': results['confusion_patterns'],
            'confidence_analysis': results['confidence_analysis']
        }
        
        if 'new_class_accuracy' in results:
            json_results['new_class_accuracy'] = results['new_class_accuracy']
        
        with open(os.path.join(save_dir, 'analysis_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # テキスト形式でも保存
        with open(os.path.join(save_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(f"Analysis Summary - Session {session}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.4f}\n")
            f.write(f"Base Class Accuracy: {results['base_class_accuracy']['mean']:.4f} ± {results['base_class_accuracy']['std']:.4f}\n")
            
            if 'new_class_accuracy' in results:
                f.write(f"New Class Accuracy: {results['new_class_accuracy']['mean']:.4f} ± {results['new_class_accuracy']['std']:.4f}\n")
            
            f.write("\nTop Confusion Patterns:\n")
            for i, conf in enumerate(results['confusion_patterns']['top_confusions'][:5]):
                f.write(f"{i+1}. Class {conf['true_class']} → Class {conf['pred_class']}: {conf['confusion_rate']:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description='既存結果の混同行列分析')
    parser.add_argument('-dataset', type=str, default='cifar100', 
                       choices=['cifar100', 'cub200', 'mini_imagenet'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-checkpoint_dir', type=str, 
                       default='checkpoint/cifar100/fact/ft_cos-avg_cos-data_init-start_0/Cosine-Epo_3-Lr_0.1000Bal0.00-LossIter0-T_16.00',
                       help='チェックポイントディレクトリ')
    parser.add_argument('-output_dir', type=str, default='detailed_analysis',
                       help='出力ディレクトリ')
    parser.add_argument('-sessions', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                       help='分析するセッション')
    
    args = parser.parse_args()
    
    # データセット設定
    args = set_up_datasets(args)
    
    # アナライザーの初期化
    analyzer = ExistingResultsAnalyzer(args)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 各セッションを分析
    all_results = {}
    checkpoint_path = os.path.join(args.checkpoint_dir, 'optimizer_best.pth')
    for session in args.sessions:
        results = analyzer.analyze_session_with_model(session, checkpoint_path, args.output_dir)
        if results:
            all_results[session] = results
    
    # セッション間の比較
    if len(all_results) > 1:
        analyzer.compare_all_sessions(all_results, args.output_dir)
    
    print(f"詳細分析完了！結果は {args.output_dir} に保存されました。")

    def compare_all_sessions(self, all_results, save_dir):
        """全セッションの比較分析"""
        print("全セッションの比較分析を実行中...")
        
        # 1. 全体精度の推移
        sessions = sorted(all_results.keys())
        overall_accuracies = [all_results[s]['overall_accuracy'] for s in sessions]
        
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
        plt.savefig(os.path.join(save_dir, 'overall_accuracy_trend.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ベースクラス vs 新規クラスの精度比較
        base_accuracies = []
        new_accuracies = []
        
        for session in sessions:
            base_accuracies.append(all_results[session]['base_class_accuracy']['mean'])
            if 'new_class_accuracy' in all_results[session]:
                new_accuracies.append(all_results[session]['new_class_accuracy']['mean'])
            else:
                new_accuracies.append(None)
        
        plt.figure(figsize=(12, 8))
        plt.plot(sessions, base_accuracies, 'b-o', linewidth=2, markersize=8, label='Base Classes')
        
        # 新規クラスがあるセッションのみプロット
        new_sessions = [s for s, acc in zip(sessions, new_accuracies) if acc is not None]
        new_accs = [acc for acc in new_accuracies if acc is not None]
        if new_accs:
            plt.plot(new_sessions, new_accs, 'r-s', linewidth=2, markersize=8, label='New Classes')
        
        plt.title('Base vs New Classes Accuracy Across Sessions')
        plt.xlabel('Session')
        plt.ylabel('Mean Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sessions)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'base_vs_new_accuracy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混同パターンの変化
        self.analyze_confusion_evolution(all_results, save_dir)
        
        # 4. 統計サマリーの保存
        self.save_comparison_summary(all_results, save_dir)
    
        def analyze_confusion_evolution(self, all_results, save_dir):
            """混同パターンの変化を分析"""
            sessions = sorted(all_results.keys())
            
            # 各セッションの混同率を収集
            confusion_rates = {}
            for session in sessions:
                patterns = all_results[session]['confusion_patterns']
                for conf in patterns['top_confusions']:
                    key = f"{conf['true_class']}→{conf['pred_class']}"
                    if key not in confusion_rates:
                        confusion_rates[key] = []
                    confusion_rates[key].append(conf['confusion_rate'])
            
            # 最も混同しやすいパターンを特定
            avg_confusion_rates = {k: np.mean(v) for k, v in confusion_rates.items()}
            top_confusions = sorted(avg_confusion_rates.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # 可視化
            if top_confusions:
                plt.figure(figsize=(15, 8))
                confusion_names = [k for k, v in top_confusions]
                avg_rates = [v for k, v in top_confusions]
                
                bars = plt.bar(range(len(confusion_names)), avg_rates, color='red', alpha=0.7)
                plt.title('Top Confusion Patterns (Average Across Sessions)')
                plt.xlabel('Confusion Pattern')
                plt.ylabel('Average Confusion Rate')
                plt.xticks(range(len(confusion_names)), confusion_names, rotation=45)
                
                # 数値をバーの上に表示
                for bar, rate in zip(bars, avg_rates):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            f'{rate:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, 'top_confusion_patterns.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        def save_comparison_summary(self, all_results, save_dir):
        # 比較サマリーを保存
            sessions = sorted(all_results.keys())
            
            with open(os.path.join(save_dir, 'comparison_summary.txt'), 'w') as f:
                f.write("FACT Model Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Session-wise Performance:\n")
                f.write("-" * 30 + "\n")
                for session in sessions:
                    results = all_results[session]
                    f.write(f"Session {session}:\n")
                    f.write(f"  Overall Accuracy: {results['overall_accuracy']:.4f}\n")
                    f.write(f"  Base Classes: {results['base_class_accuracy']['mean']:.4f} ± {results['base_class_accuracy']['std']:.4f}\n")
                    
                    if 'new_class_accuracy' in results:
                        f.write(f"  New Classes: {results['new_class_accuracy']['mean']:.4f} ± {results['new_class_accuracy']['std']:.4f}\n")
                    f.write("\n")
                
                # トレンド分析
                overall_accs = [all_results[s]['overall_accuracy'] for s in sessions]
                base_accs = [all_results[s]['base_class_accuracy']['mean'] for s in sessions]
                
                f.write("Trend Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Overall Accuracy: {overall_accs[0]:.4f} → {overall_accs[-1]:.4f} "
                    f"(Change: {overall_accs[-1] - overall_accs[0]:+.4f})\n")
                f.write(f"Base Class Accuracy: {base_accs[0]:.4f} → {base_accs[-1]:.4f} "
                    f"(Change: {base_accs[-1] - base_accs[0]:+.4f})\n")
                
                # Catastrophic Forgettingの評価
                forgetting_rate = base_accs[0] - base_accs[-1]
                f.write(f"\nCatastrophic Forgetting Rate: {forgetting_rate:.4f}\n")
                if forgetting_rate > 0.1:
                    f.write("⚠️  Significant catastrophic forgetting detected!\n")
                elif forgetting_rate > 0.05:
                    f.write("⚠️  Moderate catastrophic forgetting detected.\n")
                else:
                    f.write("✅ Catastrophic forgetting is well controlled.\n")

if __name__ == '__main__':
    main()
