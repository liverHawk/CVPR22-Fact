#!/usr/bin/env python3
"""
BaseとFACTモデルの同時比較スクリプト（修正版）
各セッションの適切なチェックポイントを使用して比較
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# プロジェクトのルートディレクトリをパスに追加
sys.path.append('/home/hawk/Documents/school/test/CVPR22-Fact')

from utils import confmatrix, count_acc, count_acc_topk
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

class BaseFactComparatorFixed:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def setup_models(self):
        """BaseとFACTモデルのセットアップ"""
        print("モデルをセットアップ中...")
        
        # FACTモデル
        self.fact_model = FACT_MYNET(self.args, mode='ft_cos')
        self.fact_model = torch.nn.DataParallel(self.fact_model, device_ids=range(torch.cuda.device_count()))
        self.fact_model = self.fact_model.to(self.device)
        
        # Baseモデル
        self.base_model = BASE_MYNET(self.args, mode='ft_cos')
        self.base_model = torch.nn.DataParallel(self.base_model, device_ids=range(torch.cuda.device_count()))
        self.base_model = self.base_model.to(self.device)
        
        print(f"デバイス: {self.device}")
        
    def load_model_checkpoints_for_session(self, fact_checkpoint_dir, base_checkpoint_dir, session):
        """指定されたセッションの両モデルのチェックポイントを読み込み"""
        print(f"Session {session} のチェックポイントを読み込み中...")
        
        # FACTモデルの読み込み
        fact_checkpoint_path = os.path.join(fact_checkpoint_dir, f'session{session}_max_acc.pth')
        if os.path.exists(fact_checkpoint_path):
            fact_checkpoint = torch.load(fact_checkpoint_path)
            self.fact_model.load_state_dict(fact_checkpoint['params'], strict=False)
            print(f"FACTモデルを読み込みました: {fact_checkpoint_path}")
        else:
            print(f"FACTチェックポイントが見つかりません: {fact_checkpoint_path}")
            return False
            
        # Baseモデルの読み込み
        base_checkpoint_path = os.path.join(base_checkpoint_dir, f'session{session}_max_acc.pth')
        if os.path.exists(base_checkpoint_path):
            base_checkpoint = torch.load(base_checkpoint_path)
            self.base_model.load_state_dict(base_checkpoint['params'], strict=False)
            print(f"Baseモデルを読み込みました: {base_checkpoint_path}")
        else:
            print(f"Baseチェックポイントが見つかりません: {base_checkpoint_path}")
            return False
            
        return True
    
    def evaluate_session(self, session, fact_checkpoint_dir, base_checkpoint_dir, output_dir):
        """指定されたセッションで両モデルを評価"""
        print("=" * 100)
        print(f"Session {session} の比較評価を実行中...")
        
        # モデルをセットアップ
        self.setup_models()
        
        # セッション固有のチェックポイントを読み込み
        if not self.load_model_checkpoints_for_session(fact_checkpoint_dir, base_checkpoint_dir, session):
            print(f"Session {session} のチェックポイント読み込みに失敗しました")
            return None
        
        # データローダーの準備
        train_set, trainloader, testloader = get_dataloader(self.args, session)
        
        # テストクラス数の計算
        test_class = self.args.base_class + session * self.args.way
        
        # 両モデルを評価モードに設定
        self.fact_model.eval()
        self.base_model.eval()
        
        # 結果を格納する辞書
        results = {
            'session': session,
            'test_class_count': test_class,
            'fact': {'predictions': [], 'labels': [], 'logits': []},
            'base': {'predictions': [], 'labels': [], 'logits': []}
        }
        
        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, labels = [_.to(self.device) for _ in batch]
                
                # FACTモデルの予測
                if session == 0:
                    fact_logits = self.fact_model(data)
                else:
                    fact_logits = self.fact_model.module.forpass_fc(data)
                fact_logits = fact_logits[:, :test_class]
                fact_predictions = torch.argmax(fact_logits, dim=1)
                
                # Baseモデルの予測
                base_logits = self.base_model(data)
                base_logits = base_logits[:, :test_class]
                base_predictions = torch.argmax(base_logits, dim=1)
                
                # 結果を保存
                results['fact']['predictions'].extend(fact_predictions.cpu().numpy())
                results['fact']['labels'].extend(labels.cpu().numpy())
                results['fact']['logits'].extend(fact_logits.cpu().numpy())
                
                results['base']['predictions'].extend(base_predictions.cpu().numpy())
                results['base']['labels'].extend(labels.cpu().numpy())
                results['base']['logits'].extend(base_logits.cpu().numpy())
        
        # 詳細分析を実行
        analysis_results = self.detailed_comparison_analysis(results, session, test_class)
        
        # 結果を保存
        self.save_comparison_results(analysis_results, session, output_dir)
        
        return analysis_results
    
    def detailed_comparison_analysis(self, results, session, test_class):
        """詳細な比較分析を実行"""
        analysis = {
            'session': session,
            'test_class_count': test_class,
            'fact': {},
            'base': {},
            'comparison': {}
        }
        
        # 各モデルの分析
        for model_name in ['fact', 'base']:
            predictions = np.array(results[model_name]['predictions'])
            labels = np.array(results[model_name]['labels'])
            logits = np.array(results[model_name]['logits'])
            
            # 基本統計
            overall_accuracy = np.mean(predictions == labels)
            
            # 混同行列
            cm = confusion_matrix(labels, predictions, normalize='true')
            
            # クラス別精度
            per_class_acc = cm.diagonal()
            
            # ベースクラスと新規クラスの分析
            base_classes = per_class_acc[:self.args.base_class]
            new_classes = per_class_acc[self.args.base_class:] if session > 0 else []
            
            analysis[model_name] = {
                'overall_accuracy': overall_accuracy,
                'confusion_matrix': cm,
                'per_class_accuracy': per_class_acc,
                'base_class_accuracy': {
                    'mean': np.mean(base_classes),
                    'std': np.std(base_classes),
                    'min': np.min(base_classes),
                    'max': np.max(base_classes),
                    'values': base_classes
                }
            }
            
            if len(new_classes) > 0:
                analysis[model_name]['new_class_accuracy'] = {
                    'mean': np.mean(new_classes),
                    'std': np.std(new_classes),
                    'min': np.min(new_classes),
                    'max': np.max(new_classes),
                    'values': new_classes
                }
        
        # 比較分析
        fact_acc = analysis['fact']['overall_accuracy']
        base_acc = analysis['base']['overall_accuracy']
        
        analysis['comparison'] = {
            'overall_accuracy_diff': fact_acc - base_acc,
            'overall_accuracy_improvement': (fact_acc - base_acc) / base_acc * 100 if base_acc > 0 else 0,
            'base_class_accuracy_diff': analysis['fact']['base_class_accuracy']['mean'] - analysis['base']['base_class_accuracy']['mean'],
            'winner': 'FACT' if fact_acc > base_acc else 'Base'
        }
        
        if session > 0 and 'new_class_accuracy' in analysis['fact']:
            fact_new_acc = analysis['fact']['new_class_accuracy']['mean']
            base_new_acc = analysis['base']['new_class_accuracy']['mean']
            analysis['comparison']['new_class_accuracy_diff'] = fact_new_acc - base_new_acc
        
        return analysis
    
    def save_comparison_results(self, analysis, session, output_dir):
        """比較結果を保存"""
        session_dir = os.path.join(output_dir, f'session_{session}')
        os.makedirs(session_dir, exist_ok=True)
        
        # 1. 混同行列の比較
        self.visualize_confusion_matrix_comparison(analysis, session, session_dir)
        
        # 2. 精度の比較
        self.visualize_accuracy_comparison(analysis, session, session_dir)
        
        # 3. クラス別精度の比較
        self.visualize_per_class_comparison(analysis, session, session_dir)
        
        # 4. 統計情報の保存
        self.save_comparison_statistics(analysis, session, session_dir)
        
        print(f"Session {session} の比較結果を保存しました: {session_dir}")
    
    def visualize_confusion_matrix_comparison(self, analysis, session, save_dir):
        """混同行列の比較可視化"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # FACTモデルの混同行列
        cm_fact = analysis['fact']['confusion_matrix']
        im1 = ax1.imshow(cm_fact, cmap='Blues', aspect='auto')
        ax1.set_title(f'FACT Model - Session {session}\nAccuracy: {analysis["fact"]["overall_accuracy"]:.3f}')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        plt.colorbar(im1, ax=ax1, label='Normalized Count')
        
        # Baseモデルの混同行列
        cm_base = analysis['base']['confusion_matrix']
        im2 = ax2.imshow(cm_base, cmap='Reds', aspect='auto')
        ax2.set_title(f'Base Model - Session {session}\nAccuracy: {analysis["base"]["overall_accuracy"]:.3f}')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        plt.colorbar(im2, ax=ax2, label='Normalized Count')
        
        # クラス名の設定
        if self.args.dataset == 'cicids2017_improved':
            class_labels = [CICIDS2017_CLASS_NAMES.get(i, f'Class {i}') for i in range(analysis['test_class_count'])]
            for ax in [ax1, ax2]:
                ax.set_xticks(range(analysis['test_class_count']))
                ax.set_xticklabels(class_labels, rotation=45, ha='right')
                ax.set_yticks(range(analysis['test_class_count']))
                ax.set_yticklabels(class_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_accuracy_comparison(self, analysis, session, save_dir):
        """精度の比較可視化"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 全体精度の比較
        models = ['FACT', 'Base']
        accuracies = [analysis['fact']['overall_accuracy'], analysis['base']['overall_accuracy']]
        colors = ['blue', 'red']
        
        bars = ax1.bar(models, accuracies, color=colors, alpha=0.7)
        ax1.set_title(f'Overall Accuracy Comparison - Session {session}')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # 数値をバーの上に表示
        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 改善率を表示
        improvement = analysis['comparison']['overall_accuracy_improvement']
        ax1.text(0.5, 0.8, f'Improvement: {improvement:+.2f}%', 
                ha='center', va='center', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if improvement > 0 else "lightcoral"))
        
        # 2. ベースクラス精度の比較
        base_acc_fact = analysis['fact']['base_class_accuracy']['mean']
        base_acc_base = analysis['base']['base_class_accuracy']['mean']
        
        bars = ax2.bar(models, [base_acc_fact, base_acc_base], color=colors, alpha=0.7)
        ax2.set_title(f'Base Classes Accuracy Comparison - Session {session}')
        ax2.set_ylabel('Base Classes Accuracy')
        ax2.set_ylim(0, 1)
        
        for bar, acc in zip(bars, [base_acc_fact, base_acc_base]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 3. 新規クラス精度の比較（存在する場合）
        if 'new_class_accuracy' in analysis['fact']:
            new_acc_fact = analysis['fact']['new_class_accuracy']['mean']
            new_acc_base = analysis['base']['new_class_accuracy']['mean']
            
            bars = ax3.bar(models, [new_acc_fact, new_acc_base], color=colors, alpha=0.7)
            ax3.set_title(f'New Classes Accuracy Comparison - Session {session}')
            ax3.set_ylabel('New Classes Accuracy')
            ax3.set_ylim(0, 1)
            
            for bar, acc in zip(bars, [new_acc_fact, new_acc_base]):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No new classes in base session', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('New Classes Accuracy Comparison')
        
        # 4. 精度差の可視化
        overall_diff = analysis['comparison']['overall_accuracy_diff']
        base_diff = analysis['comparison']['base_class_accuracy_diff']
        
        diffs = [overall_diff, base_diff]
        diff_labels = ['Overall', 'Base Classes']
        diff_colors = ['green' if d > 0 else 'red' for d in diffs]
        
        bars = ax4.bar(diff_labels, diffs, color=diff_colors, alpha=0.7)
        ax4.set_title(f'Accuracy Difference (FACT - Base) - Session {session}')
        ax4.set_ylabel('Accuracy Difference')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, diff in zip(bars, diffs):
            ax4.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.01 if diff > 0 else -0.01), 
                    f'{diff:+.3f}', ha='center', va='bottom' if diff > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_per_class_comparison(self, analysis, session, save_dir):
        """クラス別精度の比較可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 1. 全クラスの精度比較
        fact_per_class = analysis['fact']['per_class_accuracy']
        base_per_class = analysis['base']['per_class_accuracy']
        
        x = range(len(fact_per_class))
        ax1.plot(x, fact_per_class, 'b-o', label='FACT', markersize=4, linewidth=2)
        ax1.plot(x, base_per_class, 'r-s', label='Base', markersize=4, linewidth=2)
        
        # ベースクラスと新規クラスの境界線
        ax1.axvline(x=self.args.base_class-0.5, color='gray', linestyle='--', alpha=0.7, label='Base/New Boundary')
        
        ax1.set_title(f'Per-Class Accuracy Comparison - Session {session}')
        ax1.set_xlabel('Class Index')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. 精度差の可視化
        accuracy_diff = fact_per_class - base_per_class
        colors = ['green' if d > 0 else 'red' for d in accuracy_diff]
        
        bars = ax2.bar(x, accuracy_diff, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=self.args.base_class-0.5, color='gray', linestyle='--', alpha=0.7)
        
        ax2.set_title(f'Per-Class Accuracy Difference (FACT - Base) - Session {session}')
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Accuracy Difference')
        ax2.grid(True, alpha=0.3)
        
        # 数値をバーの上に表示（重要なもののみ）
        for i, (bar, diff) in enumerate(zip(bars, accuracy_diff)):
            if abs(diff) > 0.05:  # 5%以上の差がある場合のみ表示
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.01 if diff > 0 else -0.01), 
                        f'{diff:+.2f}', ha='center', va='bottom' if diff > 0 else 'top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_statistics(self, analysis, session, save_dir):
        """比較統計情報を保存"""
        # JSON形式で保存（numpy配列をリストに変換）
        json_results = {
            'session': analysis['session'],
            'fact': {
                'overall_accuracy': float(analysis['fact']['overall_accuracy']),
                'base_class_accuracy': {
                    'mean': float(analysis['fact']['base_class_accuracy']['mean']),
                    'std': float(analysis['fact']['base_class_accuracy']['std']),
                    'min': float(analysis['fact']['base_class_accuracy']['min']),
                    'max': float(analysis['fact']['base_class_accuracy']['max']),
                    'values': analysis['fact']['base_class_accuracy']['values'].tolist()
                }
            },
            'base': {
                'overall_accuracy': float(analysis['base']['overall_accuracy']),
                'base_class_accuracy': {
                    'mean': float(analysis['base']['base_class_accuracy']['mean']),
                    'std': float(analysis['base']['base_class_accuracy']['std']),
                    'min': float(analysis['base']['base_class_accuracy']['min']),
                    'max': float(analysis['base']['base_class_accuracy']['max']),
                    'values': analysis['base']['base_class_accuracy']['values'].tolist()
                }
            },
            'comparison': {
                'overall_accuracy_diff': float(analysis['comparison']['overall_accuracy_diff']),
                'overall_accuracy_improvement': float(analysis['comparison']['overall_accuracy_improvement']),
                'base_class_accuracy_diff': float(analysis['comparison']['base_class_accuracy_diff']),
                'winner': analysis['comparison']['winner']
            }
        }
        
        if 'new_class_accuracy' in analysis['fact']:
            json_results['fact']['new_class_accuracy'] = {
                'mean': float(analysis['fact']['new_class_accuracy']['mean']),
                'std': float(analysis['fact']['new_class_accuracy']['std']),
                'min': float(analysis['fact']['new_class_accuracy']['min']),
                'max': float(analysis['fact']['new_class_accuracy']['max']),
                'values': analysis['fact']['new_class_accuracy']['values'].tolist()
            }
        if 'new_class_accuracy' in analysis['base']:
            json_results['base']['new_class_accuracy'] = {
                'mean': float(analysis['base']['new_class_accuracy']['mean']),
                'std': float(analysis['base']['new_class_accuracy']['std']),
                'min': float(analysis['base']['new_class_accuracy']['min']),
                'max': float(analysis['base']['new_class_accuracy']['max']),
                'values': analysis['base']['new_class_accuracy']['values'].tolist()
            }
            # new_class_accuracy_diffも追加
            if 'new_class_accuracy_diff' in analysis['comparison']:
                json_results['comparison']['new_class_accuracy_diff'] = float(analysis['comparison']['new_class_accuracy_diff'])
        
        with open(os.path.join(save_dir, 'comparison_stats.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # テキスト形式でも保存
        with open(os.path.join(save_dir, 'comparison_summary.txt'), 'w') as f:
            f.write(f"Base vs FACT Comparison - Session {session} (Fixed)\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Overall Accuracy:\n")
            f.write(f"  FACT: {analysis['fact']['overall_accuracy']:.4f}\n")
            f.write(f"  Base: {analysis['base']['overall_accuracy']:.4f}\n")
            f.write(f"  Difference: {analysis['comparison']['overall_accuracy_diff']:+.4f}\n")
            f.write(f"  Improvement: {analysis['comparison']['overall_accuracy_improvement']:+.2f}%\n\n")
            
            f.write("Base Classes Accuracy:\n")
            f.write(f"  FACT: {analysis['fact']['base_class_accuracy']['mean']:.4f} ± {analysis['fact']['base_class_accuracy']['std']:.4f}\n")
            f.write(f"  Base: {analysis['base']['base_class_accuracy']['mean']:.4f} ± {analysis['base']['base_class_accuracy']['std']:.4f}\n")
            f.write(f"  Difference: {analysis['comparison']['base_class_accuracy_diff']:+.4f}\n\n")
            
            if 'new_class_accuracy' in analysis['fact']:
                f.write("New Classes Accuracy:\n")
                f.write(f"  FACT: {analysis['fact']['new_class_accuracy']['mean']:.4f} ± {analysis['fact']['new_class_accuracy']['std']:.4f}\n")
                f.write(f"  Base: {analysis['base']['new_class_accuracy']['mean']:.4f} ± {analysis['base']['new_class_accuracy']['std']:.4f}\n")
                f.write(f"  Difference: {analysis['comparison']['new_class_accuracy_diff']:+.4f}\n\n")
            
            f.write(f"Winner: {analysis['comparison']['winner']}\n")
    
    def compare_all_sessions(self, all_results, output_dir):
        """全セッションの比較分析"""
        print("全セッションの比較分析を実行中...")
        
        sessions = sorted(all_results.keys())
        
        # 1. 全体精度の推移比較
        fact_accuracies = [all_results[s]['fact']['overall_accuracy'] for s in sessions]
        base_accuracies = [all_results[s]['base']['overall_accuracy'] for s in sessions]
        
        plt.figure(figsize=(12, 8))
        plt.plot(sessions, fact_accuracies, 'b-o', linewidth=2, markersize=8, label='FACT')
        plt.plot(sessions, base_accuracies, 'r-s', linewidth=2, markersize=8, label='Base')
        plt.title('Overall Accuracy Comparison Across Sessions (Fixed)')
        plt.xlabel('Session')
        plt.ylabel('Overall Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sessions)
        
        # 数値を点の上に表示
        for i, (s, fact_acc, base_acc) in enumerate(zip(sessions, fact_accuracies, base_accuracies)):
            plt.text(s, fact_acc + 0.01, f'{fact_acc:.3f}', ha='center', va='bottom', color='blue')
            plt.text(s, base_acc - 0.01, f'{base_acc:.3f}', ha='center', va='top', color='red')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overall_accuracy_comparison_fixed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ベースクラス精度の推移比較
        fact_base_accuracies = [all_results[s]['fact']['base_class_accuracy']['mean'] for s in sessions]
        base_base_accuracies = [all_results[s]['base']['base_class_accuracy']['mean'] for s in sessions]
        
        plt.figure(figsize=(12, 8))
        plt.plot(sessions, fact_base_accuracies, 'b-o', linewidth=2, markersize=8, label='FACT')
        plt.plot(sessions, base_base_accuracies, 'r-s', linewidth=2, markersize=8, label='Base')
        plt.title('Base Classes Accuracy Comparison Across Sessions (Fixed)')
        plt.xlabel('Session')
        plt.ylabel('Base Classes Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(sessions)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'base_accuracy_comparison_fixed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 改善率の可視化
        improvements = [all_results[s]['comparison']['overall_accuracy_improvement'] for s in sessions]
        
        plt.figure(figsize=(12, 8))
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = plt.bar(sessions, improvements, color=colors, alpha=0.7)
        plt.title('FACT Improvement Over Base Model Across Sessions (Fixed)')
        plt.xlabel('Session')
        plt.ylabel('Improvement (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        plt.xticks(sessions)
        
        # 数値をバーの上に表示
        for bar, imp in zip(bars, improvements):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (1 if imp > 0 else -1), 
                    f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'improvement_comparison_fixed.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 統計サマリーの保存
        self.save_overall_comparison_summary(all_results, output_dir)
    
    def save_overall_comparison_summary(self, all_results, output_dir):
        """全体比較サマリーを保存"""
        sessions = sorted(all_results.keys())
        
        with open(os.path.join(output_dir, 'overall_comparison_summary_fixed.txt'), 'w') as f:
            f.write("Base vs FACT Overall Comparison Summary (Fixed)\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("Session-wise Performance:\n")
            f.write("-" * 40 + "\n")
            for session in sessions:
                results = all_results[session]
                f.write(f"Session {session}:\n")
                f.write(f"  FACT Overall: {results['fact']['overall_accuracy']:.4f}\n")
                f.write(f"  Base Overall: {results['base']['overall_accuracy']:.4f}\n")
                f.write(f"  Improvement: {results['comparison']['overall_accuracy_improvement']:+.2f}%\n")
                f.write(f"  Winner: {results['comparison']['winner']}\n\n")
            
            # 全体統計
            fact_avg = np.mean([all_results[s]['fact']['overall_accuracy'] for s in sessions])
            base_avg = np.mean([all_results[s]['base']['overall_accuracy'] for s in sessions])
            avg_improvement = np.mean([all_results[s]['comparison']['overall_accuracy_improvement'] for s in sessions])
            
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Average FACT Accuracy: {fact_avg:.4f}\n")
            f.write(f"Average Base Accuracy: {base_avg:.4f}\n")
            f.write(f"Average Improvement: {avg_improvement:+.2f}%\n")
            f.write(f"Overall Winner: {'FACT' if fact_avg > base_avg else 'Base'}\n")
            
            # Catastrophic Forgettingの比較
            fact_forgetting = all_results[sessions[0]]['fact']['base_class_accuracy']['mean'] - all_results[sessions[-1]]['fact']['base_class_accuracy']['mean']
            base_forgetting = all_results[sessions[0]]['base']['base_class_accuracy']['mean'] - all_results[sessions[-1]]['base']['base_class_accuracy']['mean']
            
            f.write(f"\nCatastrophic Forgetting Analysis:\n")
            f.write("-" * 35 + "\n")
            f.write(f"FACT Forgetting Rate: {fact_forgetting:.4f}\n")
            f.write(f"Base Forgetting Rate: {base_forgetting:.4f}\n")
            f.write(f"Forgetting Control: {'FACT' if fact_forgetting < base_forgetting else 'Base'}\n")

def main():
    parser = argparse.ArgumentParser(description='BaseとFACTモデルの同時比較（修正版）')
    parser.add_argument('-dataset', type=str, default='cicids2017_improved', 
                       choices=['cifar100', 'cub200', 'mini_imagenet', 'cicids2017_improved'])
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-fact_checkpoint_dir', type=str, 
                       default='checkpoint/cicids2017_improved/fact',
                       help='FACTモデルのチェックポイントディレクトリ')
    parser.add_argument('-base_checkpoint_dir', type=str, 
                       default='checkpoint/cicids2017_improved/base',
                       help='Baseモデルのチェックポイントディレクトリ')
    parser.add_argument('-output_dir', type=str, default='base_fact_comparison_fixed',
                       help='出力ディレクトリ')
    parser.add_argument('-sessions', type=int, nargs='+', default=[0, 1, 2, 3, 4, 5, 6],
                       help='比較するセッション')
    parser.add_argument('-batch_size_base', type=int, default=128,
                       help='ベースセッションのバッチサイズ')
    parser.add_argument('-batch_size_new', type=int, default=0,
                       help='新規セッションのバッチサイズ')
    parser.add_argument('-test_batch_size', type=int, default=100,
                       help='テストのバッチサイズ')
    parser.add_argument('-num_workers', type=int, default=8,
                       help='データローダーのワーカー数')
    parser.add_argument('-temperature', type=float, default=16.0,
                       help='温度パラメータ')
    parser.add_argument('-max_samples', type=int, default=None,
                       help='デバッグ用の最大サンプル数制限')
    
    args = parser.parse_args()
    
    # セッションリストを保存
    sessions_to_compare = args.sessions.copy()
    
    # データセット設定
    args = set_up_datasets(args)
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 比較器の初期化
    comparator = BaseFactComparatorFixed(args)
    
    # 各セッションを比較
    all_results = {}
    for session in sessions_to_compare:
        results = comparator.evaluate_session(session, args.fact_checkpoint_dir, args.base_checkpoint_dir, args.output_dir)
        if results:
            all_results[session] = results
    
    # 全セッションの比較
    if len(all_results) > 1:
        comparator.compare_all_sessions(all_results, args.output_dir)
    
    print(f"\n修正版比較分析完了！結果は {args.output_dir} に保存されました。")
    print(f"比較したセッション: {list(all_results.keys())}")

if __name__ == '__main__':
    main()
