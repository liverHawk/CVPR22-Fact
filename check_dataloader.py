#!/usr/bin/env python3
"""データローダが正しくデータを読み込んでいるか確認"""
import sys
import argparse
import numpy as np
import torch

# データローダをインポート
from dataloader.data_utils import set_up_datasets, get_new_dataloader

def check_dataloader():
    """データローダの動作を確認"""
    # パラメータを設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CICIDS2017_improved')
    parser.add_argument('--dataset_type', type=str, default='CICIDS2017_improved')
    parser.add_argument('--dataset_name', type=str, default='CICIDS2017_flow_improved')
    parser.add_argument('--dataroot', type=str, default='data/')
    parser.add_argument('--label_column', type=str, default='Label')
    parser.add_argument('--normalize_method', type=str, default='standard')
    parser.add_argument('--base_class', type=int, default=5)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--way', type=int, default=1)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--batch_size_new', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_gpu', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=False)
    
    args = parser.parse_args([])
    
    # データセットのセットアップ
    args = set_up_datasets(args)
    
    print("="*80)
    print("データローダの検証")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Base classes: {args.base_class}")
    print(f"Total classes: {args.num_classes}")
    print(f"Way: {args.way}")
    print(f"Shot: {args.shot}")
    print(f"Sessions: {args.sessions}")
    print()
    
    # 各新規セッションでのトレーニングデータを確認
    for session in range(1, args.sessions):
        print("="*80)
        print(f"Session {session} (新規クラス: {args.base_class + (session-1) * args.way})")
        print("="*80)
        
        trainset, trainloader, testloader = get_new_dataloader(args, session)
        
        # トレーニングデータのクラスを確認
        train_targets = trainset.targets
        unique_train_classes = np.unique(train_targets)
        
        print(f"\n[Training Data]")
        print(f"  Total samples: {len(train_targets)}")
        print(f"  Unique classes: {unique_train_classes}")
        print(f"  Expected new class index: {args.base_class + (session-1) * args.way}")
        
        # ラベル名を取得
        if hasattr(trainset, 'idx_to_label'):
            idx_to_label = trainset.idx_to_label
            unique_train_labels = [idx_to_label[idx] for idx in unique_train_classes]
            print(f"  Unique labels: {unique_train_labels}")
        
        # クラスごとのサンプル数
        print(f"\n  Class distribution:")
        for cls_idx in unique_train_classes:
            count = np.sum(train_targets == cls_idx)
            if hasattr(trainset, 'idx_to_label'):
                label_name = trainset.idx_to_label[cls_idx]
                print(f"    Class {cls_idx} ({label_name}): {count} samples")
            else:
                print(f"    Class {cls_idx}: {count} samples")
        
        # ベースクラスが含まれているか確認
        base_classes = np.arange(args.base_class)
        base_class_overlap = np.intersect1d(unique_train_classes, base_classes)
        
        if len(base_class_overlap) > 0:
            print(f"\n  ⚠️  WARNING: Base class indices found in training data: {base_class_overlap}")
            if hasattr(trainset, 'idx_to_label'):
                overlapping_labels = [trainset.idx_to_label[idx] for idx in base_class_overlap]
                print(f"  ⚠️  Base class labels: {overlapping_labels}")
        else:
            print(f"\n  ✓ OK: Only new class indices in training data")
        
        # テストデータのクラスを確認
        test_targets = testloader.dataset.targets
        unique_test_classes = np.unique(test_targets)
        
        print(f"\n[Test Data]")
        print(f"  Total samples: {len(test_targets)}")
        print(f"  Unique classes: {unique_test_classes}")
        print(f"  Expected classes: 0 to {args.base_class + session * args.way - 1}")
        
        if hasattr(testloader.dataset, 'idx_to_label'):
            idx_to_label = testloader.dataset.idx_to_label
            unique_test_labels = [idx_to_label[idx] for idx in unique_test_classes]
            print(f"  Unique labels: {unique_test_labels}")
        
        print()

if __name__ == '__main__':
    check_dataloader()
