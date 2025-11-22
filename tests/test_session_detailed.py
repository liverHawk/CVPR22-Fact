#!/usr/bin/env python3
"""Detailed test to verify session-based data loading."""

import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.cicids2017.cicids2017 import CICIDS2017_improved

def test_session_based_loading():
    """Test loading data using index_path (as used in training)."""
    print("=" * 60)
    print("Testing Session-Based Data Loading")
    print("=" * 60)

    dataroot = 'data/'

    # Test base session (session 0) with index_path
    print("\n1. Base Session (session_0.txt):")
    session_0_path = 'data/index_list/CICIDS2017_improved/session_0.txt'
    trainset_0 = CICIDS2017_improved(
        root=dataroot,
        train=True,
        index_path=session_0_path,
        base_sess=False
    )

    unique_0, counts_0 = np.unique(trainset_0.targets, return_counts=True)
    print(f"   Samples: {len(trainset_0)}")
    print(f"   Unique classes: {len(unique_0)}")
    print("   Class distribution:")
    for class_idx, count in zip(unique_0, counts_0):
        label_name = trainset_0.label_encoder.inverse_transform([class_idx])[0]
        print(f"      Class {class_idx} ({label_name}): {count} samples")

    # Test incremental sessions
    for session_num in range(1, 7):
        print(f"\n{session_num + 1}. Incremental Session {session_num} (session_{session_num}.txt):")
        session_path = f'data/index_list/CICIDS2017_improved/session_{session_num}.txt'

        trainset = CICIDS2017_improved(
            root=dataroot,
            train=True,
            index_path=session_path,
            base_sess=False
        )

        unique, counts = np.unique(trainset.targets, return_counts=True)
        print(f"   Samples: {len(trainset)}")
        print(f"   Unique classes: {len(unique)}")
        for class_idx, count in zip(unique, counts):
            label_name = trainset.label_encoder.inverse_transform([class_idx])[0]
            print(f"      Class {class_idx} ({label_name}): {count} samples")

    print("\n" + "=" * 60)
    print("✓ Session-based loading test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_session_based_loading()
