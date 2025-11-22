#!/usr/bin/env python3
"""Test script for CICIDS2017_improved dataset loader."""

import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.cicids2017.cicids2017 import CICIDS2017_improved

def test_base_session():
    """Test loading base session data."""
    print("=" * 60)
    print("Testing Base Session (Session 0)")
    print("=" * 60)

    dataroot = 'data/'
    class_index = np.arange(4)  # Base session: 4 classes

    print(f"Loading training set with {len(class_index)} base classes...")
    trainset = CICIDS2017_improved(root=dataroot, train=True,
                                    index=class_index, base_sess=True)

    print(f"Loading test set with {len(class_index)} base classes...")
    testset = CICIDS2017_improved(root=dataroot, train=False,
                                   index=class_index, base_sess=True)

    print(f"\nTraining set: {len(trainset)} samples")
    print(f"Test set: {len(testset)} samples")
    print(f"Feature dimension: {trainset.data.shape[1]}")
    print(f"Unique classes in train: {len(np.unique(trainset.targets))}")
    print(f"Unique classes in test: {len(np.unique(testset.targets))}")

    # Test a sample
    features, label = trainset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample label: {label}")
    print(f"Features dtype: {features.dtype}")
    print(f"Label dtype: {label.dtype}")

    # Create a DataLoader
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=128, shuffle=True, num_workers=0
    )

    # Test loading a batch
    for batch_features, batch_labels in trainloader:
        print(f"\nBatch features shape: {batch_features.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Batch labels unique: {torch.unique(batch_labels).numpy()}")
        break

    print("\n✓ Base session test passed!")
    return trainset, testset

def test_new_session():
    """Test loading new session data using index path."""
    print("\n" + "=" * 60)
    print("Testing New Session (Session 1)")
    print("=" * 60)

    dataroot = 'data/'
    index_path = 'data/index_list/CICIDS2017_improved/session_1.txt'

    print(f"Loading training set from index file: {index_path}")
    trainset = CICIDS2017_improved(root=dataroot, train=True,
                                    index_path=index_path, base_sess=False)

    print(f"\nTraining set: {len(trainset)} samples")
    print(f"Feature dimension: {trainset.data.shape[1]}")
    print(f"Unique classes: {np.unique(trainset.targets)}")

    # Test a sample
    features, label = trainset[0]
    print(f"\nSample features shape: {features.shape}")
    print(f"Sample label: {label}")

    print("\n✓ New session test passed!")
    return trainset

def test_class_distribution():
    """Test class distribution in the dataset."""
    print("\n" + "=" * 60)
    print("Testing Class Distribution")
    print("=" * 60)

    dataroot = 'data/'
    class_index = np.arange(4)

    trainset = CICIDS2017_improved(root=dataroot, train=True,
                                    index=class_index, base_sess=True)

    # Count samples per class
    unique, counts = np.unique(trainset.targets, return_counts=True)

    print(f"\nNumber of classes: {len(unique)}")
    print(f"Total samples: {len(trainset)}")
    print(f"\nClass distribution (first 10 classes):")
    for class_idx, count in list(zip(unique, counts))[:10]:
        label_name = trainset.label_encoder.inverse_transform([class_idx])[0]
        print(f"  Class {class_idx} ({label_name}): {count} samples")

    print("\n✓ Class distribution test passed!")

if __name__ == "__main__":
    try:
        # Test base session
        trainset, testset = test_base_session()

        # Test new session
        new_trainset = test_new_session()

        # Test class distribution
        test_class_distribution()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
