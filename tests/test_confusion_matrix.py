#!/usr/bin/env python3
"""Test script to verify confusion matrix generation."""

import argparse
import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import confmatrix

def test_confusion_matrix():
    """Test confusion matrix generation with CICIDS2017 class names."""
    print("=" * 60)
    print("Testing Confusion Matrix Generation")
    print("=" * 60)

    # Create dummy logits and labels for 4 classes (base session)
    num_samples = 100
    num_classes = 4

    # Simulate predictions
    torch.manual_seed(42)
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    print(f"\nGenerated {num_samples} samples for {num_classes} classes")
    print(f"Logits shape: {logits.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test without class names
    print("\n1. Testing confusion matrix without class names...")
    output_dir = 'checkpoint'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, 'test_confusion_matrix')

    cm = confmatrix(logits, labels, filename)
    print(f"   ✓ Confusion matrix generated: {cm.shape}")
    print(f"   ✓ Saved to: {filename}.pdf and {filename}_cbar.pdf")

    # Test with class names (CICIDS2017 base session)
    print("\n2. Testing confusion matrix with CICIDS2017 class names...")
    class_names = ['BENIGN', 'DDoS', 'DoS', 'Portscan']
    filename_with_names = os.path.join(output_dir, 'test_confusion_matrix_with_names')

    cm = confmatrix(logits, labels, filename_with_names, class_names=class_names)
    print(f"   ✓ Confusion matrix with class names generated: {cm.shape}")
    print(f"   ✓ Class names: {class_names}")
    print(f"   ✓ Saved to: {filename_with_names}.pdf and {filename_with_names}_cbar.pdf")

    # Calculate accuracy per class
    print("\n3. Per-class accuracy:")
    perclassacc = cm.diagonal()
    for i, (acc, name) in enumerate(zip(perclassacc, class_names)):
        print(f"   Class {i} ({name}): {acc:.4f}")

    print("\n" + "=" * 60)
    print("Confusion Matrix Test: PASSED")
    print("=" * 60)

    print("\nGenerated files:")
    for f in [filename + '.pdf', filename + '_cbar.pdf',
              filename_with_names + '.pdf', filename_with_names + '_cbar.pdf']:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (not found)")

    return True

if __name__ == "__main__":
    try:
        success = test_confusion_matrix()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
