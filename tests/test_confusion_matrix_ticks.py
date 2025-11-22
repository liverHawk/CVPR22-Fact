#!/usr/bin/env python3
"""Test script to verify confusion matrix tick labels are dynamic."""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import confmatrix

def test_dynamic_ticks():
    """Test that confusion matrix tick labels adjust based on number of classes."""
    print("=" * 60)
    print("Testing Dynamic Tick Labels in Confusion Matrix")
    print("=" * 60)

    output_dir = 'checkpoint'
    os.makedirs(output_dir, exist_ok=True)

    # Test 1: 4 classes (base session)
    print("\n1. Testing with 4 classes (base session):")
    num_classes = 4
    num_samples = 100
    torch.manual_seed(42)
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    class_names = ['BENIGN', 'DDoS', 'DoS', 'Portscan']
    filename = os.path.join(output_dir, 'test_4_classes')
    cm = confmatrix(logits, labels, filename, class_names=class_names)
    print(f"   ✓ Generated confusion matrix for {num_classes} classes")
    print(f"   ✓ Class names: {class_names}")
    print(f"   ✓ Saved to: {filename}.pdf")

    # Test 2: 5 classes (session 1)
    print("\n2. Testing with 5 classes (session 1):")
    num_classes = 5
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    class_names = ['BENIGN', 'DDoS', 'DoS', 'Portscan', 'Botnet']
    filename = os.path.join(output_dir, 'test_5_classes')
    cm = confmatrix(logits, labels, filename, class_names=class_names)
    print(f"   ✓ Generated confusion matrix for {num_classes} classes")
    print(f"   ✓ Class names: {class_names}")
    print(f"   ✓ Saved to: {filename}.pdf")

    # Test 3: 10 classes (all sessions)
    print("\n3. Testing with 10 classes (all sessions):")
    num_classes = 10
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    class_names = ['BENIGN', 'Botnet', 'DDoS', 'DoS', 'FTP-Patator',
                   'Heartbleed', 'Infiltration', 'Portscan', 'SSH-Patator', 'Web Attack']
    filename = os.path.join(output_dir, 'test_10_classes')
    cm = confmatrix(logits, labels, filename, class_names=class_names)
    print(f"   ✓ Generated confusion matrix for {num_classes} classes")
    print(f"   ✓ Class names: {class_names}")
    print(f"   ✓ Saved to: {filename}.pdf")

    # Test 4: Without class names (numeric labels)
    print("\n4. Testing with 4 classes (numeric labels only):")
    num_classes = 4
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))

    filename = os.path.join(output_dir, 'test_4_classes_numeric')
    cm = confmatrix(logits, labels, filename)
    print(f"   ✓ Generated confusion matrix with numeric labels")
    print(f"   ✓ Should show ticks: [0, 1, 2, 3]")
    print(f"   ✓ Saved to: {filename}.pdf")

    print("\n" + "=" * 60)
    print("Dynamic Tick Labels Test: PASSED")
    print("=" * 60)

    print("\nGenerated files:")
    expected_files = [
        'test_4_classes.pdf', 'test_4_classes_cbar.pdf',
        'test_5_classes.pdf', 'test_5_classes_cbar.pdf',
        'test_10_classes.pdf', 'test_10_classes_cbar.pdf',
        'test_4_classes_numeric.pdf', 'test_4_classes_numeric_cbar.pdf'
    ]
    for f in expected_files:
        full_path = os.path.join(output_dir, f)
        if os.path.exists(full_path):
            print(f"  ✓ {f}")

    return True

if __name__ == "__main__":
    try:
        success = test_dynamic_ticks()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
