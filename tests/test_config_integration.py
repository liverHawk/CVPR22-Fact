#!/usr/bin/env python3
"""Test configuration integration with train.py argument parsing."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config_utils import load_config, merge_args_with_config

def test_config_merge():
    """Test that configuration merging works correctly."""
    print("=" * 60)
    print("Testing Configuration Integration")
    print("=" * 60)

    # Create a minimal argument parser similar to train.py
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='params.yaml')
    parser.add_argument('-project', type=str, default='fact')
    parser.add_argument('-dataset', type=str, default='cifar100')
    parser.add_argument('-epochs_base', type=int, default=400)
    parser.add_argument('-lr_base', type=float, default=0.005)
    parser.add_argument('-batch_size_base', type=int, default=256)
    parser.add_argument('-gpu', default='0,1,2,3')
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Milestone')
    parser.add_argument('-milestones', nargs='+', type=int, default=[50, 100])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.25)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true')
    parser.add_argument('-batch_size_new', type=int, default=0)
    parser.add_argument('-test_batch_size', type=int, default=100)
    parser.add_argument('-base_mode', type=str, default='ft_cos')
    parser.add_argument('-new_mode', type=str, default='avg_cos')
    parser.add_argument('-balance', type=float, default=0.01)
    parser.add_argument('-loss_iter', type=int, default=0)
    parser.add_argument('-alpha', type=float, default=2.0)
    parser.add_argument('-eta', type=float, default=0.1)
    parser.add_argument('-start_session', type=int, default=0)
    parser.add_argument('-model_dir', type=str, default=None)
    parser.add_argument('-set_no_val', action='store_true')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-debug', action='store_true')

    # Parse with no command line args (use defaults)
    args = parser.parse_args([])

    print("\n1. Original args (before config merge):")
    print(f"   dataset: {args.dataset}")
    print(f"   project: {args.project}")
    print(f"   epochs_base: {args.epochs_base}")
    print(f"   lr_base: {args.lr_base}")
    print(f"   batch_size_base: {args.batch_size_base}")

    # Load and merge configuration
    config = load_config('params.yaml')
    args = merge_args_with_config(args, config)

    print("\n2. Merged args (after loading params.yaml):")
    print(f"   dataset: {args.dataset}")
    print(f"   project: {args.project}")
    print(f"   epochs_base: {args.epochs_base}")
    print(f"   lr_base: {args.lr_base}")
    print(f"   batch_size_base: {args.batch_size_base}")
    print(f"   seed: {args.seed}")
    print(f"   gpu: {args.gpu}")

    # Verify expected values from params.yaml
    tests_passed = 0
    tests_failed = 0

    print("\n3. Verification:")

    # Check dataset
    if args.dataset == 'cicids2017_improved':
        print("   ✓ Dataset correctly set to 'cicids2017_improved'")
        tests_passed += 1
    else:
        print(f"   ✗ Dataset incorrect: {args.dataset}")
        tests_failed += 1

    # Check project
    if args.project == 'fact':
        print("   ✓ Project correctly set to 'fact'")
        tests_passed += 1
    else:
        print(f"   ✗ Project incorrect: {args.project}")
        tests_failed += 1

    # Check epochs
    if args.epochs_base == 400:
        print("   ✓ Base epochs correctly set to 400")
        tests_passed += 1
    else:
        print(f"   ✗ Base epochs incorrect: {args.epochs_base}")
        tests_failed += 1

    # Check learning rate
    if args.lr_base == 0.005:
        print("   ✓ Base learning rate correctly set to 0.005")
        tests_passed += 1
    else:
        print(f"   ✗ Base learning rate incorrect: {args.lr_base}")
        tests_failed += 1

    # Check batch size
    if args.batch_size_base == 256:
        print("   ✓ Base batch size correctly set to 256")
        tests_passed += 1
    else:
        print(f"   ✗ Base batch size incorrect: {args.batch_size_base}")
        tests_failed += 1

    # Check FACT parameters
    if args.eta == 0.1:
        print("   ✓ FACT eta correctly set to 0.1")
        tests_passed += 1
    else:
        print(f"   ✗ FACT eta incorrect: {args.eta}")
        tests_failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    if tests_failed > 0:
        print("\n✗ Configuration integration test FAILED")
        return False
    else:
        print("\n✓ Configuration integration test PASSED")
        return True


if __name__ == "__main__":
    success = test_config_merge()
    sys.exit(0 if success else 1)
