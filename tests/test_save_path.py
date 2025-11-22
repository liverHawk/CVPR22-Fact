#!/usr/bin/env python3
"""Test script to verify simplified save path structure."""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fact.fscil_trainer import FSCILTrainer
from utils import set_seed

def test_save_path():
    """Test that save path is correctly set to checkpoint/"""
    print("=" * 60)
    print("Testing Save Path Configuration")
    print("=" * 60)

    # Create minimal args
    parser = argparse.ArgumentParser()
    parser.add_argument('-project', type=str, default='fact')
    parser.add_argument('-dataset', type=str, default='cicids2017_improved')
    parser.add_argument('-dataroot', type=str, default='data/')
    parser.add_argument('-epochs_base', type=int, default=1)
    parser.add_argument('-epochs_new', type=int, default=1)
    parser.add_argument('-lr_base', type=float, default=0.005)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Milestone')
    parser.add_argument('-milestones', nargs='+', type=int, default=[50, 100])
    parser.add_argument('-step', type=int, default=20)
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.25)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true')
    parser.add_argument('-batch_size_base', type=int, default=128)
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
    parser.add_argument('-gpu', default='0')
    parser.add_argument('-num_workers', type=int, default=0)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')

    args = parser.parse_args([])
    set_seed(args.seed)
    args.num_gpu = 1

    # Initialize trainer (this will call set_save_path)
    print("\nInitializing trainer...")
    try:
        trainer = FSCILTrainer(args)

        # Check save path
        print(f"\nConfigured save path: {trainer.args.save_path}")

        # Verify it's the simple structure
        expected_path = 'checkpoint'
        if trainer.args.save_path == expected_path:
            print(f"✓ Save path correctly set to: {expected_path}")
        else:
            print(f"✗ Expected: {expected_path}")
            print(f"✗ Got: {trainer.args.save_path}")
            return False

        # Check that checkpoint directory exists
        if os.path.exists(expected_path):
            print(f"✓ Checkpoint directory exists")
        else:
            print(f"✗ Checkpoint directory does not exist")
            return False

        print("\n" + "=" * 60)
        print("Save Path Configuration Test: PASSED")
        print("=" * 60)

        print("\nExpected model save locations:")
        print(f"  - Best model (session 0): {os.path.join(expected_path, 'best_model.pth')}")
        print(f"  - Optimizer state: {os.path.join(expected_path, 'optimizer.pth')}")
        print(f"  - Session N models: {os.path.join(expected_path, 'session_N.pth')}")
        print(f"  - Results: {os.path.join(expected_path, 'results.txt')}")

        return True

    except Exception as e:
        print(f"\n✗ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_save_path()
    sys.exit(0 if success else 1)
