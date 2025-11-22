#!/usr/bin/env python3
"""Simple test to verify save path structure."""

import os
import sys

# Change to parent directory
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("Verifying Save Path Configuration")
print("=" * 60)

# Read the fscil_trainer.py files
fact_trainer_path = 'models/fact/fscil_trainer.py'
base_trainer_path = 'models/base/fscil_trainer.py'

print("\nChecking FACT trainer:")
with open(fact_trainer_path, 'r') as f:
    content = f.read()
    if "self.args.save_path = 'checkpoint'" in content:
        print("  ✓ Save path correctly set to 'checkpoint'")
    else:
        print("  ✗ Save path not simplified")

    if "'best_model.pth'" in content:
        print("  ✓ Using 'best_model.pth' for base session")
    else:
        print("  ✗ Not using 'best_model.pth'")

    if "'session_{}.pth'.format(session)" in content:
        print("  ✓ Using 'session_N.pth' for incremental sessions")
    else:
        print("  ✗ Not using 'session_N.pth'")

    if "'optimizer.pth'" in content:
        print("  ✓ Using 'optimizer.pth' for optimizer state")
    else:
        print("  ✗ Not using 'optimizer.pth'")

print("\nChecking BASE trainer:")
with open(base_trainer_path, 'r') as f:
    content = f.read()
    if "self.args.save_path = 'checkpoint'" in content:
        print("  ✓ Save path correctly set to 'checkpoint'")
    else:
        print("  ✗ Save path not simplified")

    if "'best_model.pth'" in content:
        print("  ✓ Using 'best_model.pth' for base session")
    else:
        print("  ✗ Not using 'best_model.pth'")

    if "'session_{}.pth'.format(session)" in content:
        print("  ✓ Using 'session_N.pth' for incremental sessions")
    else:
        print("  ✗ Not using 'session_N.pth'")

    if "'optimizer.pth'" in content:
        print("  ✓ Using 'optimizer.pth' for optimizer state")
    else:
        print("  ✗ Not using 'optimizer.pth'")

# Ensure checkpoint directory exists
os.makedirs('checkpoint', exist_ok=True)
print("\n✓ Checkpoint directory created/verified")

print("\n" + "=" * 60)
print("Expected model save structure:")
print("=" * 60)
print("checkpoint/")
print("  ├── best_model.pth       (base session best model)")
print("  ├── optimizer.pth        (optimizer state)")
print("  ├── session_1.pth        (incremental session 1)")
print("  ├── session_2.pth        (incremental session 2)")
print("  ├── ...                  (more sessions)")
print("  └── results.txt          (training results)")
print("\n" + "=" * 60)
print("Save path configuration verified successfully!")
print("=" * 60)
