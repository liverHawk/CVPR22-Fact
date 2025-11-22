#!/usr/bin/env python3
"""
Create session index files for CICIDS2017_improved dataset.
Configuration is loaded from params.yaml
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.config_utils import load_config, get_dataset_config, get_index_generation_config


def get_delete_columns():
    """Get list of columns to delete (same as in dataloader)."""
    return [
        "id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp",
        "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd RST Flags",
        "Bwd RST Flags", "FIN Flag Count", "RST Flag Count", "URG Flag Count",
        "CWR Flag Count", "ECE Flag Count", "Fwd Bytes/Bulk Avg",
        "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg", "Bwd Bytes/Bulk Avg",
        "ICMP Code", "ICMP Type", "Total TCP Flow Time",
    ]


def preprocess_dataframe(df):
    """Apply preprocessing to dataframe (same as in dataloader)."""
    # Drop columns
    delete_cols = get_delete_columns()
    existing_delete_cols = [col for col in delete_cols if col in df.columns]
    df = df.drop(existing_delete_cols, axis=1)

    if 'Attempted Category' in df.columns:
        df = df.drop(columns=['Attempted Category'])

    # Replace infinite values with NaN and drop NaN rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Label consolidation
    labels = df["Label"].unique()
    for label in labels:
        if "Attempted" in label:
            df.loc[df["Label"] == label, "Label"] = "BENIGN"
        if "Web Attack" in label:
            df.loc[df["Label"] == label, "Label"] = "Web Attack"
        if "Infiltration" in label:
            df.loc[df["Label"] == label, "Label"] = "Infiltration"
        if "DoS" in label and label != "DDoS":
            df.loc[df["Label"] == label, "Label"] = "DoS"

    return df


def main():
    # Load configuration
    print("Loading configuration from params.yaml...")
    config = load_config('params.yaml')
    dataset_cfg = get_dataset_config(config)
    index_cfg = get_index_generation_config(config)

    # Extract parameters
    dataset_name = dataset_cfg.get('name', 'cicids2017_improved')
    dataroot = dataset_cfg.get('dataroot', 'data/')
    data_dir = os.path.join(dataroot, 'CICIDS2017_improved')
    index_dir = index_cfg.get('output_dir', 'data/index_list/cicids2017_improved')
    num_shots = index_cfg.get('shots_per_class', 5)
    random_seed = index_cfg.get('random_seed', 42)

    # Get session configuration
    session_cfg = dataset_cfg.get('sessions', {})
    base_class_count = session_cfg.get('base_class', 4)
    num_classes = session_cfg.get('num_classes', 10)
    way = session_cfg.get('way', 1)

    # Get CICIDS2017 specific configuration
    cicids_cfg = dataset_cfg.get('cicids2017', {})
    base_class_names = cicids_cfg.get('base_classes', ['BENIGN', 'DDoS', 'DoS', 'Portscan'])
    incremental_class_names = cicids_cfg.get('incremental_classes', [
        'Botnet', 'FTP-Patator', 'Heartbleed', 'Infiltration', 'SSH-Patator', 'Web Attack'
    ])

    # Set random seed
    np.random.seed(random_seed)

    # Create output directory
    os.makedirs(index_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Data directory: {data_dir}")
    print(f"  Output directory: {index_dir}")
    print(f"  Base classes: {base_class_count}")
    print(f"  Total classes: {num_classes}")
    print(f"  Way: {way}")
    print(f"  Shots per class: {num_shots}")
    print(f"  Random seed: {random_seed}")

    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    # Apply preprocessing
    print("Applying preprocessing...")
    train_df = preprocess_dataframe(train_df)

    # Encode labels
    le = LabelEncoder()
    train_labels = le.fit_transform(train_df["Label"].values)

    print(f"\nUnique classes after preprocessing: {len(le.classes_)}")
    print("Class names:", le.classes_)

    # Count samples per class
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    print("\nSamples per class:")
    for label_idx, count in zip(unique_labels, counts):
        print(f"  {label_idx}: {le.classes_[label_idx]:20s} - {count:7d} samples")

    # Map class names to indices
    base_classes = []
    incremental_classes = []

    for class_name in base_class_names:
        if class_name in le.classes_:
            idx = np.where(le.classes_ == class_name)[0][0]
            base_classes.append(idx)
        else:
            print(f"Warning: Base class '{class_name}' not found in dataset")

    for class_name in incremental_class_names:
        if class_name in le.classes_:
            idx = np.where(le.classes_ == class_name)[0][0]
            incremental_classes.append(idx)
        else:
            print(f"Warning: Incremental class '{class_name}' not found in dataset")

    base_classes = sorted(base_classes)
    incremental_classes = sorted(incremental_classes)

    print(f"\nBase session classes ({len(base_classes)}): {[le.classes_[i] for i in base_classes]}")
    print(f"Incremental classes ({len(incremental_classes)}): {[le.classes_[i] for i in incremental_classes]}")

    # Validate configuration
    if len(base_classes) != base_class_count:
        print(f"Warning: Expected {base_class_count} base classes, got {len(base_classes)}")

    expected_incremental = num_classes - base_class_count
    if len(incremental_classes) != expected_incremental:
        print(f"Warning: Expected {expected_incremental} incremental classes, got {len(incremental_classes)}")

    # Create session 0 (base session)
    print("\nCreating session index files...")
    session_0_indices = []
    for class_idx in base_classes:
        class_sample_indices = np.where(train_labels == class_idx)[0]
        session_0_indices.extend(class_sample_indices.tolist())

    session_0_file = os.path.join(index_dir, 'session_0.txt')
    with open(session_0_file, 'w') as f:
        for idx in session_0_indices:
            f.write(f"{idx}\n")
    print(f"  Created {session_0_file}: {len(session_0_indices)} samples")

    # Create incremental sessions
    for session_num, class_idx in enumerate(incremental_classes, start=1):
        class_sample_indices = np.where(train_labels == class_idx)[0]

        # Select samples for few-shot learning
        if len(class_sample_indices) > num_shots:
            selected_indices = np.random.choice(class_sample_indices, num_shots, replace=False)
        else:
            selected_indices = class_sample_indices

        session_file = os.path.join(index_dir, f'session_{session_num}.txt')
        with open(session_file, 'w') as f:
            for idx in selected_indices:
                f.write(f"{idx}\n")

        print(f"  Created {session_file}: {len(selected_indices)} samples from class {le.classes_[class_idx]}")

    print("\n✓ All session index files created successfully!")


if __name__ == "__main__":
    main()
