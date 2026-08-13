"""
Create patient-level data splits with manifests and leakage verification.

This script:
1. Loads preprocessed point cloud data
2. Creates patient-level train/val/test splits (no data leakage)
3. Generates split manifests (frozen for reproducibility)
4. Verifies leakage constraints
5. Creates experiment directories

Usage (basic):
  python scripts/create_data_splits.py --input ./data/processed/full.h5

Usage (custom splits):
  python scripts/create_data_splits.py \\
    --input ./data/processed/full.h5 \\
    --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2

Usage (with stratification):
  python scripts/create_data_splits.py --input ./data/processed/full.h5 --stratify
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import h5py
from tqdm import tqdm

from utils import set_random_seed, split_by_patient, check_split_leakage


def load_split_data(h5_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load point clouds, labels, and patient IDs from HDF5.
    
    Args:
        h5_path: Path to HDF5 file
        
    Returns:
        Tuple of (points, labels, patient_ids)
    """
    with h5py.File(h5_path, 'r') as f:
        points = f['points'][:]
        labels = f['labels'][:]
        patient_ids = [pid.decode() if isinstance(pid, bytes) else pid 
                      for pid in f['patient_ids'][:]]
    
    return points, labels, patient_ids


def create_split_manifest(split_name: str,
                         indices: List[int],
                         points: np.ndarray,
                         labels: np.ndarray,
                         patient_ids: List[str],
                         output_dir: str) -> Dict:
    """
    Create a split manifest.
    
    Args:
        split_name: Name of split (train/val/test)
        indices: Indices for this split
        points: All point clouds
        labels: All labels
        patient_ids: All patient IDs
        output_dir: Output directory
        
    Returns:
        Split manifest dictionary
    """
    manifest = {
        'split': split_name,
        'num_samples': len(indices),
        'samples': []
    }
    
    # Collect label statistics
    split_labels = labels[indices]
    label_counts = np.bincount(split_labels[split_labels >= 0])  # Ignore -1 (unlabeled)
    
    manifest['label_distribution'] = {
        'negative': int(label_counts[0]) if len(label_counts) > 0 else 0,
        'positive': int(label_counts[1]) if len(label_counts) > 1 else 0,
    }
    
    # List unique patients in this split
    split_patient_ids = [patient_ids[i] for i in indices]
    manifest['unique_patients'] = sorted(list(set(split_patient_ids)))
    manifest['num_patients'] = len(manifest['unique_patients'])
    
    # Create sample records
    for idx in indices:
        manifest['samples'].append({
            'index': int(idx),
            'patient_id': patient_ids[idx],
            'label': int(labels[idx]),
            'num_points': int(points[idx].shape[0]),
            'num_features': int(points[idx].shape[1])
        })
    
    return manifest


def create_splits(h5_path: str,
                 output_dir: str,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 stratify: bool = False,
                 seed: int = 42) -> Dict:
    """
    Create patient-level train/val/test splits.
    
    Args:
        h5_path: Path to preprocessed data HDF5
        output_dir: Output directory
        train_ratio: Training split ratio
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        stratify: Whether to stratify by label (if available)
        seed: Random seed
        
    Returns:
        Dictionary with split results and statistics
    """
    set_random_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n📂 Loading data from {h5_path}...")
    points, labels, patient_ids = load_split_data(h5_path)
    print(f"✅ Loaded {len(points)} samples from {len(set(patient_ids))} patients")
    
    # Create patient-level split
    print(f"\n🔀 Creating patient-level splits (train/val/test = {train_ratio}/{val_ratio}/{test_ratio})...")
    train_patients, val_patients, test_patients = split_by_patient(
        patient_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Map patients to indices
    train_indices = [i for i, pid in enumerate(patient_ids) if pid in train_patients]
    val_indices = [i for i, pid in enumerate(patient_ids) if pid in val_patients]
    test_indices = [i for i, pid in enumerate(patient_ids) if pid in test_patients]
    
    print(f"✅ Train: {len(train_indices)} samples ({len(train_patients)} patients)")
    print(f"✅ Val: {len(val_indices)} samples ({len(val_patients)} patients)")
    print(f"✅ Test: {len(test_indices)} samples ({len(test_patients)} patients)")
    
    # Check leakage
    print(f"\n🔍 Checking for data leakage...")
    leakage_check = check_split_leakage(train_patients, val_patients, test_patients)
    print(leakage_check['report'])
    
    if leakage_check['has_leakage']:
        raise RuntimeError("Patient-level leakage detected! Cannot proceed.")
    
    # Create manifests for each split
    print(f"\n📋 Creating split manifests...")
    splits_data = {}
    
    for split_name, indices in [
        ('train', train_indices),
        ('val', val_indices),
        ('test', test_indices)
    ]:
        manifest = create_split_manifest(
            split_name, indices, points, labels, patient_ids, output_dir
        )
        splits_data[split_name] = manifest
        
        # Save individual split manifest
        manifest_path = output_path / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"✅ Saved {split_name} manifest to {manifest_path}")
    
    # Save combined manifest
    combined_manifest = {
        'dataset': h5_path,
        'total_samples': len(points),
        'total_patients': len(set(patient_ids)),
        'splits': splits_data,
        'seed': seed,
        'parameters': {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'stratify': stratify
        }
    }
    
    combined_manifest_path = output_path / "splits_manifest.json"
    with open(combined_manifest_path, 'w') as f:
        json.dump(combined_manifest, f, indent=2)
    print(f"✅ Saved combined manifest to {combined_manifest_path}")
    
    # Create experiment directory structure
    print(f"\n📁 Creating experiment directories...")
    experiment_dirs = [
        'experiments/T0_leakage',
        'experiments/T1_detection_baseline',
        'experiments/T1_smoke',
        'experiments/T2_robustness',
        'experiments/T3_pinn_smoke',
        'experiments/T4_ablation',
        'experiments/T5_uncertainty_calibration'
    ]
    
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        exp_path.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder README
        readme_path = exp_path / "README.md"
        if not readme_path.exists():
            with open(readme_path, 'w') as f:
                f.write(f"# {exp_dir.replace('experiments/', '').replace('_', ' ')}\n\n")
                f.write("Experiment outputs will be saved here.\n")
        
        print(f"✅ {exp_dir}")
    
    # Save leakage verification
    leakage_report_path = output_path / "leakage_verification.txt"
    with open(leakage_report_path, 'w') as f:
        f.write(leakage_check['report'])
    print(f"✅ Leakage verification saved to {leakage_report_path}")
    
    return combined_manifest


def main():
    parser = argparse.ArgumentParser(
        description="Create patient-level data splits with manifests and leakage verification"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to preprocessed data HDF5 file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/manifests',
        help='Output directory for manifests'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training split ratio'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation split ratio'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='Test split ratio'
    )
    
    parser.add_argument(
        '--stratify',
        action='store_true',
        help='Stratify splits by label'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    # Create splits
    result = create_splits(
        args.input,
        args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        stratify=args.stratify,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("  SPLITTING COMPLETE")
    print("="*80)
    print(json.dumps(result, indent=2))
    print("="*80)


if __name__ == '__main__':
    main()
