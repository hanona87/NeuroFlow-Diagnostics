"""
Preprocess meshes into standardized point cloud datasets.

This script:
1. Loads meshes from a dataset adapter (IntrA or synthetic)
2. Converts to fixed-size point clouds with normals
3. Preserves patient/rupture metadata
4. Saves to HDF5 files with manifests

Usage (with IntrA):
  python scripts/preprocess_datasets.py --dataset intra --data-root ./data/datasets/intra

Usage (synthetic fallback):
  python scripts/preprocess_datasets.py --dataset synthetic --n-patients 20

Usage (override output):
  python scripts/preprocess_datasets.py --dataset synthetic --output-dir ./data/processed_v2
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
from data.adapters import BaseDatasetAdapter, IntraAdapter, SyntheticAdapter
from data.preprocessing import PointCloudPreprocessor, PointCloudDatasetWriter
from utils import set_random_seed, compute_file_hash


def get_adapter(dataset: str, 
                data_root: Optional[str] = None,
                **kwargs) -> BaseDatasetAdapter:
    """
    Get appropriate dataset adapter.
    
    Args:
        dataset: Dataset name ("intra" or "synthetic")
        data_root: Root directory for dataset
        **kwargs: Additional arguments for synthetic adapter
        
    Returns:
        BaseDatasetAdapter instance
    """
    if dataset.lower() == "intra":
        if data_root is None:
            data_root = "./data/datasets/intra"
        adapter = IntraAdapter(data_root=data_root)
        
        # If IntrA not available, fall back to synthetic
        if not adapter.available:
            print(f"\n⚠️  {adapter.error_message}")
            print("→ Falling back to synthetic dataset for pipeline testing\n")
            adapter = SyntheticAdapter(**kwargs)
    
    elif dataset.lower() == "synthetic":
        adapter = SyntheticAdapter(**kwargs)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return adapter


def preprocess_sample(adapter: BaseDatasetAdapter,
                     sample: Any,
                     preprocessor: PointCloudPreprocessor,
                     num_points: int = 8192) -> Dict[str, Any]:
    """
    Preprocess a single sample.
    
    Args:
        adapter: Dataset adapter
        sample: Sample metadata
        preprocessor: PointCloudPreprocessor instance
        num_points: Target number of points
        
    Returns:
        Dictionary with processed point cloud and metadata
    """
    try:
        # Load mesh
        vertices, faces = adapter.load_mesh(sample)
        
        # Sample points from mesh
        preprocessor.num_points = num_points
        sampled_points = preprocessor.sample_points_from_mesh(vertices, faces)
        
        # Apply FPS
        from utils import fps
        if len(sampled_points) > num_points:
            indices = fps(sampled_points, num_points)
            points = sampled_points[indices]
        else:
            # Oversample if needed
            n_repeat = num_points // len(sampled_points) + 1
            points = np.tile(sampled_points, (n_repeat, 1))
            indices = fps(points, num_points)
            points = points[indices]
        
        # Normalize
        from utils import normalize_pc, compute_normals
        points, norm_params = normalize_pc(
            points,
            method=preprocessor.normalization_method,
            center_mode=preprocessor.center_mode
        )
        
        # Compute normals
        normals = compute_normals(points, k=20)
        
        # Stack points and normals
        points_with_normals = np.hstack([points, normals]).astype(np.float32)
        
        # Get rupture label
        rupture_label = adapter.get_rupture_label(sample)
        
        return {
            'points': points_with_normals,
            'label': rupture_label if rupture_label is not None else -1,  # -1 = unlabeled
            'patient_id': sample.patient_id,
            'aneurysm_id': sample.aneurysm_id,
            'normalization_method': preprocessor.normalization_method,
            'normalization_params': norm_params,
            'source': sample.source,
            'file_path': sample.file_path,
            'file_hash': sample.file_hash,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'patient_id': sample.patient_id,
            'aneurysm_id': sample.aneurysm_id,
            'success': False,
            'error': str(e),
            'points': None,
            'label': None
        }


def preprocess_dataset(adapter: BaseDatasetAdapter,
                      output_dir: str,
                      num_points: int = 8192,
                      seed: int = 42) -> Dict[str, Any]:
    """
    Preprocess entire dataset.
    
    Args:
        adapter: Dataset adapter
        output_dir: Output directory
        num_points: Target number of points
        seed: Random seed
        
    Returns:
        Dictionary with preprocessing results and statistics
    """
    set_random_seed(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = PointCloudPreprocessor(
        num_points=num_points,
        seed=seed
    )
    
    # Discover samples
    print(f"\n📊 Discovering {adapter.dataset_name} samples...")
    samples = adapter.discover_samples()
    print(f"✅ Found {len(samples)} samples from {len(adapter.get_patient_groups())} patients")
    
    # Validate dataset
    validation = adapter.validate_dataset()
    print(f"\n📈 Dataset validation:")
    for key, value in validation.items():
        if key not in ['dataset', 'errors', 'warnings']:
            print(f"   {key}: {value}")
    
    if validation['errors']:
        print("\n❌ Validation errors:")
        for error in validation['errors']:
            print(f"   - {error}")
    
    # Preprocess samples
    print(f"\n🔄 Preprocessing {len(samples)} samples...")
    processed_samples = []
    failed_samples = []
    
    for sample in tqdm(samples):
        result = preprocess_sample(adapter, sample, preprocessor, num_points)
        
        if result['success']:
            processed_samples.append(result)
        else:
            failed_samples.append(result)
    
    print(f"\n✅ Successfully processed: {len(processed_samples)}")
    print(f"❌ Failed: {len(failed_samples)}")
    
    if failed_samples:
        print("\nFailed samples:")
        for sample in failed_samples[:5]:  # Show first 5
            print(f"  - {sample['patient_id']}_{sample['aneurysm_id']}: {sample['error']}")
    
    # Write to HDF5
    print(f"\n💾 Writing to HDF5...")
    writer = PointCloudDatasetWriter(output_dir, compression=True)
    output_file = writer.write_split("full", processed_samples)
    print(f"✅ Saved to: {output_file}")
    
    # Save manifest
    print(f"\n📋 Creating manifest...")
    manifest_path = output_path / "manifest.json"
    adapter.save_manifest(str(manifest_path))
    print(f"✅ Manifest saved to: {manifest_path}")
    
    # Save preprocessing metadata
    metadata = {
        'dataset': adapter.dataset_name,
        'processed_samples': len(processed_samples),
        'failed_samples': len(failed_samples),
        'num_points': num_points,
        'normalization_method': preprocessor.normalization_method,
        'total_patients': len(adapter.get_patient_groups()),
        'output_file': output_file,
        'manifest_file': str(manifest_path),
        'seed': seed
    }
    
    metadata_path = output_path / "preprocessing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved to: {metadata_path}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess meshes into standardized point cloud datasets"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='synthetic',
        choices=['intra', 'synthetic'],
        help='Dataset to preprocess'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Root directory for dataset (for IntrA)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed',
        help='Output directory for processed data'
    )
    
    parser.add_argument(
        '--num-points',
        type=int,
        default=8192,
        help='Target number of points per point cloud'
    )
    
    parser.add_argument(
        '--n-patients',
        type=int,
        default=20,
        help='Number of patients (for synthetic dataset)'
    )
    
    parser.add_argument(
        '--samples-per-patient',
        type=int,
        default=2,
        help='Number of aneurysms per patient (for synthetic)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    args = parser.parse_args()
    
    # Get adapter
    adapter = get_adapter(
        args.dataset,
        data_root=args.data_root,
        n_patients=args.n_patients,
        samples_per_patient=args.samples_per_patient,
        seed=args.seed
    )
    
    # Preprocess dataset
    results = preprocess_dataset(
        adapter,
        args.output_dir,
        num_points=args.num_points,
        seed=args.seed
    )
    
    print("\n" + "="*80)
    print("  PREPROCESSING COMPLETE")
    print("="*80)
    print(json.dumps(results, indent=2))
    print("="*80)


if __name__ == '__main__':
    main()
