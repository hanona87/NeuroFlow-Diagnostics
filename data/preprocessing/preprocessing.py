"""
Data preprocessing pipeline for NeuroFlow-Diagnostics.
Handles mesh loading, point cloud extraction, normalization, and augmentation.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import h5py
from tqdm import tqdm

try:
    import trimesh
except ImportError:
    trimesh = None

try:
    import pyvista as pv
except ImportError:
    pv = None

from utils import (
    fps, compute_normals, normalize_pc, compute_file_hash,
    split_by_patient, set_random_seed
)


@dataclass
class PointCloudMetadata:
    """Metadata for a point cloud sample."""
    patient_id: str
    aneurysm_id: str
    num_points: int
    normalization_method: str
    normalization_params: Dict[str, Any]
    has_normals: bool
    label: Optional[int] = None  # None for unlabeled, 0/1 for binary labels
    source_file: str = ""


class PointCloudPreprocessor:
    """
    Preprocessing pipeline for converting meshes to point clouds.
    """
    
    def __init__(self, 
                 num_points: int = 8192,
                 normalization_method: str = "unit_sphere",
                 center_mode: str = "centroid",
                 compute_normals_flag: bool = True,
                 seed: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            num_points: Target number of points for FPS
            normalization_method: Method for normalizing point clouds
            center_mode: How to center the point cloud
            compute_normals_flag: Whether to compute normals
            seed: Random seed for reproducibility
        """
        self.num_points = num_points
        self.normalization_method = normalization_method
        self.center_mode = center_mode
        self.compute_normals_flag = compute_normals_flag
        self.seed = seed
        set_random_seed(seed)
    
    def load_mesh(self, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh from file (STL, OBJ, or VTK).
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Tuple of (vertices, faces)
        """
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
        extension = Path(mesh_path).suffix.lower()
        
        if trimesh is not None:
            # Use trimesh (more robust)
            mesh = trimesh.load(mesh_path, force='mesh')
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces
            
            # Check and repair mesh if needed
            if not mesh.is_watertight:
                # Try to fill small holes
                mesh.fill_holes()
            
            # Remove disconnected components (keep largest)
            if len(mesh.split()) > 1:
                meshes = mesh.split()
                mesh = max(meshes, key=lambda m: len(m.vertices))
                vertices = mesh.vertices.astype(np.float32)
                faces = mesh.faces
            
            return vertices, faces
        
        elif pv is not None:
            # Fallback to PyVista
            mesh = pv.read(mesh_path)
            vertices = mesh.points.astype(np.float32)
            faces = mesh.faces.reshape(-1, 4)[:, 1:]  # Remove count column
            return vertices, faces
        
        else:
            raise ImportError("Please install trimesh or pyvista for mesh loading")
    
    def sample_points_from_mesh(self, vertices: np.ndarray, 
                                faces: np.ndarray) -> np.ndarray:
        """
        Sample points from mesh surface.
        
        Args:
            vertices: Mesh vertices (N_v, 3)
            faces: Mesh faces (N_f, 3)
            
        Returns:
            Sampled points (num_samples, 3)
        """
        if trimesh is not None:
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            # Sample more than needed to account for duplicates
            sample_count = min(self.num_points * 2, max(100000, len(vertices) * 10))
            points = mesh.sample(sample_count)
        else:
            # Simple uniform sampling from faces
            n_samples = self.num_points * 2
            face_indices = np.random.choice(len(faces), n_samples, replace=True)
            
            # Random barycentric coordinates
            r1 = np.random.rand(n_samples, 1)
            r2 = np.random.rand(n_samples, 1)
            
            # Ensure they sum to 1
            mask = (r1 + r2) > 1
            r1[mask] = 1 - r1[mask]
            r2[mask] = 1 - r2[mask]
            
            # Sample points
            v1 = vertices[faces[face_indices, 0]]
            v2 = vertices[faces[face_indices, 1]]
            v3 = vertices[faces[face_indices, 2]]
            
            points = (1 - r1 - r2) * v1 + r1 * v2 + r2 * v3
        
        return points.astype(np.float32)
    
    def process_mesh(self, mesh_path: str) -> np.ndarray:
        """
        Process mesh file and return preprocessed point cloud.
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Processed point cloud (num_points, 3 or 6)
        """
        # Load mesh
        vertices, faces = self.load_mesh(mesh_path)
        
        # Sample points
        points = self.sample_points_from_mesh(vertices, faces)
        
        # Apply FPS to get exact number of points
        if len(points) > self.num_points:
            indices = fps(points, self.num_points)
            points = points[indices]
        else:
            # Oversample if not enough points
            n_repeat = self.num_points // len(points) + 1
            points = np.tile(points, (n_repeat, 1))
            indices = fps(points, self.num_points)
            points = points[indices]
        
        # Normalize
        points, norm_params = normalize_pc(
            points,
            method=self.normalization_method,
            center_mode=self.center_mode
        )
        
        # Compute normals if requested
        if self.compute_normals_flag:
            normals = compute_normals(points, k=20)
            output = np.hstack([points, normals]).astype(np.float32)
        else:
            output = points.astype(np.float32)
        
        return output, norm_params
    
    def create_data_manifest(self, data_list: List[Dict[str, Any]]) -> Dict[str, Dict]:
        """
        Create a manifest of all processed data.
        
        Args:
            data_list: List of data dictionaries
            
        Returns:
            Manifest dictionary with file hashes and metadata
        """
        manifest = {}
        
        for item in data_list:
            patient_id = item['patient_id']
            aneurysm_id = item.get('aneurysm_id', f"{patient_id}_a1")
            
            key = f"{patient_id}_{aneurysm_id}"
            manifest[key] = {
                'patient_id': patient_id,
                'aneurysm_id': aneurysm_id,
                'label': item.get('label', None),
                'source_file': item.get('source_file', ''),
                'processed': False
            }
        
        return manifest


class PointCloudDatasetWriter:
    """
    Write point cloud datasets to HDF5 files.
    """
    
    def __init__(self, output_dir: str, compression: bool = True, 
                 compression_level: int = 4):
        """
        Initialize dataset writer.
        
        Args:
            output_dir: Output directory for HDF5 files
            compression: Whether to compress data
            compression_level: Compression level (0-9)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.compression_level = compression_level
    
    def write_split(self, split_name: str, 
                    samples: List[Dict[str, Any]]) -> str:
        """
        Write a dataset split to HDF5 file.
        
        Args:
            split_name: Name of split (train, val, test)
            samples: List of sample dictionaries
            
        Returns:
            Path to written HDF5 file
        """
        output_path = self.output_dir / f"{split_name}.h5"
        
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            n_samples = len(samples)
            n_points = samples[0]['points'].shape[0]
            n_features = samples[0]['points'].shape[1]
            
            points_dset = f.create_dataset(
                'points',
                shape=(n_samples, n_points, n_features),
                dtype=np.float32,
                compression='gzip' if self.compression else None,
                compression_opts=self.compression_level if self.compression else None
            )
            
            labels_dset = f.create_dataset(
                'labels',
                shape=(n_samples,),
                dtype=np.int32,
                compression='gzip' if self.compression else None,
                compression_opts=self.compression_level if self.compression else None
            )
            
            # Create string datasets
            patient_ids_dset = f.create_dataset(
                'patient_ids',
                shape=(n_samples,),
                dtype=h5py.special_dtype(vlen=str)
            )
            
            aneurysm_ids_dset = f.create_dataset(
                'aneurysm_ids',
                shape=(n_samples,),
                dtype=h5py.special_dtype(vlen=str)
            )
            
            # Write data
            for i, sample in enumerate(samples):
                points_dset[i] = sample['points']
                labels_dset[i] = sample['label']
                patient_ids_dset[i] = sample['patient_id']
                aneurysm_ids_dset[i] = sample['aneurysm_id']
            
            # Store attributes
            f.attrs['split'] = split_name
            f.attrs['num_samples'] = n_samples
            f.attrs['num_points'] = n_points
            f.attrs['num_features'] = n_features
            f.attrs['normalization_method'] = samples[0].get('normalization_method', 'unit_sphere')
        
        return str(output_path)


class PointCloudDataset:
    """
    PyTorch-compatible dataset for point cloud data.
    """
    
    def __init__(self, h5_path: str, augmentation: bool = False,
                 augmentation_config: Optional[Dict] = None):
        """
        Initialize dataset.
        
        Args:
            h5_path: Path to HDF5 file
            augmentation: Whether to apply augmentation
            augmentation_config: Augmentation configuration dict
        """
        self.h5_path = h5_path
        self.augmentation = augmentation
        self.augmentation_config = augmentation_config or {}
        
        # Load metadata from HDF5
        with h5py.File(h5_path, 'r') as f:
            self.num_samples = f.attrs['num_samples']
            self.num_points = f.attrs['num_points']
            self.num_features = f.attrs['num_features']
    
    def __len__(self) -> int:
        """Return dataset size."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'points', 'label', 'patient_id', 'aneurysm_id'
        """
        with h5py.File(self.h5_path, 'r') as f:
            points = f['points'][idx].astype(np.float32)
            label = f['labels'][idx]
            patient_id = f['patient_ids'][idx]
            aneurysm_id = f['aneurysm_ids'][idx]
        
        # Apply augmentation
        if self.augmentation:
            points = self._augment_points(points)
        
        return {
            'points': points,
            'label': label,
            'patient_id': patient_id,
            'aneurysm_id': aneurysm_id
        }
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to point cloud.
        
        Args:
            points: Input point cloud (N, 3+)
            
        Returns:
            Augmented point cloud
        """
        points = points.copy()
        
        # Rotation
        if self.augmentation_config.get('rotation_enabled', True):
            angle = np.random.rand() * np.pi * 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            points[:, :3] = points[:, :3] @ rotation_matrix.T
        
        # Jitter
        if self.augmentation_config.get('jitter_enabled', True):
            jitter_std = self.augmentation_config.get('jitter_std', 0.01)
            points[:, :3] += np.random.randn(points.shape[0], 3) * jitter_std
        
        # Point dropout
        if self.augmentation_config.get('dropout_enabled', True):
            dropout_rate = self.augmentation_config.get('dropout_rate', 0.1)
            n_keep = int(len(points) * (1 - dropout_rate))
            keep_indices = np.random.choice(len(points), n_keep, replace=False)
            points = points[keep_indices]
            
            # Resample to original size
            if len(points) < len(points):
                n_resample = len(points) - len(points)
                resample_indices = np.random.choice(len(points), n_resample, replace=True)
                points = np.vstack([points, points[resample_indices]])
        
        # Scaling
        if self.augmentation_config.get('scaling_enabled', True):
            scaling_range = self.augmentation_config.get('scaling_range', [0.9, 1.1])
            scale = np.random.uniform(scaling_range[0], scaling_range[1])
            points[:, :3] *= scale
        
        return points.astype(np.float32)


def create_synthetic_dataset(n_samples: int = 100, 
                             n_positive: int = 50,
                             num_points: int = 8192,
                             output_dir: str = './data/synthetic') -> str:
    """
    Create a synthetic dataset for testing and development.
    
    Args:
        n_samples: Total number of samples
        n_positive: Number of positive (aneurysm) samples
        num_points: Number of points per sample
        output_dir: Output directory
        
    Returns:
        Path to created HDF5 file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic data
    samples = []
    
    for i in range(n_samples):
        # Generate random point cloud
        if i < n_positive:
            # Aneurysm: cluster of points with aneurysm bulge
            base_points = np.random.randn(num_points // 2, 3) * 0.05
            bulge_points = np.random.randn(num_points // 2, 3) * 0.05 + np.array([0.1, 0, 0])
            points = np.vstack([base_points, bulge_points]).astype(np.float32)
            label = 1
        else:
            # Normal vasculature: elongated structure
            t = np.linspace(0, 4 * np.pi, num_points)
            points = np.column_stack([
                np.cos(t) * 0.05,
                np.sin(t) * 0.05,
                np.linspace(-0.1, 0.1, num_points)
            ]).astype(np.float32)
            label = 0
        
        # Add normals
        normals = np.random.randn(num_points, 3)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        
        points_with_normals = np.hstack([points, normals]).astype(np.float32)
        
        samples.append({
            'points': points_with_normals,
            'label': label,
            'patient_id': f'synthetic_{i//10}',
            'aneurysm_id': f'a_{i % 10}',
            'normalization_method': 'unit_sphere'
        })
    
    # Write to HDF5
    output_path = Path(output_dir) / 'synthetic_data.h5'
    
    with h5py.File(output_path, 'w') as f:
        n_samples = len(samples)
        n_points = samples[0]['points'].shape[0]
        n_features = samples[0]['points'].shape[1]
        
        points_dset = f.create_dataset(
            'points',
            shape=(n_samples, n_points, n_features),
            dtype=np.float32,
            compression='gzip'
        )
        
        labels_dset = f.create_dataset(
            'labels',
            shape=(n_samples,),
            dtype=np.int32
        )
        
        patient_ids_dset = f.create_dataset(
            'patient_ids',
            shape=(n_samples,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        aneurysm_ids_dset = f.create_dataset(
            'aneurysm_ids',
            shape=(n_samples,),
            dtype=h5py.special_dtype(vlen=str)
        )
        
        for i, sample in enumerate(samples):
            points_dset[i] = sample['points']
            labels_dset[i] = sample['label']
            patient_ids_dset[i] = sample['patient_id']
            aneurysm_ids_dset[i] = sample['aneurysm_id']
        
        f.attrs['split'] = 'full'
        f.attrs['num_samples'] = n_samples
        f.attrs['num_points'] = n_points
        f.attrs['num_features'] = n_features
    
    return str(output_path)
