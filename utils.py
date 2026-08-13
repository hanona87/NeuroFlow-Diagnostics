"""
Utility functions for NeuroFlow-Diagnostics project.
Includes seeding, device management, path utilities, and common helpers.
"""

import os
import sys
import random
import hashlib
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic algorithms (may be slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'  # For CUDA >= 11
    else:
        cudnn.benchmark = True


def get_device(device_str: str = "cuda", gpu_ids: Optional[List[int]] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device_str: "cuda" or "cpu"
        gpu_ids: List of GPU IDs to use
        
    Returns:
        torch.device object
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def setup_logging(log_file: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (if None, logs to console only)
        level: Logging level
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger("neuroflow")
    logger.setLevel(getattr(logging, level))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_directories(base_path: str, subdirs: List[str]) -> Dict[str, Path]:
    """
    Create multiple directories and return their paths.
    
    Args:
        base_path: Base directory path
        subdirs: List of subdirectory names
        
    Returns:
        Dictionary mapping subdirectory names to Path objects
    """
    paths = {}
    for subdir in subdirs:
        path = Path(base_path) / subdir
        path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = path
    
    return paths


def compute_file_hash(file_path: str, algorithm: str = 'sha256', chunk_size: int = 8192) -> str:
    """
    Compute hash of a file for integrity verification.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        chunk_size: Chunk size for reading large files
        
    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_data_manifest(data_dir: str, format: str = "h5") -> Dict[str, Dict[str, str]]:
    """
    Compute manifest of all data files with their hashes.
    
    Args:
        data_dir: Directory containing data files
        format: File format to include
        
    Returns:
        Dictionary mapping filenames to their metadata (path, hash, size)
    """
    manifest = {}
    
    data_path = Path(data_dir)
    for file_path in data_path.glob(f"*.{format}"):
        file_hash = compute_file_hash(str(file_path))
        file_size = os.path.getsize(file_path)
        
        manifest[file_path.name] = {
            "path": str(file_path),
            "hash": file_hash,
            "size": file_size,
            "format": format
        }
    
    return manifest


def normalize_pc(pc: np.ndarray, method: str = "unit_sphere", 
                 center_mode: str = "centroid") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize point cloud using specified method.
    
    Args:
        pc: Point cloud array (N, 3)
        method: Normalization method ('unit_sphere', 'bbox', 'z_score')
        center_mode: Centering method ('centroid', 'origin')
        
    Returns:
        Normalized point cloud and normalization parameters dict
    """
    pc = pc.copy().astype(np.float32)
    norm_params = {}
    
    # Center the point cloud
    if center_mode == "centroid":
        centroid = pc.mean(axis=0)
        pc = pc - centroid
        norm_params['centroid'] = centroid
    elif center_mode == "origin":
        norm_params['centroid'] = np.array([0, 0, 0])
    
    # Apply normalization
    if method == "unit_sphere":
        # Scale to unit sphere
        max_dist = np.sqrt((pc ** 2).sum(axis=1)).max()
        if max_dist > 0:
            pc = pc / max_dist
        norm_params['scale'] = max_dist
    
    elif method == "bbox":
        # Normalize to [-1, 1] bounding box
        pc_min = pc.min(axis=0)
        pc_max = pc.max(axis=0)
        pc_range = pc_max - pc_min
        pc = 2 * (pc - pc_min) / (pc_range + 1e-8) - 1
        norm_params['bbox_min'] = pc_min
        norm_params['bbox_range'] = pc_range
    
    elif method == "z_score":
        # Z-score normalization
        pc_std = pc.std(axis=0)
        pc = pc / (pc_std + 1e-8)
        norm_params['std'] = pc_std
    
    norm_params['method'] = method
    norm_params['center_mode'] = center_mode
    
    return pc, norm_params


def denormalize_pc(pc: np.ndarray, norm_params: Dict[str, Any]) -> np.ndarray:
    """
    Denormalize a point cloud using stored normalization parameters.
    
    Args:
        pc: Normalized point cloud (N, 3)
        norm_params: Normalization parameters dict
        
    Returns:
        Denormalized point cloud
    """
    pc = pc.copy().astype(np.float32)
    method = norm_params.get('method', 'unit_sphere')
    
    # Reverse normalization
    if method == "unit_sphere":
        scale = norm_params.get('scale', 1.0)
        pc = pc * scale
    
    elif method == "bbox":
        bbox_min = norm_params.get('bbox_min')
        bbox_range = norm_params.get('bbox_range')
        pc = (pc + 1) / 2 * bbox_range + bbox_min
    
    elif method == "z_score":
        std = norm_params.get('std', 1.0)
        pc = pc * std
    
    # Reverse centering
    centroid = norm_params.get('centroid', np.array([0, 0, 0]))
    pc = pc + centroid
    
    return pc


def fps(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) on point cloud.
    
    Args:
        points: Point cloud array (N, 3)
        num_samples: Number of samples to select
        
    Returns:
        Indices of selected points (num_samples,)
    """
    N = points.shape[0]
    
    if num_samples >= N:
        return np.arange(N)
    
    # Initialize
    selected_indices = np.zeros(num_samples, dtype=np.int32)
    distances = np.full(N, np.inf)
    
    # Random first point
    selected_indices[0] = np.random.randint(0, N)
    distances[selected_indices[0]] = 0
    
    # Select remaining points
    for i in range(1, num_samples):
        # Find farthest point
        farthest_idx = np.argmax(distances)
        selected_indices[i] = farthest_idx
        
        # Update distances
        farthest_point = points[farthest_idx]
        distances_to_farthest = np.linalg.norm(
            points - farthest_point, axis=1
        )
        distances = np.minimum(distances, distances_to_farthest)
    
    return selected_indices


def compute_normals(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Compute surface normals for point cloud using k-NN.
    
    Args:
        points: Point cloud array (N, 3)
        k: Number of neighbors for normal estimation
        
    Returns:
        Surface normals (N, 3)
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float32)
    
    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    # Compute normals using PCA on neighbors
    for i in range(N):
        neighbors = points[indices[i]]
        
        # Center neighbors
        centered = neighbors - neighbors.mean(axis=0)
        
        # SVD
        U, _, _ = np.linalg.svd(centered.T @ centered)
        
        # Normal is the direction with smallest variance
        normals[i] = U[:, -1]
    
    return normals


def split_by_patient(patient_ids: List[str], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data by patient (ensure no data leakage).
    
    Args:
        patient_ids: List of patient IDs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        
    Returns:
        Tuple of (train_patients, val_patients, test_patients)
    """
    unique_patients = sorted(set(patient_ids))
    n_patients = len(unique_patients)
    
    np.random.seed(seed)
    indices = np.random.permutation(n_patients)
    
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)
    
    train_patients = [unique_patients[i] for i in indices[:n_train]]
    val_patients = [unique_patients[i] for i in indices[n_train:n_train+n_val]]
    test_patients = [unique_patients[i] for i in indices[n_train+n_val:]]
    
    return train_patients, val_patients, test_patients


def check_split_leakage(train_patients: List[str],
                        val_patients: List[str],
                        test_patients: List[str]) -> Dict[str, Any]:
    """
    Check for patient-level data leakage across train/val/test splits.
    
    Ensures that no patient appears in more than one split (critical for medical ML).
    
    Args:
        train_patients: List of training patient IDs
        val_patients: List of validation patient IDs
        test_patients: List of test patient IDs
        
    Returns:
        Dictionary with:
            'has_leakage': bool, whether any leakage was detected
            'train_val_overlap': set of overlapping patient IDs
            'train_test_overlap': set of overlapping patient IDs
            'val_test_overlap': set of overlapping patient IDs
            'report': str, formatted report
    """
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    has_leakage = bool(train_val_overlap or train_test_overlap or val_test_overlap)
    
    report = []
    report.append("\n" + "="*80)
    report.append("  PATIENT-LEVEL LEAKAGE CHECK")
    report.append("="*80)
    
    report.append(f"  Train patients: {len(train_patients)}")
    report.append(f"  Val patients: {len(val_patients)}")
    report.append(f"  Test patients: {len(test_patients)}")
    
    if has_leakage:
        report.append("\n  ⚠️  LEAKAGE DETECTED:")
        if train_val_overlap:
            report.append(f"    Train-Val overlap: {train_val_overlap}")
        if train_test_overlap:
            report.append(f"    Train-Test overlap: {train_test_overlap}")
        if val_test_overlap:
            report.append(f"    Val-Test overlap: {val_test_overlap}")
    else:
        report.append("\n  ✅ NO LEAKAGE DETECTED (splits are clean)")
    
    report.append("="*80 + "\n")
    
    return {
        'has_leakage': has_leakage,
        'train_val_overlap': train_val_overlap,
        'train_test_overlap': train_test_overlap,
        'val_test_overlap': val_test_overlap,
        'report': "\n".join(report)
    }


def print_summary(title: str, content: Dict[str, Any]) -> None:
    """
    Print a formatted summary.
    
    Args:
        title: Summary title
        content: Dictionary of content to print
    """
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)
    
    for key, value in content.items():
        print(f"  {key:.<50} {value}")
    
    print("="*80 + "\n")
