"""Adapter for synthetic dataset (for testing and fallback)."""

import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from .base import BaseDatasetAdapter, DatasetMetadata


class SyntheticAdapter(BaseDatasetAdapter):
    """
    Adapter for synthetic dataset.
    
    Generates synthetic aneurysm point clouds for:
    - Pipeline testing and development
    - Fallback when real datasets are unavailable
    - Smoke testing and benchmarking
    
    Synthetic samples follow the same patient-level structure as real data,
    ensuring that evaluation protocols are identical even when using synthetic data.
    """
    
    def __init__(self, 
                 n_patients: int = 10,
                 samples_per_patient: int = 2,
                 rupture_prevalence: float = 0.3,
                 seed: int = 42):
        """
        Initialize synthetic dataset adapter.
        
        Args:
            n_patients: Number of unique patients to generate
            samples_per_patient: Number of aneurysms per patient
            rupture_prevalence: Fraction of samples with rupture label
            seed: Random seed for reproducibility
        """
        super().__init__("synthetic", "./data/synthetic")
        
        self.n_patients = n_patients
        self.samples_per_patient = samples_per_patient
        self.rupture_prevalence = rupture_prevalence
        self.seed = seed
        
        np.random.seed(seed)
        self.discover_samples()
    
    def discover_samples(self) -> List[DatasetMetadata]:
        """
        Generate synthetic samples (no real files).
        
        Returns:
            List of DatasetMetadata for synthetic samples
        """
        self.samples = []
        
        for p_idx in range(self.n_patients):
            patient_id = f"synthetic_patient_{p_idx:03d}"
            
            for a_idx in range(self.samples_per_patient):
                aneurysm_id = f"aneurysm_{a_idx:02d}"
                
                # Assign rupture labels
                if np.random.rand() < self.rupture_prevalence:
                    rupture_label = 1
                else:
                    rupture_label = 0
                
                sample = DatasetMetadata(
                    patient_id=patient_id,
                    aneurysm_id=aneurysm_id,
                    source="synthetic",
                    file_path=f"synthetic://{patient_id}/{aneurysm_id}",  # Virtual path
                    file_hash=f"synthetic_{p_idx}_{a_idx}",
                    rupture_label=rupture_label,
                    modality="synthetic point cloud",
                    notes="Generated for testing/fallback"
                )
                
                self.samples.append(sample)
        
        return self.samples
    
    def load_mesh(self, sample: DatasetMetadata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic mesh for a sample.
        
        Args:
            sample: DatasetMetadata
            
        Returns:
            Tuple of (vertices, faces) - synthetic geometry
        """
        np.random.seed(hash(sample.patient_id + sample.aneurysm_id) % 2**32)
        
        # Determine geometry type based on label
        if sample.rupture_label == 1:
            # Aneurysm: cluster of points with bulge
            n_vessel_points = 500
            n_bulge_points = 300
            
            # Vessel: elongated cylinder
            t = np.linspace(0, 4 * np.pi, n_vessel_points)
            vessel = np.column_stack([
                np.cos(t) * 0.05,
                np.sin(t) * 0.05,
                np.linspace(-0.15, 0.15, n_vessel_points)
            ])
            
            # Aneurysm bulge: sphere at z=0
            phi = np.random.rand(n_bulge_points) * 2 * np.pi
            theta = np.random.rand(n_bulge_points) * np.pi
            r = np.random.rand(n_bulge_points) * 0.08
            
            bulge = np.column_stack([
                r * np.sin(theta) * np.cos(phi) + 0.08,
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta)
            ])
            
            vertices = np.vstack([vessel, bulge]).astype(np.float32)
        
        else:
            # Normal vessel: elongated structure
            n_points = 800
            t = np.linspace(0, 4 * np.pi, n_points)
            
            vertices = np.column_stack([
                np.cos(t) * 0.05,
                np.sin(t) * 0.05,
                np.linspace(-0.15, 0.15, n_points)
            ]).astype(np.float32)
        
        # Create minimal face structure (not used for point cloud sampling)
        faces = np.array([[0, 1, 2]], dtype=np.uint32)
        
        return vertices, faces
    
    def get_rupture_label(self, sample: DatasetMetadata) -> Optional[int]:
        """
        Get rupture label for synthetic sample.
        
        Args:
            sample: DatasetMetadata
            
        Returns:
            0 (normal) or 1 (ruptured)
        """
        return sample.rupture_label
