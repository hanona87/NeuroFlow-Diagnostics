"""Adapter for IntrA dataset (cerebral aneurysm imaging)."""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np

from .base import BaseDatasetAdapter, DatasetMetadata


class IntraAdapter(BaseDatasetAdapter):
    """
    Adapter for IntrA dataset.
    
    IntrA is a public dataset of cerebral aneurysm images from Université de Strasbourg.
    Dataset: https://github.com/rjdmoore/IntrA
    
    Note: This adapter will gracefully degrade to synthetic data if IntrA is not available.
    This ensures the pipeline always works even without the real dataset.
    """
    
    def __init__(self, data_root: str = "./data/datasets/intra"):
        """
        Initialize IntrA adapter.
        
        Args:
            data_root: Root directory for IntrA dataset
        """
        super().__init__("intra", data_root)
        self.available = False
        self.error_message = None
        
        # Try to discover samples
        try:
            self._validate_dataset_presence()
            self.discover_samples()
            self.available = True
        except Exception as e:
            self.available = False
            self.error_message = str(e)
            print(f"⚠️  IntrA dataset not available: {e}")
            print("   → Synthetic fallback will be used for preprocessing pipeline")
    
    def _validate_dataset_presence(self) -> bool:
        """
        Check if IntrA dataset exists at expected location.
        
        Returns:
            True if dataset directory exists and has expected structure
            
        Raises:
            FileNotFoundError if dataset is not found
        """
        data_path = Path(self.data_root)
        
        if not data_path.exists():
            raise FileNotFoundError(
                f"IntrA dataset directory not found: {self.data_root}\n"
                "To use real data, download IntrA from:\n"
                "  https://github.com/rjdmoore/IntrA\n"
                "  Extract to: {}\n"
                "  Expected structure:\n"
                "    intra/\n"
                "    ├── images/          (CT/MR scans)\n"
                "    ├── segmentations/   (vessel segmentations)\n"
                "    ├── surfaces/        (3D mesh files)\n"
                "    └── metadata.json    (patient metadata)\n"
                "Until then, synthetic data will be used.".format(self.data_root)
            )
        
        # Check for key subdirectories
        required_dirs = ['images', 'segmentations', 'surfaces']
        present_dirs = [d for d in required_dirs if (data_path / d).exists()]
        
        if not present_dirs:
            raise FileNotFoundError(
                f"IntrA dataset structure incomplete at {self.data_root}\n"
                f"Found subdirectories: {list(data_path.iterdir())}\n"
                f"Expected to find at least one of: {required_dirs}"
            )
        
        return True
    
    def discover_samples(self) -> List[DatasetMetadata]:
        """
        Discover all IntrA samples.
        
        Returns:
            List of DatasetMetadata for available samples
        """
        if not self.available:
            return []
        
        self.samples = []
        surfaces_path = Path(self.data_root) / 'surfaces'
        
        # Look for mesh files (STL, OBJ, VTK)
        mesh_extensions = ['.stl', '.obj', '.vtk', '.vtp']
        
        for mesh_file in surfaces_path.glob('*'):
            if mesh_file.suffix.lower() in mesh_extensions:
                # Extract patient/aneurysm IDs from filename
                # Assuming naming convention: patient_<pid>_aneurysm_<aid>.stl
                stem = mesh_file.stem
                
                patient_id = self._extract_patient_id(stem)
                aneurysm_id = self._extract_aneurysm_id(stem)
                
                # Compute file hash
                file_hash = self._compute_file_hash(str(mesh_file))
                
                # Get rupture label if metadata available
                rupture_label = self._get_rupture_label_from_metadata(patient_id)
                
                sample = DatasetMetadata(
                    patient_id=patient_id,
                    aneurysm_id=aneurysm_id,
                    source="intra",
                    file_path=str(mesh_file),
                    file_hash=file_hash,
                    rupture_label=rupture_label,
                    modality="3D mesh"
                )
                
                self.samples.append(sample)
        
        return self.samples
    
    def _extract_patient_id(self, filename: str) -> str:
        """Extract patient ID from filename."""
        # Simple heuristic: look for 'patient_' or use first component
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() == 'patient' and i + 1 < len(parts):
                return f"patient_{parts[i+1]}"
        
        # Fallback: use full stem
        return f"patient_{filename}"
    
    def _extract_aneurysm_id(self, filename: str) -> str:
        """Extract aneurysm ID from filename."""
        # Look for 'aneurysm_' pattern
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.lower() == 'aneurysm' and i + 1 < len(parts):
                return f"aneurysm_{parts[i+1]}"
        
        # Fallback
        return f"aneurysm_{filename[-2:]}"
    
    def _compute_file_hash(self, filepath: str, algorithm: str = 'sha256') -> str:
        """Compute file hash for integrity verification."""
        import hashlib
        
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            while True:
                chunk = f.read(8192)
                if not chunk:
                    break
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _get_rupture_label_from_metadata(self, patient_id: str) -> Optional[int]:
        """
        Get rupture label from metadata file if available.
        
        Returns:
            None (unlabeled), 0 (non-ruptured), or 1 (ruptured)
        """
        metadata_path = Path(self.data_root) / 'metadata.json'
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            if patient_id in metadata:
                return metadata[patient_id].get('rupture_label')
        except (json.JSONDecodeError, KeyError):
            pass
        
        return None
    
    def load_mesh(self, sample: DatasetMetadata) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh for IntrA sample.
        
        Args:
            sample: DatasetMetadata describing the sample
            
        Returns:
            Tuple of (vertices, faces)
        """
        if not os.path.exists(sample.file_path):
            raise FileNotFoundError(f"Mesh file not found: {sample.file_path}")
        
        # Use trimesh if available (more robust)
        try:
            import trimesh
            mesh = trimesh.load(sample.file_path, force='mesh')
            vertices = mesh.vertices.astype(np.float32)
            faces = mesh.faces
            
            # Repair if needed
            if not mesh.is_watertight:
                mesh.fill_holes()
            
            # Handle multiple components
            if len(mesh.split()) > 1:
                meshes = mesh.split()
                mesh = max(meshes, key=lambda m: len(m.vertices))
                vertices = mesh.vertices.astype(np.float32)
                faces = mesh.faces
            
            return vertices, faces
        
        except ImportError:
            # Fallback to PyVista
            try:
                import pyvista as pv
                mesh = pv.read(sample.file_path)
                vertices = mesh.points.astype(np.float32)
                faces = mesh.faces.reshape(-1, 4)[:, 1:]
                return vertices, faces
            except ImportError:
                raise ImportError(
                    "Please install trimesh or pyvista to load mesh files:\n"
                    "  pip install trimesh pyvista"
                )
    
    def get_rupture_label(self, sample: DatasetMetadata) -> Optional[int]:
        """
        Get rupture label for a sample.
        
        Args:
            sample: DatasetMetadata
            
        Returns:
            None (unlabeled), 0 (non-ruptured), or 1 (ruptured)
        """
        return sample.rupture_label
