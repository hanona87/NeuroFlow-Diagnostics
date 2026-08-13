"""Base dataset adapter interface for aneurysm datasets."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class DatasetMetadata:
    """Metadata for a dataset sample."""
    patient_id: str
    aneurysm_id: str
    source: str  # "intra", "synthetic", etc.
    file_path: str
    file_hash: str
    rupture_label: Optional[int] = None  # None = unlabeled, 0 = non-ruptured, 1 = ruptured
    modality: str = "3D mesh"  # "3D mesh", "point cloud", etc.
    notes: str = ""


class BaseDatasetAdapter(ABC):
    """
    Base class for dataset adapters.
    
    Adapters provide a uniform interface to load/preprocess different aneurysm datasets.
    Each adapter handles:
    - Dataset discovery and validation
    - Mesh loading and normalization
    - Metadata extraction (patient ID, rupture label, etc.)
    - Patient-level grouping for proper train/val/test splitting
    """
    
    def __init__(self, dataset_name: str, data_root: str):
        """
        Initialize adapter.
        
        Args:
            dataset_name: Name of the dataset (e.g., "intra", "synthetic")
            data_root: Root directory containing the dataset
        """
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.samples: List[DatasetMetadata] = []
    
    @abstractmethod
    def discover_samples(self) -> List[DatasetMetadata]:
        """
        Discover all available samples in the dataset.
        
        Returns:
            List of DatasetMetadata objects describing available samples
        """
        pass
    
    @abstractmethod
    def load_mesh(self, sample: DatasetMetadata) -> Tuple[Any, Any]:
        """
        Load mesh for a given sample.
        
        Args:
            sample: DatasetMetadata describing the sample
            
        Returns:
            Tuple of (vertices, faces) or mesh object
        """
        pass
    
    @abstractmethod
    def get_rupture_label(self, sample: DatasetMetadata) -> Optional[int]:
        """
        Get rupture label for a sample if available.
        
        Args:
            sample: DatasetMetadata
            
        Returns:
            None (unlabeled), 0 (non-ruptured), or 1 (ruptured)
        """
        pass
    
    def validate_dataset(self) -> Dict[str, Any]:
        """
        Validate dataset integrity.
        
        Returns:
            Dictionary with validation results
        """
        result = {
            'dataset': self.dataset_name,
            'total_samples': len(self.samples),
            'errors': [],
            'warnings': []
        }
        
        if len(self.samples) == 0:
            result['errors'].append("No samples discovered!")
        
        # Check for patient grouping
        patient_ids = [s.patient_id for s in self.samples]
        unique_patients = set(patient_ids)
        
        result['unique_patients'] = len(unique_patients)
        result['avg_samples_per_patient'] = len(self.samples) / max(len(unique_patients), 1)
        
        # Check for labels
        labeled_samples = sum(1 for s in self.samples if s.rupture_label is not None)
        result['labeled_samples'] = labeled_samples
        result['labeling_coverage'] = labeled_samples / max(len(self.samples), 1)
        
        return result
    
    def get_patient_groups(self) -> Dict[str, List[DatasetMetadata]]:
        """
        Group samples by patient.
        
        Returns:
            Dictionary mapping patient_id -> list of samples for that patient
        """
        groups = {}
        for sample in self.samples:
            if sample.patient_id not in groups:
                groups[sample.patient_id] = []
            groups[sample.patient_id].append(sample)
        return groups
    
    def save_manifest(self, output_path: str) -> None:
        """
        Save dataset manifest to JSON file.
        
        Args:
            output_path: Path to save manifest JSON
        """
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        manifest_data = {
            'dataset': self.dataset_name,
            'total_samples': len(self.samples),
            'samples': [
                {
                    'patient_id': s.patient_id,
                    'aneurysm_id': s.aneurysm_id,
                    'source': s.source,
                    'file_path': s.file_path,
                    'file_hash': s.file_hash,
                    'rupture_label': s.rupture_label,
                    'modality': s.modality,
                    'notes': s.notes
                }
                for s in self.samples
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
    
    def load_manifest(self, input_path: str) -> None:
        """
        Load dataset manifest from JSON file.
        
        Args:
            input_path: Path to manifest JSON
        """
        import os
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Manifest file not found: {input_path}")
        
        with open(input_path, 'r') as f:
            manifest_data = json.load(f)
        
        self.samples = [
            DatasetMetadata(
                patient_id=item['patient_id'],
                aneurysm_id=item['aneurysm_id'],
                source=item['source'],
                file_path=item['file_path'],
                file_hash=item['file_hash'],
                rupture_label=item.get('rupture_label'),
                modality=item.get('modality', '3D mesh'),
                notes=item.get('notes', '')
            )
            for item in manifest_data['samples']
        ]
