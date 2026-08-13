"""
Patient-level deterministic splitting for dataset manifests.

Ensures:
- No data leakage (patients never split across train/val/test)
- Reproducibility (deterministic with seed)
- Stratification by rupture status when needed
- Proper grouping of multiple aneurysms per patient
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from .manifest import DatasetManifest, ManifestEntry, DuplicateDetector


class PatientLevelSplitter:
    """Performs patient-level train/val/test splitting without leakage."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
    
    def split_manifest(
        self,
        manifest: DatasetManifest,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_by_rupture: bool = True,
        stratify_by_site: bool = False
    ) -> Tuple[DatasetManifest, DatasetManifest, DatasetManifest]:
        """
        Split a manifest into train/val/test at patient level.
        
        Args:
            manifest: Source manifest to split
            train_ratio: Fraction for training (default 0.7)
            val_ratio: Fraction for validation (default 0.15)
            test_ratio: Fraction for testing (default 0.15)
            stratify_by_rupture: If True, stratify by rupture status
            stratify_by_site: If True, stratify by site (secondary)
            
        Returns:
            (train_manifest, val_manifest, test_manifest)
        """
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0; got {total}")
        
        # Group entries by patient
        patient_groups = self._group_by_patient(manifest)
        
        # Get patient IDs and their rupture status
        patients = list(patient_groups.keys())
        patient_rupture_status = self._get_patient_rupture_status(manifest, patient_groups)
        
        # Stratified split if requested
        if stratify_by_rupture:
            train_patients, val_patients, test_patients = self._stratified_split(
                patients,
                patient_rupture_status,
                train_ratio,
                val_ratio,
                test_ratio
            )
        else:
            train_patients, val_patients, test_patients = self._random_split(
                patients,
                train_ratio,
                val_ratio,
                test_ratio
            )
        
        # Create new manifests
        train_manifest = self._create_manifest_from_patients(
            manifest, patient_groups, train_patients, "train"
        )
        val_manifest = self._create_manifest_from_patients(
            manifest, patient_groups, val_patients, "validation"
        )
        test_manifest = self._create_manifest_from_patients(
            manifest, patient_groups, test_patients, "test"
        )
        
        return train_manifest, val_manifest, test_manifest
    
    @staticmethod
    def _group_by_patient(manifest: DatasetManifest) -> Dict[str, List[ManifestEntry]]:
        """Group manifest entries by patient ID."""
        groups = {}
        
        for entry in manifest.entries:
            if entry.patient_id not in groups:
                groups[entry.patient_id] = []
            groups[entry.patient_id].append(entry)
        
        return groups
    
    @staticmethod
    def _get_patient_rupture_status(
        manifest: DatasetManifest,
        patient_groups: Dict[str, List[ManifestEntry]]
    ) -> Dict[str, Optional[int]]:
        """
        Get rupture status for each patient.
        
        When a patient has multiple aneurysms with different labels,
        use majority vote.
        """
        patient_status = {}
        
        for patient_id, entries in patient_groups.items():
            # Get all rupture labels for this patient
            labels = [e.rupture_status for e in entries if e.rupture_status is not None]
            
            if not labels:
                # No labels for this patient
                patient_status[patient_id] = None
            else:
                # Majority vote
                patient_status[patient_id] = 1 if sum(labels) / len(labels) >= 0.5 else 0
        
        return patient_status
    
    def _stratified_split(
        self,
        patients: List[str],
        patient_rupture_status: Dict[str, Optional[int]],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Split patients with stratification by rupture status.
        
        Stratification ensures that rupture prevalence is similar
        across train/val/test.
        """
        # Separate patients by rupture status
        ruptured = [p for p in patients if patient_rupture_status.get(p) == 1]
        unruptured = [p for p in patients if patient_rupture_status.get(p) == 0]
        unlabeled = [p for p in patients if patient_rupture_status.get(p) is None]
        
        # Shuffle each group
        np.random.shuffle(ruptured)
        np.random.shuffle(unruptured)
        np.random.shuffle(unlabeled)
        
        # Split each group
        train_ruptured, val_ruptured, test_ruptured = self._split_group(
            ruptured, train_ratio, val_ratio, test_ratio
        )
        train_unruptured, val_unruptured, test_unruptured = self._split_group(
            unruptured, train_ratio, val_ratio, test_ratio
        )
        train_unlabeled, val_unlabeled, test_unlabeled = self._split_group(
            unlabeled, train_ratio, val_ratio, test_ratio
        )
        
        # Combine
        train = train_ruptured + train_unruptured + train_unlabeled
        val = val_ruptured + val_unruptured + val_unlabeled
        test = test_ruptured + test_unruptured + test_unlabeled
        
        return train, val, test
    
    def _random_split(
        self,
        patients: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split patients randomly without stratification."""
        np.random.shuffle(patients)
        
        return self._split_group(patients, train_ratio, val_ratio, test_ratio)
    
    @staticmethod
    def _split_group(
        group: List[str],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split a group of patients by ratios."""
        n = len(group)
        
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        train = group[:train_idx]
        val = group[train_idx:val_idx]
        test = group[val_idx:]
        
        return train, val, test
    
    @staticmethod
    def _create_manifest_from_patients(
        source_manifest: DatasetManifest,
        patient_groups: Dict[str, List[ManifestEntry]],
        patient_ids: List[str],
        manifest_id: str
    ) -> DatasetManifest:
        """Create a new manifest from selected patients."""
        new_manifest = DatasetManifest(
            manifest_id=manifest_id,
            description=f"Patient-level split: {manifest_id}"
        )
        
        for patient_id in patient_ids:
            for entry in patient_groups.get(patient_id, []):
                new_manifest.add_entry(entry)
        
        return new_manifest


class ManifestGenerator:
    """Generates manifests from raw dataset sources."""
    
    @staticmethod
    def generate_from_intra(
        dataset_root: str,
        output_dir: str = "./data/manifests",
        seed: int = 42
    ) -> Tuple[DatasetManifest, DatasetManifest, DatasetManifest]:
        """
        Generate manifests from IntrA dataset.
        
        Args:
            dataset_root: Root directory of IntrA dataset
            output_dir: Directory to save manifests
            seed: Random seed for splits
            
        Returns:
            (train_manifest, val_manifest, test_manifest)
        """
        from .adapters import IntraAdapter
        
        # Discover samples
        adapter = IntraAdapter(dataset_root)
        
        if not adapter.available:
            raise FileNotFoundError(
                f"IntrA dataset not available at {dataset_root}\n"
                "Please download from: https://github.com/rjdmoore/IntrA"
            )
        
        # Get all samples
        samples = adapter.discover_samples()
        
        # Create manifest
        full_manifest = DatasetManifest("full_dataset", "All IntrA samples")
        
        for sample in samples:
            entry = ManifestEntry(
                patient_id=sample.patient_id,
                study_id=sample.patient_id,  # IntrA typically one study per patient
                aneurysm_id=sample.aneurysm_id,
                source="intra",
                geometry_path=str(sample.file_path),
                rupture_status=sample.rupture_label,
                modality="3D mesh",
                quality_control_status="pass"
            )
            full_manifest.add_entry(entry)
        
        # Split
        splitter = PatientLevelSplitter(seed=seed)
        train, val, test = splitter.split_manifest(
            full_manifest,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train.to_csv(f"{output_dir}/development.csv")
        val.to_csv(f"{output_dir}/validation.csv")
        test.to_csv(f"{output_dir}/internal_test.csv")
        
        print(f"✅ Manifests generated:")
        print(f"   Train: {len(train.entries)} entries ({len(train.get_patients())} patients)")
        print(f"   Val:   {len(val.entries)} entries ({len(val.get_patients())} patients)")
        print(f"   Test:  {len(test.entries)} entries ({len(test.get_patients())} patients)")
        
        return train, val, test
    
    @staticmethod
    def generate_synthetic(
        n_patients: int = 100,
        samples_per_patient: int = 2,
        rupture_prevalence: float = 0.3,
        output_dir: str = "./data/manifests",
        seed: int = 42
    ) -> Tuple[DatasetManifest, DatasetManifest, DatasetManifest]:
        """
        Generate synthetic manifests for testing.
        
        Args:
            n_patients: Number of patients to generate
            samples_per_patient: Aneurysms per patient
            rupture_prevalence: Fraction with rupture_status=1
            output_dir: Directory to save manifests
            seed: Random seed
            
        Returns:
            (train_manifest, val_manifest, test_manifest)
        """
        np.random.seed(seed)
        
        # Create full manifest
        full_manifest = DatasetManifest("synthetic_full", "Synthetic dataset for testing")
        
        for p_idx in range(n_patients):
            patient_id = f"synthetic_patient_{p_idx:04d}"
            study_id = f"synthetic_study_{p_idx:04d}"
            
            for a_idx in range(samples_per_patient):
                aneurysm_id = f"aneurysm_{a_idx:02d}"
                
                # Assign rupture label
                rupture_label = 1 if np.random.rand() < rupture_prevalence else 0
                
                entry = ManifestEntry(
                    patient_id=patient_id,
                    study_id=study_id,
                    aneurysm_id=aneurysm_id,
                    source="synthetic",
                    geometry_path=f"synthetic://{patient_id}/{aneurysm_id}",
                    rupture_status=rupture_label,
                    modality="synthetic",
                    quality_control_status="pass"
                )
                
                full_manifest.add_entry(entry)
        
        # Split
        splitter = PatientLevelSplitter(seed=seed)
        train, val, test = splitter.split_manifest(
            full_manifest,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_by_rupture=True
        )
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        train.to_csv(f"{output_dir}/synthetic_development.csv")
        val.to_csv(f"{output_dir}/synthetic_validation.csv")
        test.to_csv(f"{output_dir}/synthetic_internal_test.csv")
        
        print(f"✅ Synthetic manifests generated:")
        print(f"   Train: {len(train.entries)} entries ({len(train.get_patients())} patients)")
        print(f"   Val:   {len(val.entries)} entries ({len(val.get_patients())} patients)")
        print(f"   Test:  {len(test.entries)} entries ({len(test.get_patients())} patients)")
        
        return train, val, test
