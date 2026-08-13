"""
Versioned manifest system for NeuroFlow dataset tracking.

Provides:
- Standardized manifest format (CSV/JSON)
- Dataset versioning and hashing
- Duplicate detection
- Patient-level grouping
- Deterministic split generation
- Quality control status tracking
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime
import pandas as pd
import numpy as np


@dataclass
class ManifestEntry:
    """
    Single row in a dataset manifest.
    
    Required fields:
        patient_id: Unique patient identifier
        study_id: Unique study identifier
        aneurysm_id: Unique aneurysm identifier
        source: Dataset source (e.g., "intra", "synthetic")
        geometry_path: Path to 3D mesh file
        rupture_status: 0 (unruptured), 1 (ruptured), None (unlabeled)
    
    Optional fields for richer metadata:
        segmentation_path: Path to vessel segmentation
        modality: Imaging modality (e.g., "CT", "MRA")
        site: Hospital/clinic site identifier
        scanner: Scanner manufacturer/model
        voxel_spacing: Imaging voxel spacing (if applicable)
        acquisition_date: Date of acquisition (if permitted)
        parent_vessel_available: Whether parent vessel context is present
        flow_reference_available: Whether CFD/reference flow data exist
        clinical_variables_available: Whether clinical metadata available
        wall_thickness_available: Whether wall thickness data exist
        image_quality_score: QC quality rating (0-100)
        quality_control_status: "pass", "fail", "review", "excluded"
        exclusion_reason: If excluded, why
        notes: Free-form notes
    """
    
    # Required fields
    patient_id: str
    study_id: str
    aneurysm_id: str
    source: str
    geometry_path: str
    rupture_status: Optional[int]  # 0, 1, or None
    
    # Optional fields
    segmentation_path: Optional[str] = None
    modality: Optional[str] = None
    site: Optional[str] = None
    scanner: Optional[str] = None
    voxel_spacing: Optional[float] = None
    acquisition_date: Optional[str] = None
    parent_vessel_available: bool = False
    flow_reference_available: bool = False
    clinical_variables_available: bool = False
    wall_thickness_available: bool = False
    image_quality_score: Optional[float] = None
    quality_control_status: str = "pass"  # "pass", "fail", "review", "excluded"
    exclusion_reason: Optional[str] = None
    notes: str = ""
    
    # Metadata
    file_hash: str = ""  # SHA256 of geometry file
    dataset_version: str = "1.0"
    added_date: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate manifest entry.
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        if not self.patient_id:
            errors.append("patient_id is required")
        if not self.study_id:
            errors.append("study_id is required")
        if not self.aneurysm_id:
            errors.append("aneurysm_id is required")
        if not self.source:
            errors.append("source is required")
        if not self.geometry_path:
            errors.append("geometry_path is required")
        
        # Check rupture_status
        if self.rupture_status is not None and self.rupture_status not in (0, 1):
            errors.append(f"rupture_status must be 0, 1, or None; got {self.rupture_status}")
        
        # Check quality control status
        valid_qc = ["pass", "fail", "review", "excluded"]
        if self.quality_control_status not in valid_qc:
            errors.append(f"quality_control_status must be one of {valid_qc}")
        
        return len(errors) == 0, errors


class DatasetManifest:
    """Manages a versioned dataset manifest."""
    
    def __init__(self, manifest_id: str, description: str = ""):
        """
        Initialize manifest.
        
        Args:
            manifest_id: Identifier (e.g., "development", "validation", "test")
            description: Human-readable description
        """
        self.manifest_id = manifest_id
        self.description = description
        self.entries: List[ManifestEntry] = []
        self.created_date = datetime.now().isoformat()
        self.version = "1.0"
        self.manifest_hash = ""
    
    def add_entry(self, entry: ManifestEntry) -> Tuple[bool, Optional[str]]:
        """
        Add an entry to manifest with validation.
        
        Args:
            entry: ManifestEntry to add
            
        Returns:
            (success, error_message)
        """
        is_valid, errors = entry.validate()
        if not is_valid:
            error_msg = "; ".join(errors)
            return False, error_msg
        
        self.entries.append(entry)
        return True, None
    
    def get_patients(self) -> List[str]:
        """Get unique patient IDs."""
        return sorted(list(set(e.patient_id for e in self.entries)))
    
    def get_studies(self) -> List[str]:
        """Get unique study IDs."""
        return sorted(list(set(e.study_id for e in self.entries)))
    
    def get_by_patient(self, patient_id: str) -> List[ManifestEntry]:
        """Get all entries for a patient."""
        return [e for e in self.entries if e.patient_id == patient_id]
    
    def get_by_study(self, study_id: str) -> List[ManifestEntry]:
        """Get all entries for a study."""
        return [e for e in self.entries if e.study_id == study_id]
    
    def statistics(self) -> Dict[str, Any]:
        """
        Generate dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        df = pd.DataFrame([e.to_dict() for e in self.entries])
        
        stats = {
            "total_entries": len(self.entries),
            "unique_patients": len(set(e.patient_id for e in self.entries)),
            "unique_studies": len(set(e.study_id for e in self.entries)),
            "unique_aneurysms": len(set(e.aneurysm_id for e in self.entries)),
            "sources": df['source'].value_counts().to_dict() if len(df) > 0 else {},
        }
        
        # Rupture status breakdown
        if 'rupture_status' in df.columns:
            rupture_counts = df['rupture_status'].value_counts().to_dict()
            stats['rupture_status'] = rupture_counts
            stats['rupture_prevalence'] = rupture_counts.get(1, 0) / len(df) if len(df) > 0 else 0
        
        # QC status breakdown
        if 'quality_control_status' in df.columns:
            qc_counts = df['quality_control_status'].value_counts().to_dict()
            stats['quality_control_status'] = qc_counts
            stats['pass_rate'] = qc_counts.get('pass', 0) / len(df) if len(df) > 0 else 0
        
        # Missing data
        stats['missing_data'] = {
            col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0
        }
        
        return stats
    
    def to_csv(self, filepath: str):
        """Save manifest to CSV."""
        df = pd.DataFrame([e.to_dict() for e in self.entries])
        df.to_csv(filepath, index=False)
        print(f"✅ Manifest saved to {filepath} ({len(self.entries)} entries)")
    
    def from_csv(self, filepath: str):
        """Load manifest from CSV."""
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            # Convert NaN to None for optional fields
            row_dict = row.to_dict()
            for key in row_dict:
                if pd.isna(row_dict[key]):
                    row_dict[key] = None
            
            entry = ManifestEntry(**row_dict)
            success, error = self.add_entry(entry)
            if not success:
                print(f"⚠️  Warning: Skipped entry due to validation error: {error}")
        
        print(f"✅ Manifest loaded from {filepath} ({len(self.entries)} entries)")
    
    def to_json(self, filepath: str):
        """Save manifest to JSON."""
        data = {
            "manifest_id": self.manifest_id,
            "description": self.description,
            "created_date": self.created_date,
            "version": self.version,
            "entries": [e.to_dict() for e in self.entries]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Manifest saved to {filepath} ({len(self.entries)} entries)")
    
    def from_json(self, filepath: str):
        """Load manifest from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for entry_dict in data.get('entries', []):
            entry = ManifestEntry(**entry_dict)
            success, error = self.add_entry(entry)
            if not success:
                print(f"⚠️  Warning: Skipped entry due to validation error: {error}")
        
        print(f"✅ Manifest loaded from {filepath} ({len(self.entries)} entries)")
    
    def compute_hash(self) -> str:
        """
        Compute SHA256 hash of manifest for reproducibility.
        
        Returns:
            Hex digest of manifest contents
        """
        # Sort entries by patient/study/aneurysm for determinism
        sorted_entries = sorted(
            self.entries,
            key=lambda e: (e.patient_id, e.study_id, e.aneurysm_id)
        )
        
        # Create string representation
        manifest_str = json.dumps(
            [e.to_dict() for e in sorted_entries],
            sort_keys=True,
            default=str
        )
        
        # Compute hash
        self.manifest_hash = hashlib.sha256(manifest_str.encode()).hexdigest()
        return self.manifest_hash


class DuplicateDetector:
    """Detects duplicates and potential data leakage in manifests."""
    
    @staticmethod
    def find_duplicate_patients(manifests: List[DatasetManifest]) -> Dict[str, List[str]]:
        """
        Find patients appearing in multiple manifests (data leakage).
        
        Args:
            manifests: List of manifests to check (e.g., train, val, test)
            
        Returns:
            Dict mapping patient_id to list of manifests they appear in
        """
        patient_to_manifests = {}
        
        for manifest in manifests:
            for patient_id in manifest.get_patients():
                if patient_id not in patient_to_manifests:
                    patient_to_manifests[patient_id] = []
                patient_to_manifests[patient_id].append(manifest.manifest_id)
        
        # Filter to only duplicates
        duplicates = {
            pid: manifests for pid, manifests in patient_to_manifests.items()
            if len(manifests) > 1
        }
        
        return duplicates
    
    @staticmethod
    def find_duplicate_studies(manifest: DatasetManifest) -> Dict[str, int]:
        """
        Find studies appearing multiple times within a manifest.
        
        Args:
            manifest: Manifest to check
            
        Returns:
            Dict of study_id → count for duplicates (count > 1)
        """
        study_counts = {}
        
        for entry in manifest.entries:
            study_counts[entry.study_id] = study_counts.get(entry.study_id, 0) + 1
        
        # Filter to only duplicates
        duplicates = {sid: count for sid, count in study_counts.items() if count > 1}
        
        return duplicates
    
    @staticmethod
    def find_geometry_duplicates(manifest: DatasetManifest) -> Dict[str, List[str]]:
        """
        Find identical geometries (by file hash) appearing multiple times.
        
        Args:
            manifest: Manifest to check
            
        Returns:
            Dict mapping file_hash → list of aneurysm_ids
        """
        hash_to_aneurysms = {}
        
        for entry in manifest.entries:
            if entry.file_hash:
                if entry.file_hash not in hash_to_aneurysms:
                    hash_to_aneurysms[entry.file_hash] = []
                hash_to_aneurysms[entry.file_hash].append(entry.aneurysm_id)
        
        # Filter to only duplicates
        duplicates = {
            fhash: aneurysms for fhash, aneurysms in hash_to_aneurysms.items()
            if len(aneurysms) > 1
        }
        
        return duplicates


class ClassBalanceAnalyzer:
    """Analyzes class balance in rupture prediction datasets."""
    
    @staticmethod
    def analyze(manifest: DatasetManifest) -> Dict[str, Any]:
        """
        Analyze rupture status class balance.
        
        Args:
            manifest: Manifest to analyze
            
        Returns:
            Dictionary with balance statistics
        """
        entries_with_label = [e for e in manifest.entries if e.rupture_status is not None]
        
        if not entries_with_label:
            return {
                "total_labeled": 0,
                "total_unlabeled": len(manifest.entries),
                "message": "No rupture labels available"
            }
        
        unruptured = sum(1 for e in entries_with_label if e.rupture_status == 0)
        ruptured = sum(1 for e in entries_with_label if e.rupture_status == 1)
        total = len(entries_with_label)
        
        return {
            "total_labeled": total,
            "total_unlabeled": len(manifest.entries) - total,
            "unruptured_count": unruptured,
            "ruptured_count": ruptured,
            "unruptured_ratio": unruptured / total if total > 0 else 0,
            "ruptured_ratio": ruptured / total if total > 0 else 0,
            "balance_ratio": max(unruptured, ruptured) / min(unruptured, ruptured) if min(unruptured, ruptured) > 0 else float('inf'),
        }
