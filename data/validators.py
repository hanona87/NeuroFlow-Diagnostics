"""
Dataset validation and quality control for NeuroFlow manifests.

Provides:
- Schema validation
- File existence checking
- Geometry integrity validation
- Metadata consistency checking
- Quality control flags and reports
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import numpy as np

from .manifest import DatasetManifest, ManifestEntry, DuplicateDetector


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: str  # "error", "warning", "info"
    message: str
    details: Optional[Dict[str, Any]] = None


class SchemaValidator:
    """Validates manifest schema and required fields."""
    
    @staticmethod
    def validate_entry(entry: ManifestEntry) -> List[ValidationResult]:
        """
        Validate a single manifest entry against schema.
        
        Args:
            entry: ManifestEntry to validate
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        # Required fields
        required_fields = ['patient_id', 'study_id', 'aneurysm_id', 'source', 'geometry_path', 'rupture_status']
        for field in required_fields:
            value = getattr(entry, field, None)
            if value is None or value == "":
                results.append(ValidationResult(
                    is_valid=False,
                    severity="error",
                    message=f"Required field missing: {field}"
                ))
        
        # Type checks
        if entry.rupture_status is not None and not isinstance(entry.rupture_status, int):
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message=f"rupture_status must be int or None; got {type(entry.rupture_status)}"
            ))
        
        if entry.rupture_status is not None and entry.rupture_status not in (0, 1):
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message=f"rupture_status must be 0 or 1; got {entry.rupture_status}"
            ))
        
        # Consistency checks
        if entry.quality_control_status == "excluded" and not entry.exclusion_reason:
            results.append(ValidationResult(
                is_valid=False,
                severity="warning",
                message="Excluded entry should have exclusion_reason"
            ))
        
        return results
    
    @staticmethod
    def validate_manifest(manifest: DatasetManifest) -> List[ValidationResult]:
        """
        Validate entire manifest.
        
        Args:
            manifest: DatasetManifest to validate
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        if not manifest.entries:
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message="Manifest contains no entries"
            ))
            return results
        
        # Validate each entry
        for i, entry in enumerate(manifest.entries):
            entry_results = SchemaValidator.validate_entry(entry)
            for result in entry_results:
                result.message = f"Entry {i} ({entry.aneurysm_id}): {result.message}"
                results.append(result)
        
        # Check for patient/study/aneurysm uniqueness
        patient_ids = [e.patient_id for e in manifest.entries]
        study_ids = [e.study_id for e in manifest.entries]
        aneurysm_ids = [e.aneurysm_id for e in manifest.entries]
        
        # Note: aneurysm_id should be unique within a patient, not globally
        # patient/study/aneurysm combination should be unique
        composite_ids = [(e.patient_id, e.study_id, e.aneurysm_id) for e in manifest.entries]
        if len(composite_ids) != len(set(composite_ids)):
            results.append(ValidationResult(
                is_valid=False,
                severity="error",
                message="Duplicate (patient_id, study_id, aneurysm_id) combination detected"
            ))
        
        return results


class FileValidator:
    """Validates file existence and integrity."""
    
    @staticmethod
    def check_geometry_files(manifest: DatasetManifest, base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if all geometry files exist.
        
        Args:
            manifest: DatasetManifest to validate
            base_path: Optional base path for relative paths
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_entries": len(manifest.entries),
            "files_found": 0,
            "files_missing": 0,
            "missing_files": [],
            "is_valid": True
        }
        
        for entry in manifest.entries:
            # Skip synthetic entries
            if entry.source == "synthetic":
                continue
            
            filepath = entry.geometry_path
            if base_path and not Path(filepath).is_absolute():
                filepath = os.path.join(base_path, filepath)
            
            if Path(filepath).exists():
                results["files_found"] += 1
            else:
                results["files_missing"] += 1
                results["missing_files"].append({
                    "patient_id": entry.patient_id,
                    "aneurysm_id": entry.aneurysm_id,
                    "path": entry.geometry_path
                })
                results["is_valid"] = False
        
        return results
    
    @staticmethod
    def check_segmentation_files(manifest: DatasetManifest, base_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Check if segmentation files exist (when present).
        
        Args:
            manifest: DatasetManifest to validate
            base_path: Optional base path for relative paths
            
        Returns:
            Dictionary with validation results
        """
        results = {
            "total_entries": len(manifest.entries),
            "segmentations_expected": 0,
            "files_found": 0,
            "files_missing": 0,
            "missing_files": [],
            "is_valid": True
        }
        
        for entry in manifest.entries:
            # Skip if no segmentation path
            if not entry.segmentation_path:
                continue
            
            results["segmentations_expected"] += 1
            
            # Skip synthetic
            if entry.source == "synthetic":
                results["files_found"] += 1
                continue
            
            filepath = entry.segmentation_path
            if base_path and not Path(filepath).is_absolute():
                filepath = os.path.join(base_path, filepath)
            
            if Path(filepath).exists():
                results["files_found"] += 1
            else:
                results["files_missing"] += 1
                results["missing_files"].append({
                    "patient_id": entry.patient_id,
                    "aneurysm_id": entry.aneurysm_id,
                    "path": entry.segmentation_path
                })
                results["is_valid"] = False
        
        return results


class DataLeakageValidator:
    """Detects data leakage across train/val/test splits."""
    
    @staticmethod
    def check_patient_leakage(
        train_manifest: DatasetManifest,
        val_manifest: DatasetManifest,
        test_manifest: DatasetManifest
    ) -> Dict[str, Any]:
        """
        Check for patient-level leakage across splits.
        
        Args:
            train_manifest: Training manifest
            val_manifest: Validation manifest
            test_manifest: Test manifest
            
        Returns:
            Leakage report
        """
        train_patients = set(train_manifest.get_patients())
        val_patients = set(val_manifest.get_patients())
        test_patients = set(test_manifest.get_patients())
        
        leakage = {
            "has_leakage": False,
            "train_val_overlap": train_patients & val_patients,
            "train_test_overlap": train_patients & test_patients,
            "val_test_overlap": val_patients & test_patients,
            "all_patients": len(train_patients | val_patients | test_patients),
        }
        
        if leakage["train_val_overlap"] or leakage["train_test_overlap"] or leakage["val_test_overlap"]:
            leakage["has_leakage"] = True
        
        return leakage
    
    @staticmethod
    def check_study_leakage(
        train_manifest: DatasetManifest,
        val_manifest: DatasetManifest,
        test_manifest: DatasetManifest
    ) -> Dict[str, Any]:
        """
        Check for study-level leakage across splits.
        
        Args:
            train_manifest: Training manifest
            val_manifest: Validation manifest
            test_manifest: Test manifest
            
        Returns:
            Leakage report
        """
        train_studies = set(train_manifest.get_studies())
        val_studies = set(val_manifest.get_studies())
        test_studies = set(test_manifest.get_studies())
        
        leakage = {
            "has_leakage": False,
            "train_val_overlap": train_studies & val_studies,
            "train_test_overlap": train_studies & test_studies,
            "val_test_overlap": val_studies & test_studies,
            "all_studies": len(train_studies | val_studies | test_studies),
        }
        
        if leakage["train_val_overlap"] or leakage["train_test_overlap"] or leakage["val_test_overlap"]:
            leakage["has_leakage"] = True
        
        return leakage


class QualityControlValidator:
    """Checks quality control status of datasets."""
    
    @staticmethod
    def assess_qc_status(manifest: DatasetManifest) -> Dict[str, Any]:
        """
        Assess overall QC status of manifest.
        
        Args:
            manifest: DatasetManifest to assess
            
        Returns:
            QC assessment report
        """
        entries = manifest.entries
        
        qc_statuses = {}
        for entry in entries:
            status = entry.quality_control_status
            qc_statuses[status] = qc_statuses.get(status, 0) + 1
        
        pass_count = qc_statuses.get('pass', 0)
        fail_count = qc_statuses.get('fail', 0)
        review_count = qc_statuses.get('review', 0)
        excluded_count = qc_statuses.get('excluded', 0)
        
        total = len(entries)
        usable_count = pass_count + review_count  # Can use after review
        
        return {
            "total_entries": total,
            "pass": pass_count,
            "fail": fail_count,
            "review": review_count,
            "excluded": excluded_count,
            "usable_count": usable_count,
            "usable_ratio": usable_count / total if total > 0 else 0,
            "pass_rate": pass_count / total if total > 0 else 0,
            "fail_rate": fail_count / total if total > 0 else 0,
        }
    
    @staticmethod
    def identify_failed_entries(manifest: DatasetManifest) -> List[Dict[str, Any]]:
        """
        Identify entries that failed QC.
        
        Args:
            manifest: DatasetManifest to check
            
        Returns:
            List of failed entries with reasons
        """
        failed = []
        
        for entry in manifest.entries:
            if entry.quality_control_status in ("fail", "excluded"):
                failed.append({
                    "patient_id": entry.patient_id,
                    "aneurysm_id": entry.aneurysm_id,
                    "status": entry.quality_control_status,
                    "reason": entry.exclusion_reason or "Not specified",
                    "notes": entry.notes
                })
        
        return failed


class MissingDataValidator:
    """Detects and reports missing data in manifests."""
    
    @staticmethod
    def check_missing_data(manifest: DatasetManifest) -> Dict[str, Any]:
        """
        Check for missing data across manifest.
        
        Args:
            manifest: DatasetManifest to check
            
        Returns:
            Missing data report
        """
        missing_by_field = {}
        
        for entry in manifest.entries:
            for field_name in entry.__dataclass_fields__:
                value = getattr(entry, field_name)
                if value is None or (isinstance(value, str) and value == ""):
                    if field_name not in missing_by_field:
                        missing_by_field[field_name] = []
                    missing_by_field[field_name].append({
                        "patient_id": entry.patient_id,
                        "aneurysm_id": entry.aneurysm_id
                    })
        
        # Summarize
        missing_summary = {}
        for field, instances in missing_by_field.items():
            missing_summary[field] = {
                "count": len(instances),
                "ratio": len(instances) / len(manifest.entries),
                "samples": instances[:3]  # First 3 examples
            }
        
        return missing_summary


class ComprehensiveValidator:
    """Comprehensive dataset validation combining all validators."""
    
    @staticmethod
    def full_audit(
        train_manifest: DatasetManifest,
        val_manifest: DatasetManifest,
        test_manifest: DatasetManifest,
        base_path: Optional[str] = None,
        output_dir: str = "./reports/data_audit"
    ) -> Dict[str, Any]:
        """
        Run comprehensive data audit.
        
        Args:
            train_manifest: Training manifest
            val_manifest: Validation manifest
            test_manifest: Test manifest
            base_path: Base path for files
            output_dir: Directory to save audit report
            
        Returns:
            Complete audit report
        """
        os.makedirs(output_dir, exist_ok=True)
        
        audit_report = {
            "timestamp": str(np.datetime64('now')),
            "audit_type": "comprehensive",
            "datasets": {
                "train": {},
                "val": {},
                "test": {}
            },
            "cross_dataset": {},
            "overall_status": "PASS"
        }
        
        # Audit each manifest
        for manifest, name in [(train_manifest, "train"), (val_manifest, "val"), (test_manifest, "test")]:
            audit_report["datasets"][name] = {
                "schema_validation": [r.__dict__ for r in SchemaValidator.validate_manifest(manifest)],
                "statistics": manifest.statistics(),
                "qc_status": QualityControlValidator.assess_qc_status(manifest),
                "missing_data": MissingDataValidator.check_missing_data(manifest),
                "duplicate_studies": DuplicateDetector.find_duplicate_studies(manifest),
                "duplicate_geometries": DuplicateDetector.find_geometry_duplicates(manifest),
                "file_validation": FileValidator.check_geometry_files(manifest, base_path),
                "segmentation_validation": FileValidator.check_segmentation_files(manifest, base_path),
            }
        
        # Cross-dataset leakage check
        audit_report["cross_dataset"]["patient_leakage"] = DataLeakageValidator.check_patient_leakage(
            train_manifest, val_manifest, test_manifest
        )
        audit_report["cross_dataset"]["study_leakage"] = DataLeakageValidator.check_study_leakage(
            train_manifest, val_manifest, test_manifest
        )
        
        # Determine overall status
        if audit_report["cross_dataset"]["patient_leakage"]["has_leakage"]:
            audit_report["overall_status"] = "FAIL"
        if audit_report["cross_dataset"]["study_leakage"]["has_leakage"]:
            audit_report["overall_status"] = "FAIL"
        
        for dataset_name in ["train", "val", "test"]:
            if audit_report["datasets"][dataset_name].get("file_validation", {}).get("files_missing", 0) > 0:
                audit_report["overall_status"] = "FAIL"
        
        # Save report
        report_path = os.path.join(output_dir, "audit_report.json")
        with open(report_path, 'w') as f:
            json.dump(audit_report, f, indent=2, default=str)
        
        print(f"✅ Audit report saved to {report_path}")
        print(f"   Overall Status: {audit_report['overall_status']}")
        
        return audit_report
