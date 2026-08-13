#!/usr/bin/env python3
"""
T0: Data Audit and Leakage Detection

Comprehensive data validation pipeline for NeuroFlow datasets.

Performs:
- Schema validation
- File existence checking
- Patient-level leakage detection
- Quality control assessment
- Class balance analysis
- Cohort flow diagram generation
- Comprehensive audit report
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from data import (
    DatasetManifest,
    ComprehensiveValidator,
    ClassBalanceAnalyzer,
    ManifestHasher,
)


def load_manifest(csv_path: str) -> DatasetManifest:
    """Load manifest from CSV file."""
    manifest = DatasetManifest(Path(csv_path).stem)
    manifest.from_csv(csv_path)
    return manifest


def generate_cohort_flow(
    train_manifest: DatasetManifest,
    val_manifest: DatasetManifest,
    test_manifest: DatasetManifest,
    output_dir: str
) -> str:
    """
    Generate cohort flow diagram data.
    
    Returns:
        Markdown text for cohort flow
    """
    total_patients = len(set(
        train_manifest.get_patients() +
        val_manifest.get_patients() +
        test_manifest.get_patients()
    ))
    
    total_entries = (
        len(train_manifest.entries) +
        len(val_manifest.entries) +
        len(test_manifest.entries)
    )
    
    flow_md = f"""
# Cohort Flow Diagram

```
Initial Dataset
  ↓
  Total Patients: {total_patients}
  Total Aneurysms: {total_entries}
  ↓
  [Duplicates Removed]
  [QC Failures Excluded]
  ↓
Development Set: {len(train_manifest.entries)} aneurysms ({len(train_manifest.get_patients())} patients)
  ├─ Used for: Training + hyperparameter tuning
  └─ Rupture prevalence: {ClassBalanceAnalyzer.analyze(train_manifest)['ruptured_ratio']:.1%}
  
Validation Set: {len(val_manifest.entries)} aneurysms ({len(val_manifest.get_patients())} patients)
  ├─ Used for: Early stopping + threshold selection
  └─ Rupture prevalence: {ClassBalanceAnalyzer.analyze(val_manifest)['ruptured_ratio']:.1%}
  
Internal Test Set: {len(test_manifest.entries)} aneurysms ({len(test_manifest.get_patients())} patients)
  ├─ Used for: Final model evaluation
  ├─ Status: LOCKED (not used for tuning)
  └─ Rupture prevalence: {ClassBalanceAnalyzer.analyze(test_manifest)['ruptured_ratio']:.1%}
  
External Test Set: NOT AVAILABLE
  └─ Status: BLOCKED_PENDING_DATA
```
"""
    
    return flow_md


def generate_audit_report(
    train_manifest: DatasetManifest,
    val_manifest: DatasetManifest,
    test_manifest: DatasetManifest,
    audit_result: dict,
    output_dir: str
) -> str:
    """Generate markdown audit report."""
    
    overall_status = audit_result["overall_status"]
    status_color = "🟢" if overall_status == "PASS" else "🔴"
    
    # Determine issue summary
    issues = []
    
    if audit_result["cross_dataset"]["patient_leakage"]["has_leakage"]:
        issues.append("❌ PATIENT LEAKAGE DETECTED")
    
    if audit_result["cross_dataset"]["study_leakage"]["has_leakage"]:
        issues.append("❌ STUDY LEAKAGE DETECTED")
    
    for dataset_name in ["train", "val", "test"]:
        if audit_result["datasets"][dataset_name].get("file_validation", {}).get("files_missing", 0) > 0:
            issues.append(f"❌ Missing files in {dataset_name} set")
    
    issue_text = "\n".join(["- " + issue for issue in issues]) if issues else "✅ No critical issues"
    
    report = f"""
# T0 Data Audit Report

**Overall Status**: {status_color} {overall_status}

## Critical Issues

{issue_text}

## Dataset Statistics

### Development Set
- Total Entries: {len(train_manifest.entries)}
- Unique Patients: {len(train_manifest.get_patients())}
- Unique Studies: {len(train_manifest.get_studies())}
- Rupture Prevalence: {ClassBalanceAnalyzer.analyze(train_manifest)['ruptured_ratio']:.1%}
- Pass Rate: {audit_result['datasets']['train'].get('qc_status', {}).get('pass_rate', 'N/A')}

### Validation Set
- Total Entries: {len(val_manifest.entries)}
- Unique Patients: {len(val_manifest.get_patients())}
- Unique Studies: {len(val_manifest.get_studies())}
- Rupture Prevalence: {ClassBalanceAnalyzer.analyze(val_manifest)['ruptured_ratio']:.1%}
- Pass Rate: {audit_result['datasets']['val'].get('qc_status', {}).get('pass_rate', 'N/A')}

### Internal Test Set
- Total Entries: {len(test_manifest.entries)}
- Unique Patients: {len(test_manifest.get_patients())}
- Unique Studies: {len(test_manifest.get_studies())}
- Rupture Prevalence: {ClassBalanceAnalyzer.analyze(test_manifest)['ruptured_ratio']:.1%}
- Pass Rate: {audit_result['datasets']['test'].get('qc_status', {}).get('pass_rate', 'N/A')}

## Leakage Detection

### Patient Leakage
- Train-Val Overlap: {len(audit_result['cross_dataset']['patient_leakage']['train_val_overlap'])} patients
- Train-Test Overlap: {len(audit_result['cross_dataset']['patient_leakage']['train_test_overlap'])} patients
- Val-Test Overlap: {len(audit_result['cross_dataset']['patient_leakage']['val_test_overlap'])} patients
- Total Unique Patients: {audit_result['cross_dataset']['patient_leakage']['all_patients']}

### Study Leakage
- Train-Val Overlap: {len(audit_result['cross_dataset']['study_leakage']['train_val_overlap'])} studies
- Train-Test Overlap: {len(audit_result['cross_dataset']['study_leakage']['train_test_overlap'])} studies
- Val-Test Overlap: {len(audit_result['cross_dataset']['study_leakage']['val_test_overlap'])} studies
- Total Unique Studies: {audit_result['cross_dataset']['study_leakage']['all_studies']}

## Quality Control

### Development Set QC Status
- Pass: {audit_result['datasets']['train'].get('qc_status', {}).get('pass', 0)}
- Fail: {audit_result['datasets']['train'].get('qc_status', {}).get('fail', 0)}
- Review: {audit_result['datasets']['train'].get('qc_status', {}).get('review', 0)}
- Excluded: {audit_result['datasets']['train'].get('qc_status', {}).get('excluded', 0)}

### Validation Set QC Status
- Pass: {audit_result['datasets']['val'].get('qc_status', {}).get('pass', 0)}
- Fail: {audit_result['datasets']['val'].get('qc_status', {}).get('fail', 0)}
- Review: {audit_result['datasets']['val'].get('qc_status', {}).get('review', 0)}
- Excluded: {audit_result['datasets']['val'].get('qc_status', {}).get('excluded', 0)}

### Internal Test Set QC Status
- Pass: {audit_result['datasets']['test'].get('qc_status', {}).get('pass', 0)}
- Fail: {audit_result['datasets']['test'].get('qc_status', {}).get('fail', 0)}
- Review: {audit_result['datasets']['test'].get('qc_status', {}).get('review', 0)}
- Excluded: {audit_result['datasets']['test'].get('qc_status', {}).get('excluded', 0)}

## Manifest Integrity

### Development Set Manifest
- Hash: {audit_result['datasets']['train'].get('manifest_hash', 'N/A')[:16]}...
- Entries: {len(train_manifest.entries)}

### Validation Set Manifest
- Hash: {audit_result['datasets']['val'].get('manifest_hash', 'N/A')[:16]}...
- Entries: {len(val_manifest.entries)}

### Internal Test Set Manifest
- Hash: {audit_result['datasets']['test'].get('manifest_hash', 'N/A')[:16]}...
- Entries: {len(test_manifest.entries)}

## Recommendations

"""
    
    if overall_status == "PASS":
        report += "✅ Dataset is ready for training. Proceed to T1 (detector baseline).\n"
    else:
        report += "❌ Dataset has critical issues. Fix issues before proceeding to training.\n"
        if issues:
            report += "\n**Required Actions**:\n"
            for issue in issues:
                report += f"- {issue}\n"
    
    return report


def main():
    """Main entry point for T0 data audit."""
    parser = argparse.ArgumentParser(
        description="T0: Data Audit - Comprehensive dataset validation"
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/manifests/development.csv",
        help="Path to training manifest CSV"
    )
    parser.add_argument(
        "--val",
        type=str,
        default="data/manifests/validation.csv",
        help="Path to validation manifest CSV"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/manifests/internal_test.csv",
        help="Path to test manifest CSV"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/datasets",
        help="Base path for data files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./reports/T0_data_audit",
        help="Output directory for audit reports"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("T0: DATA AUDIT AND LEAKAGE DETECTION")
    print("="*70)
    
    # Check if manifests exist
    train_path = Path(args.train)
    val_path = Path(args.val)
    test_path = Path(args.test)
    
    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print(f"\n❌ ERROR: Manifest files not found")
        print(f"   Expected:")
        print(f"   - {args.train}")
        print(f"   - {args.val}")
        print(f"   - {args.test}")
        print(f"\n   Please run: python scripts/create_data_splits.py")
        sys.exit(1)
    
    print(f"\nLoading manifests...")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")
    print(f"  Test:  {args.test}")
    
    # Load manifests
    train_manifest = load_manifest(str(train_path))
    val_manifest = load_manifest(str(val_path))
    test_manifest = load_manifest(str(test_path))
    
    print(f"✅ Loaded: {len(train_manifest.entries)} + {len(val_manifest.entries)} + {len(test_manifest.entries)} = {len(train_manifest.entries) + len(val_manifest.entries) + len(test_manifest.entries)} total entries")
    
    # Run comprehensive audit
    print(f"\nRunning comprehensive audit...")
    audit_result = ComprehensiveValidator.full_audit(
        train_manifest,
        val_manifest,
        test_manifest,
        base_path=args.data_root,
        output_dir=str(output_dir)
    )
    
    # Compute manifest hashes
    print(f"\nComputing manifest hashes...")
    train_hash = ManifestHasher.hash_manifest_csv(str(train_path))
    val_hash = ManifestHasher.hash_manifest_csv(str(val_path))
    test_hash = ManifestHasher.hash_manifest_csv(str(test_path))
    
    audit_result["datasets"]["train"]["manifest_hash"] = train_hash
    audit_result["datasets"]["val"]["manifest_hash"] = val_hash
    audit_result["datasets"]["test"]["manifest_hash"] = test_hash
    
    print(f"  Train hash: {train_hash[:16]}...")
    print(f"  Val hash:   {val_hash[:16]}...")
    print(f"  Test hash:  {test_hash[:16]}...")
    
    # Generate reports
    print(f"\nGenerating reports...")
    
    # Cohort flow
    cohort_flow = generate_cohort_flow(train_manifest, val_manifest, test_manifest, str(output_dir))
    cohort_flow_path = output_dir / "cohort_flow.md"
    with open(cohort_flow_path, 'w') as f:
        f.write(cohort_flow)
    print(f"✅ Cohort flow: {cohort_flow_path}")
    
    # Audit report
    audit_report = generate_audit_report(train_manifest, val_manifest, test_manifest, audit_result, str(output_dir))
    audit_report_path = output_dir / "audit_report.md"
    with open(audit_report_path, 'w') as f:
        f.write(audit_report)
    print(f"✅ Audit report: {audit_report_path}")
    
    # Save detailed JSON
    audit_json_path = output_dir / "audit_result.json"
    with open(audit_json_path, 'w') as f:
        json.dump(audit_result, f, indent=2, default=str)
    print(f"✅ Detailed results: {audit_json_path}")
    
    # Final status
    print(f"\n" + "="*70)
    overall_status = audit_result["overall_status"]
    if overall_status == "PASS":
        print(f"✅ AUDIT PASSED - Dataset is ready for training")
        print(f"\nNext step: Run T1 detector baseline")
        print(f"  python scripts/train_detector.py --config configs/experiments/T1.yaml")
    else:
        print(f"❌ AUDIT FAILED - Fix issues before proceeding")
        print(f"\nReview audit report: {audit_report_path}")
    
    print(f"="*70 + "\n")
    
    sys.exit(0 if overall_status == "PASS" else 1)


if __name__ == "__main__":
    main()
