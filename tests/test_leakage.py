"""
Test for patient-level data leakage in train/val/test splits.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import split_by_patient, check_split_leakage


def test_split_by_patient_no_leakage():
    """Test that split_by_patient produces no leakage."""
    print("\n" + "="*60)
    print("TEST: Patient-Level Split (No Leakage)")
    print("="*60)
    
    # Create synthetic patient IDs
    patient_ids = [f'patient_{i}' for i in range(10)] * 5  # 50 samples, 10 unique patients
    
    train_patients, val_patients, test_patients = split_by_patient(
        patient_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    # Check for leakage
    leakage_result = check_split_leakage(train_patients, val_patients, test_patients)
    
    print(leakage_result['report'])
    
    if leakage_result['has_leakage']:
        print("❌ FAILED: Leakage detected!")
        return False
    else:
        print("✅ PASSED: No leakage detected")
        return True


def test_split_by_patient_uniqueness():
    """Test that splits partition all unique patients."""
    print("\n" + "="*60)
    print("TEST: Patient-Level Split Uniqueness")
    print("="*60)
    
    patient_ids = [f'patient_{i}' for i in range(20)] * 3  # 60 samples, 20 unique patients
    
    train_patients, val_patients, test_patients = split_by_patient(
        patient_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    all_split_patients = set(train_patients + val_patients + test_patients)
    unique_patients = set([p for p in patient_ids])
    
    if all_split_patients != unique_patients:
        print(f"❌ FAILED: Not all patients assigned to splits")
        print(f"   Expected: {unique_patients}")
        print(f"   Got: {all_split_patients}")
        return False
    
    print(f"✅ PASSED: All {len(unique_patients)} patients assigned correctly")
    return True


def test_split_ratios():
    """Test that split ratios are approximately correct."""
    print("\n" + "="*60)
    print("TEST: Split Ratio Distribution")
    print("="*60)
    
    n_patients = 100
    patient_ids = [f'patient_{i}' for i in range(n_patients)]
    
    train_patients, val_patients, test_patients = split_by_patient(
        patient_ids, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42
    )
    
    train_ratio = len(train_patients) / n_patients
    val_ratio = len(val_patients) / n_patients
    test_ratio = len(test_patients) / n_patients
    
    print(f"  Expected: train=0.60, val=0.20, test=0.20")
    print(f"  Got:      train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
    
    # Allow 5% tolerance
    tolerance = 0.05
    if (abs(train_ratio - 0.6) < tolerance and
        abs(val_ratio - 0.2) < tolerance and
        abs(test_ratio - 0.2) < tolerance):
        print("✅ PASSED: Ratios within tolerance")
        return True
    else:
        print("❌ FAILED: Ratios outside tolerance")
        return False


if __name__ == '__main__':
    tests = [
        test_split_by_patient_no_leakage,
        test_split_by_patient_uniqueness,
        test_split_ratios
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n❌ EXCEPTION in {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "="*60)
    print(f"LEAKAGE TESTS: {sum(results)}/{len(results)} passed")
    print("="*60)
    
    sys.exit(0 if all(results) else 1)
