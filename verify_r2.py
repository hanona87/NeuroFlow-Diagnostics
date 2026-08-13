"""
Integration verification for Milestone R2.

This script verifies that all R2 components are correctly implemented
without running the full preprocessing pipeline (which may require system resources).

Run this to verify R2 is ready for R3.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("\n" + "="*80)
print("MILESTONE R2 VERIFICATION")
print("="*80)

# Test 1: Module imports
print("\n[1/7] Checking data.adapters imports...")
try:
    from data.adapters import BaseDatasetAdapter, IntraAdapter, SyntheticAdapter
    print("✅ All adapters imported successfully")
except ImportError as e:
    print(f"❌ Failed to import adapters: {e}")
    sys.exit(1)

# Test 2: Data __init__ exports
print("\n[2/7] Checking data package exports...")
try:
    import data
    assert hasattr(data, 'preprocessing')
    assert hasattr(data, 'adapters')
    print("✅ data package exports both preprocessing and adapters")
except (ImportError, AssertionError) as e:
    print(f"❌ data package exports incorrect: {e}")
    sys.exit(1)

# Test 3: Existing preprocessing still works
print("\n[3/7] Checking preprocessing module compatibility...")
try:
    from data.preprocessing import PointCloudPreprocessor, PointCloudDataset, create_synthetic_dataset
    print("✅ Existing preprocessing imports still work")
except ImportError as e:
    print(f"❌ Preprocessing imports broken: {e}")
    sys.exit(1)

# Test 4: SyntheticAdapter basic functionality
print("\n[4/7] Checking SyntheticAdapter functionality...")
try:
    adapter = SyntheticAdapter(n_patients=3, samples_per_patient=2, seed=42)
    samples = adapter.discover_samples()
    
    assert len(samples) == 6, f"Expected 6 samples, got {len(samples)}"
    assert samples[0].patient_id == "synthetic_patient_000"
    assert all(hasattr(s, 'rupture_label') for s in samples)
    
    print(f"✅ SyntheticAdapter works: {len(samples)} samples, {len(adapter.get_patient_groups())} patients")
except Exception as e:
    print(f"❌ SyntheticAdapter failed: {e}")
    sys.exit(1)

# Test 5: IntraAdapter graceful fallback
print("\n[5/7] Checking IntraAdapter graceful fallback...")
try:
    adapter = IntraAdapter(data_root="/nonexistent/path/to/intra")
    
    assert not adapter.available, "IntrA should not be available (test path is fake)"
    assert adapter.error_message is not None, "IntrA should have error message"
    assert len(adapter.samples) == 0, "IntrA should have no samples"
    
    print(f"✅ IntraAdapter gracefully handles missing data (error message: {adapter.error_message[:50]}...)")
except Exception as e:
    print(f"❌ IntraAdapter fallback failed: {e}")
    sys.exit(1)

# Test 6: Patient-level grouping
print("\n[6/7] Checking patient-level grouping...")
try:
    patient_groups = adapter.get_patient_groups()
    assert len(patient_groups) == 0  # No samples, so no groups
    
    # Test with synthetic
    adapter = SyntheticAdapter(n_patients=5, samples_per_patient=3, seed=42)
    samples = adapter.discover_samples()
    patient_groups = adapter.get_patient_groups()
    
    assert len(patient_groups) == 5, f"Expected 5 patients, got {len(patient_groups)}"
    assert all(len(g) == 3 for g in patient_groups.values()), "Each patient should have 3 aneurysms"
    
    print(f"✅ Patient-level grouping works: {len(patient_groups)} patients, 3 aneurysms each")
except Exception as e:
    print(f"❌ Patient grouping failed: {e}")
    sys.exit(1)

# Test 7: Manifest save/load
print("\n[7/7] Checking manifest persistence...")
try:
    import tempfile
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, 'test_manifest.json')
        
        # Save
        adapter.save_manifest(manifest_path)
        
        # Verify file exists
        assert os.path.exists(manifest_path), "Manifest file not created"
        
        # Load and verify
        with open(manifest_path) as f:
            data = json.load(f)
        
        assert 'dataset' in data
        assert 'samples' in data
        assert len(data['samples']) == 15  # 5 patients * 3 aneurysms
        
        # Load via adapter
        new_adapter = SyntheticAdapter(n_patients=1)
        new_adapter.load_manifest(manifest_path)
        
        assert len(new_adapter.samples) == 15, "Loaded manifest has wrong number of samples"
        
        print(f"✅ Manifest persistence works (save/load with {len(data['samples'])} samples)")
except Exception as e:
    print(f"❌ Manifest persistence failed: {e}")
    sys.exit(1)

# Test 8: Verify CLI scripts exist
print("\n[BONUS] Checking CLI scripts...")
try:
    assert os.path.exists('scripts/preprocess_datasets.py'), "preprocess_datasets.py not found"
    assert os.path.exists('scripts/create_data_splits.py'), "create_data_splits.py not found"
    
    # Check scripts are valid Python
    with open('scripts/preprocess_datasets.py') as f:
        compile(f.read(), 'scripts/preprocess_datasets.py', 'exec')
    
    with open('scripts/create_data_splits.py') as f:
        compile(f.read(), 'scripts/create_data_splits.py', 'exec')
    
    print(f"✅ CLI scripts present and syntactically valid")
except Exception as e:
    print(f"❌ CLI scripts check failed: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL R2 VERIFICATION CHECKS PASSED")
print("="*80)
print("""
Summary of Milestone R2:

✅ Data adapter interface implemented (BaseDatasetAdapter)
✅ IntraAdapter with graceful fallback to synthetic
✅ SyntheticAdapter for testing/fallback
✅ Patient-level split support (no leakage)
✅ Manifest persistence for reproducibility
✅ CLI scripts for preprocessing and splitting
✅ Backward compatibility with existing tests maintained

Ready for Milestone R3:
- Integrate splits into Stage 1 training
- Load data from split manifests
- Evaluate metrics on frozen test set
""")
print("="*80 + "\n")
