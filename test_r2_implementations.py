"""Quick test to verify R2 implementations work."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing R2 implementations...")
print("=" * 80)

# Test 1: Import data adapters
print("\n[Test 1] Import data adapters...")
try:
    from data.adapters import BaseDatasetAdapter, IntraAdapter, SyntheticAdapter
    print("✅ Adapters imported successfully")
except Exception as e:
    print(f"❌ Failed to import adapters: {e}")
    sys.exit(1)

# Test 2: Create synthetic adapter
print("\n[Test 2] Create synthetic adapter...")
try:
    adapter = SyntheticAdapter(n_patients=5, samples_per_patient=2, seed=42)
    samples = adapter.discover_samples()
    print(f"✅ Synthetic adapter created: {len(samples)} samples from {len(adapter.get_patient_groups())} patients")
except Exception as e:
    print(f"❌ Failed to create synthetic adapter: {e}")
    sys.exit(1)

# Test 3: Validate IntrA adapter graceful fallback
print("\n[Test 3] Test IntrA adapter (should gracefully fallback to synthetic)...")
try:
    intra_adapter = IntraAdapter(data_root="/nonexistent/path")
    if not intra_adapter.available:
        print(f"✅ IntrA adapter correctly reports unavailable (expected): {intra_adapter.error_message[:80]}...")
    else:
        print("⚠️  IntrA unexpectedly available (OK if running on system with IntrA)")
except Exception as e:
    print(f"❌ IntrA adapter failed unexpectedly: {e}")
    sys.exit(1)

# Test 4: Load mesh from synthetic
print("\n[Test 4] Load mesh from synthetic sample...")
try:
    sample = samples[0]
    vertices, faces = adapter.load_mesh(sample)
    print(f"✅ Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
except Exception as e:
    print(f"❌ Failed to load mesh: {e}")
    sys.exit(1)

# Test 5: Preprocess single sample
print("\n[Test 5] Preprocess single sample...")
try:
    from data.preprocessing import PointCloudPreprocessor
    from utils import fps, normalize_pc, compute_normals
    
    preprocessor = PointCloudPreprocessor(num_points=1024, seed=42)
    
    # Sample points
    sampled_points = preprocessor.sample_points_from_mesh(vertices, faces)
    
    # FPS
    if len(sampled_points) > 1024:
        indices = fps(sampled_points, 1024)
        points = sampled_points[indices]
    else:
        indices = list(range(len(sampled_points)))
        points = sampled_points
    
    # Normalize
    points, norm_params = normalize_pc(points, method='unit_sphere')
    
    # Compute normals
    normals = compute_normals(points, k=10)
    
    result = np.hstack([points, normals]).astype(np.float32)
    
    print(f"✅ Sample preprocessed: {result.shape}")
except Exception as e:
    print(f"❌ Failed to preprocess: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test patient-level split checking
print("\n[Test 6] Test patient-level split checking...")
try:
    from utils import split_by_patient, check_split_leakage
    
    # Get patient groups
    patient_groups = adapter.get_patient_groups()
    patient_ids = list(patient_groups.keys())
    
    # Create splits
    train_p, val_p, test_p = split_by_patient(patient_ids, seed=42)
    
    # Check leakage
    result = check_split_leakage(train_p, val_p, test_p)
    
    if not result['has_leakage']:
        print(f"✅ Splits validated (no leakage)")
        print(f"   Train: {len(train_p)} patients, Val: {len(val_p)} patients, Test: {len(test_p)} patients")
    else:
        print(f"❌ Leakage detected: {result}")
        sys.exit(1)

except Exception as e:
    print(f"❌ Failed split check: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test manifest save/load
print("\n[Test 7] Test manifest save/load...")
try:
    import json
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_path = os.path.join(tmpdir, 'manifest.json')
        
        # Save
        adapter.save_manifest(manifest_path)
        print(f"✅ Manifest saved")
        
        # Load
        new_adapter = SyntheticAdapter(n_patients=1, samples_per_patient=1)
        new_adapter.load_manifest(manifest_path)
        
        if len(new_adapter.samples) == len(adapter.samples):
            print(f"✅ Manifest loaded: {len(new_adapter.samples)} samples")
        else:
            print(f"❌ Manifest samples mismatch: {len(new_adapter.samples)} vs {len(adapter.samples)}")
            sys.exit(1)

except Exception as e:
    print(f"❌ Failed manifest test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL R2 TESTS PASSED")
print("=" * 80)
