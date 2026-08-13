# Milestone R2 Quick Reference

## Commands to Run

### 1. Verify R2 Installation
```bash
python verify_r2.py
```
**Expected Output:**
```
✅ All adapters imported successfully
✅ data package exports both preprocessing and adapters
✅ Existing preprocessing imports still work
✅ SyntheticAdapter works: 6 samples, 3 patients
✅ IntraAdapter gracefully handles missing data
✅ Patient-level grouping works: 5 patients, 3 aneurysms each
✅ Manifest persistence works (save/load with 15 samples)
✅ CLI scripts present and syntactically valid

✅ ALL R2 VERIFICATION CHECKS PASSED
```

### 2. Run Full R2 Test Suite
```bash
python test_r2_implementations.py
```
**Expected Output:**
```
Testing R2 implementations...
[Test 1] Import data adapters... ✅
[Test 2] Create synthetic adapter... ✅
[Test 3] Test IntrA adapter... ✅
[Test 4] Load mesh from synthetic sample... ✅
[Test 5] Preprocess single sample... ✅
[Test 6] Test patient-level split checking... ✅
[Test 7] Test manifest save/load... ✅

✅ ALL R2 TESTS PASSED
```

### 3. Generate Synthetic Dataset
```bash
python scripts/preprocess_datasets.py \
  --dataset synthetic \
  --n-patients 20 \
  --samples-per-patient 2 \
  --num-points 8192 \
  --output-dir ./data/processed
```
**Expected Output:**
```
📊 Discovering synthetic samples...
✅ Found 40 samples from 20 patients

📈 Dataset validation:
   total_samples: 40
   unique_patients: 20
   avg_samples_per_patient: 2.0

🔄 Preprocessing 40 samples...
[████████████████████████] 40/40

✅ Successfully processed: 40
❌ Failed: 0

💾 Writing to HDF5...
✅ Saved to: data/processed/full.h5

📋 Creating manifest...
✅ Manifest saved to: data/processed/manifest.json

✅ Metadata saved to: data/processed/preprocessing_metadata.json

================================================================================
  PREPROCESSING COMPLETE
================================================================================
{
  "dataset": "synthetic",
  "processed_samples": 40,
  "failed_samples": 0,
  "num_points": 8192,
  "normalization_method": "unit_sphere",
  "total_patients": 20,
  "output_file": "data/processed/full.h5",
  "manifest_file": "data/processed/manifest.json",
  "seed": 42
}
================================================================================
```

### 4. Create Patient-Level Splits
```bash
python scripts/create_data_splits.py \
  --input ./data/processed/full.h5 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15
```
**Expected Output:**
```
📂 Loading data from data/processed/full.h5...
✅ Loaded 40 samples from 20 patients

🔀 Creating patient-level splits (train/val/test = 0.7/0.15/0.15)...
✅ Train: 28 samples (14 patients)
✅ Val: 6 samples (3 patients)
✅ Test: 6 samples (3 patients)

🔍 Checking for data leakage...
================================================================================
  PATIENT-LEVEL LEAKAGE CHECK
================================================================================
  Train patients: 14
  Val patients: 3
  Test patients: 3

  ✅ NO LEAKAGE DETECTED (splits are clean)
================================================================================

📋 Creating split manifests...
✅ Saved train manifest to data/manifests/train_manifest.json
✅ Saved val manifest to data/manifests/val_manifest.json
✅ Saved test manifest to data/manifests/test_manifest.json
✅ Saved combined manifest to data/manifests/splits_manifest.json

📁 Creating experiment directories...
✅ experiments/T0_leakage
✅ experiments/T1_detection_baseline
✅ experiments/T1_smoke
✅ experiments/T2_robustness
✅ experiments/T3_pinn_smoke
✅ experiments/T4_ablation
✅ experiments/T5_uncertainty_calibration

✅ Leakage verification saved to data/manifests/leakage_verification.txt

================================================================================
  SPLITTING COMPLETE
================================================================================
{
  "dataset": "data/processed/full.h5",
  "total_samples": 40,
  "total_patients": 20,
  "splits": { ... },
  "seed": 42,
  "parameters": { ... }
}
================================================================================
```

### 5. Verify No Leakage
```bash
cat data/manifests/leakage_verification.txt
```
**Expected Output:**
```
================================================================================
  PATIENT-LEVEL LEAKAGE CHECK
================================================================================
  Train patients: 14
  Val patients: 3
  Test patients: 3

  ✅ NO LEAKAGE DETECTED (splits are clean)
================================================================================
```

### 6. Inspect Split Manifests
```bash
# View train split
python -m json.tool data/manifests/train_manifest.json | head -30

# View split summary
python -c "import json; m=json.load(open('data/manifests/splits_manifest.json')); \
  print(f\"Train: {m['splits']['train']['num_samples']} samples, \
  {m['splits']['train']['num_patients']} patients\"); \
  print(f\"Val: {m['splits']['val']['num_samples']} samples, \
  {m['splits']['val']['num_patients']} patients\"); \
  print(f\"Test: {m['splits']['test']['num_samples']} samples, \
  {m['splits']['test']['num_patients']} patients\")"
```

### 7. Verify Original Tests Still Work
```bash
python test_project.py
```
**Expected Output:**
```
============================================================
TEST 1: Module Imports
============================================================
✅ utils module imported
✅ models module imported
✅ losses module imported
✅ evaluation module imported
✅ trainers module imported
✅ data.preprocessing module imported

... (more tests) ...

============================================================
TEST SUMMARY
============================================================
✅ PASSED - Module Imports
✅ PASSED - Configuration
✅ PASSED - Model Instantiation
✅ PASSED - Loss Functions
✅ PASSED - Synthetic Data
✅ PASSED - Metrics
✅ PASSED - Device Handling
✅ PASSED - Random Seed
============================================================
Results: 8/8 tests passed
============================================================
```

### 8. Run Original --smoke Test
```bash
python scripts/train_stage1_synthetic.py --smoke
```
**Expected Output:**
```
Training Stage 1: PointNet++ Detection (Synthetic Data)
... (training progress) ...

✅ Stage 1 training complete
Results: experiments/T1_smoke/{checkpoint_*.pt, training_history.json, metrics.json}
```

---

## Output Folder Structure

After running all commands:
```
data/
├── processed/
│   ├── full.h5
│   ├── manifest.json
│   └── preprocessing_metadata.json
└── manifests/
    ├── train_manifest.json
    ├── val_manifest.json
    ├── test_manifest.json
    ├── splits_manifest.json
    ├── leakage_verification.txt
    └── preprocessing_metadata.json

experiments/
├── T0_leakage/
├── T1_detection_baseline/
├── T1_smoke/
├── T2_robustness/
├── T3_pinn_smoke/
├── T4_ablation/
└── T5_uncertainty_calibration/
```

---

## Sample Manifest Structure

### Training Manifest (data/manifests/train_manifest.json)
```json
{
  "split": "train",
  "num_samples": 28,
  "num_patients": 14,
  "unique_patients": [
    "synthetic_patient_000",
    "synthetic_patient_001",
    ...
  ],
  "label_distribution": {
    "negative": 20,
    "positive": 8
  },
  "samples": [
    {
      "index": 0,
      "patient_id": "synthetic_patient_000",
      "label": 1,
      "num_points": 8192,
      "num_features": 6
    },
    {
      "index": 1,
      "patient_id": "synthetic_patient_000",
      "label": 1,
      "num_points": 8192,
      "num_features": 6
    },
    ...
  ]
}
```

### Preprocessing Metadata (data/processed/preprocessing_metadata.json)
```json
{
  "dataset": "synthetic",
  "processed_samples": 40,
  "failed_samples": 0,
  "num_points": 8192,
  "normalization_method": "unit_sphere",
  "total_patients": 20,
  "output_file": "data/processed/full.h5",
  "manifest_file": "data/processed/manifest.json",
  "seed": 42
}
```

---

## Troubleshooting

### "Module not found" errors
```bash
# Make sure you're in the repo root
cd /path/to/NeuroFlow-Diagnostics

# Verify imports work
python -c "from data.adapters import BaseDatasetAdapter; print('✅ OK')"
```

### "IntrA not found" warning (expected)
```
⚠️  IntrA dataset not available: IntrA dataset directory not found...
→ Falling back to synthetic dataset for pipeline testing
```
This is **expected and OK**. The system automatically uses synthetic data.

### Permissions errors when creating directories
```bash
# Ensure you have write permissions
chmod -R u+w data/ experiments/
```

### Memory issues with large datasets
```bash
# Use smaller point cloud size
python scripts/preprocess_datasets.py --num-points 2048 ...

# Use smaller batch
python scripts/preprocess_datasets.py --n-patients 10 ...
```

---

## Success Criteria

✅ All 8 existing tests in test_project.py pass  
✅ verify_r2.py runs without errors  
✅ test_r2_implementations.py passes all tests  
✅ preprocess_datasets.py generates full.h5 with manifests  
✅ create_data_splits.py generates frozen split manifests  
✅ Leakage verification shows "NO LEAKAGE DETECTED"  
✅ Original --smoke test still works  

If all above pass, **R2 is ready for R3**.

---

## Next Steps (Milestone R3)

Once R2 is verified, proceed to R3:

1. Integrate splits into Stage 1 training script
2. Load data from split manifests instead of generating inline
3. Evaluate metrics on frozen test set
4. Create T0 leakage verification experiment
5. Export metrics to experiments/T1_detection_baseline/

### R3 Command (to be implemented)
```bash
python scripts/train_stage1_synthetic.py \
  --train-manifest data/manifests/train_manifest.json \
  --val-manifest data/manifests/val_manifest.json \
  --test-manifest data/manifests/test_manifest.json
```

---

**Status:** ✅ Milestone R2 Complete  
**Ready for R3:** ✅ YES
