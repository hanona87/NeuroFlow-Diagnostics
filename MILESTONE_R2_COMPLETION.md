# Milestone R2: Real/Semi-Real Data Path

## Overview

Milestone R2 implements a **data adapter interface** for plugging in different aneurysm datasets with graceful fallback to synthetic data. This ensures the pipeline always works even when real datasets are unavailable.

## Completed Components

### 1. Data Adapter Interface (`data/adapters/`)

**BaseDatasetAdapter** — Abstract interface for dataset adapters:
- `discover_samples()` — Find all samples in dataset
- `load_mesh()` — Load mesh for a sample
- `get_rupture_label()` — Get rupture classification if available
- `validate_dataset()` — Check dataset integrity
- `save/load_manifest()` — Frozen manifests for reproducibility

**IntraAdapter** — Adapter for IntrA dataset:
- Auto-discovers IntrA if present at `./data/datasets/intra/`
- Extracts patient/aneurysm IDs from mesh filenames
- Reads rupture labels from `metadata.json` if available
- **Gracefully falls back to synthetic data if IntrA not found**
- Clear setup instructions if IntrA is missing

**SyntheticAdapter** — Synthetic dataset for testing/fallback:
- Generates configurable number of patients with multiple aneurysms per patient
- Assigns rupture labels with configurable prevalence
- Uses **identical patient-level structure as real data** for protocol compatibility
- Always available, no external dependencies

### 2. Preprocessing CLI (`scripts/preprocess_datasets.py`)

Converts meshes to standardized point cloud datasets:

```bash
# Synthetic dataset (no dependencies):
python scripts/preprocess_datasets.py --dataset synthetic --n-patients 20 --output-dir ./data/processed

# IntrA dataset (if available):
python scripts/preprocess_datasets.py --dataset intra --data-root ./data/datasets/intra

# Custom point cloud size:
python scripts/preprocess_datasets.py --dataset synthetic --num-points 2048 --n-patients 20
```

**Features:**
- Loads meshes via adapters
- Samples points uniformly from mesh surface
- Applies Farthest Point Sampling (FPS) to fixed size (default 8192)
- Computes surface normals (k-NN based)
- Normalizes to unit sphere
- Preserves metadata (patient_id, rupture_label, source)
- Saves to HDF5 with compression
- Generates manifest JSON

**Output:**
```
data/processed/
├── full.h5                      (point clouds + labels + metadata)
├── manifest.json                (dataset metadata)
└── preprocessing_metadata.json   (processing parameters)
```

### 3. Data Splitting CLI (`scripts/create_data_splits.py`)

Creates patient-level train/val/test splits with leakage verification:

```bash
# Create splits from preprocessed data:
python scripts/create_data_splits.py --input ./data/processed/full.h5

# Custom split ratios:
python scripts/create_data_splits.py \\
  --input ./data/processed/full.h5 \\
  --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2
```

**Features:**
- Patient-level splitting (no data leakage)
- Mandatory leakage verification (asserts no patient in multiple splits)
- Stratified splitting by rupture label (optional)
- Frozen manifests for reproducibility
- Creates experiment directory structure (T0–T5)

**Output:**
```
data/manifests/
├── train_manifest.json
├── val_manifest.json
├── test_manifest.json
├── splits_manifest.json      (combined)
├── leakage_verification.txt
└── preprocessing_metadata.json
```

## Workflow

### Step 1: Preprocess Data
```bash
python scripts/preprocess_datasets.py --dataset synthetic --n-patients 50
# Output: data/processed/full.h5 + manifest.json
```

### Step 2: Create Splits
```bash
python scripts/create_data_splits.py --input data/processed/full.h5
# Output: data/manifests/ with frozen split manifests
```

### Step 3: Train (use splits in Stage 1)
```bash
python scripts/train_stage1_synthetic.py --data-manifest data/manifests/train_manifest.json
# (Integration to follow in R3)
```

## IntrA Dataset Setup (Optional)

If you have access to the IntrA dataset from Université de Strasbourg:

1. Download from https://github.com/rjdmoore/IntrA
2. Extract to `./data/datasets/intra/`
3. Expected structure:
   ```
   data/datasets/intra/
   ├── images/          (CT/MR scans)
   ├── segmentations/   (vessel segmentations)
   ├── surfaces/        (3D mesh files in STL/OBJ/VTK format)
   └── metadata.json    (patient demographics + rupture labels)
   ```
4. Run preprocessing:
   ```bash
   python scripts/preprocess_datasets.py --dataset intra --output-dir ./data/processed_intra
   ```

**If IntrA is not available:**
- Preprocessing script automatically falls back to synthetic data
- Pipeline remains fully functional with synthetic samples
- Identical patient-level structure ensures evaluation protocols work the same way

## Manifest Structure

**Sample manifest entry:**
```json
{
  "patient_id": "synthetic_patient_001",
  "aneurysm_id": "aneurysm_01",
  "source": "synthetic",
  "file_path": "synthetic://synthetic_patient_001/aneurysm_01",
  "file_hash": "synthetic_0_1",
  "rupture_label": 1,
  "modality": "synthetic point cloud",
  "notes": "Generated for testing/fallback"
}
```

**Split manifest example:**
```json
{
  "split": "train",
  "num_samples": 140,
  "num_patients": 35,
  "unique_patients": ["synthetic_patient_001", ...],
  "label_distribution": {
    "negative": 100,
    "positive": 40
  },
  "samples": [
    {
      "index": 0,
      "patient_id": "synthetic_patient_001",
      "label": 1,
      "num_points": 8192,
      "num_features": 6
    },
    ...
  ]
}
```

## Code Integration

New modules added:
- `data/adapters/` — Dataset adapter implementations
- `scripts/preprocess_datasets.py` — Preprocessing CLI
- `scripts/create_data_splits.py` — Splitting CLI

Updated modules:
- `data/__init__.py` — Now exports adapters module

## Key Design Decisions

1. **Graceful Degradation:** IntrA adapter automatically falls back to synthetic if not available
   - Users get clear error message explaining how to install IntrA
   - Pipeline continues working with synthetic data
   - No blocking on external dataset availability

2. **Patient-Level Splits:** All splits are at patient level
   - Prevents data leakage (critical for medical ML)
   - Leakage checks are mandatory (asserts in code)
   - Applies to both real and synthetic data equally

3. **Frozen Manifests:** Split manifests are JSON files
   - Enable reproducibility across runs
   - Can be version-controlled
   - Facilitate external validation

4. **Synthetic Mirrors Real Structure:**
   - Synthetic adapter generates patient/aneurysm groups like real data
   - Same preprocessing pipeline works for both
   - Easy transition when real data becomes available

## Testing Without IntrA

The entire R2 implementation is testable without IntrA:

```bash
# 1. Generate synthetic point clouds
python scripts/preprocess_datasets.py --dataset synthetic --n-patients 20

# 2. Create splits with leakage checks
python scripts/create_data_splits.py --input data/processed/full.h5

# 3. Verify no leakage
cat data/manifests/leakage_verification.txt
```

No external dependencies required beyond what's already in `requirements.txt`.

## What's Next (R3)

- Integrate splits into Stage 1 training script
- Load data from split manifests instead of generating synthetic on-the-fly
- Evaluate metrics on frozen test set
- Document how to verify no test-set overfitting

## Backward Compatibility

The existing `--smoke` test remains unchanged:
- `scripts/train_stage1_synthetic.py --smoke` still works
- Generates its own synthetic dataset inline
- Does not depend on R2 adapters or splits
- This ensures no regression in existing pipeline

## Hard Constraints Met

✅ Patient-level leakage checks mandatory (asserts in code)  
✅ Graceful IntrA fallback (no blocking on external data)  
✅ Metadata preservation (patient_id, rupture_label, source)  
✅ Reproducible splits (frozen manifests)  
✅ No invented data (synthetic is clearly labeled)  
✅ No torch_scatter/torch_geometric reintroduced
