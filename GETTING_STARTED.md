# Getting Started with NeuroFlow-Diagnostics

A step-by-step guide to setting up, configuring, and running the complete pipeline.

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU (V100+ recommended, or CPU for testing)
- 16+ GB RAM (32 GB with GPU training)
- 50 GB disk space (for datasets and checkpoints)

### Step 1: Clone Repository
```bash
cd /workspaces
git clone https://github.com/your-repo/NeuroFlow-Diagnostics.git
cd NeuroFlow-Diagnostics
```

### Step 2: Create Environment

**Option A: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate neuroflow-diagnostics

# Verify installation
python -c "import torch; print(f'✅ PyTorch {torch.__version__}')"
python -c "import torch_geometric; print('✅ PyTorch Geometric OK')"
```

**Option B: pip (In virtual environment)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.__version__)"
```

### Step 3: Verify Installation
```bash
python scripts/test_components.py
```

Expected output:
```
✅ PyTorch available
✅ CUDA available (if GPU present)
✅ All models can be instantiated
✅ All modules import successfully
```

---

## 📁 Directory Setup

### Automatic (via script)
```bash
python -c "
from utils import create_directories
dirs = create_directories('.', [
    'data/datasets',
    'data/processed',
    'data/manifests',
    'experiments',
    'results',
    'reports/figures',
    'reports/tables',
    'checkpoints'
])
print('✅ Directories created')
"
```

### Manual
```bash
mkdir -p data/{datasets,processed,manifests}
mkdir -p experiments results reports/{figures,tables} checkpoints
```

---

## 🔧 Configuration

### 1. Review Master Config
```bash
# View main configuration
cat configs/config.yaml | head -100

# Key sections to review:
# - reproducibility.seed (line 4)
# - data.num_points (line 15)
# - stage1_detection.training (line 45)
# - hardware.device (line 230)
```

### 2. Customize for Your Setup

**For CPU-only (slower, for debugging)**:
```yaml
# In configs/config.yaml
hardware:
  device: "cpu"
  gpu_ids: []
  num_workers: 0

# And reduce parameters
data:
  num_points: 2048  # Less memory
  splits:
    train_ratio: 0.5  # Smaller dataset
```

**For Multi-GPU**:
```yaml
hardware:
  device: "cuda"
  gpu_ids: [0, 1, 2, 3]  # Use 4 GPUs
  num_workers: 8
```

**For Limited Memory** (< 8 GB GPU):
```yaml
stage1_detection:
  training:
    batch_size: 8  # From 20

data:
  num_points: 4096  # From 8192
```

---

## 🎯 Quick Demo (5 minutes)

### Run on Synthetic Data
```bash
python scripts/train_stage1.py \
    --config configs/config.yaml \
    --experiment demo_quick \
    --device cpu  # Use GPU if available
```

**Expected Output**:
```
Experiment: demo_quick
Device: cpu
Starting training for 200 epochs...
Epoch 1/200 | Train Loss: 0.6892 | Val Loss: 0.6541
Epoch 2/200 | Train Loss: 0.5234 | Val Loss: 0.4982
...
Epoch 200/200 | Train Loss: 0.0234 | Val Loss: 0.0412

Test set results saved to experiments/demo_quick/results/
```

**Check Results**:
```bash
# View metrics
cat experiments/demo_quick/results/results.json

# Expected structure:
# {
#   "training_results": {...},
#   "test_metrics": {
#     "auc": 0.8543,
#     "accuracy": 0.8234,
#     ...
#   }
# }
```

---

## 📊 Working with Real Data

### Step 1: Prepare Mesh Files
```bash
# Organize your mesh files:
mkdir -p data/datasets/intra
mkdir -p data/datasets/aneumo

# Copy mesh files (.stl, .obj, .vtk)
cp /path/to/meshes/*.stl data/datasets/intra/
```

### Step 2: Create Data Processing Script
```python
# process_data.py
from data.preprocessing import PointCloudPreprocessor, PointCloudDatasetWriter
from pathlib import Path
import pandas as pd

# Load mesh metadata
metadata = pd.read_csv('data/datasets/intra/metadata.csv')
# Expected columns: patient_id, aneurysm_id, mesh_file, label (ruptured: 0/1)

# Initialize preprocessor
preprocessor = PointCloudPreprocessor(
    num_points=8192,
    normalization_method='unit_sphere',
    compute_normals_flag=True,
    seed=42
)

# Process each mesh
samples = []
for idx, row in metadata.iterrows():
    mesh_path = f"data/datasets/intra/{row['mesh_file']}"
    
    try:
        points, norm_params = preprocessor.process_mesh(mesh_path)
        samples.append({
            'points': points,
            'label': int(row['label']),
            'patient_id': row['patient_id'],
            'aneurysm_id': row['aneurysm_id'],
            'normalization_method': 'unit_sphere'
        })
    except Exception as e:
        print(f"Error processing {mesh_path}: {e}")

print(f"Processed {len(samples)} samples")

# Write to HDF5
writer = PointCloudDatasetWriter('data/processed')
writer.write_split('full', samples)
```

### Step 3: Create Data Splits
```python
# split_data.py
from utils import split_by_patient
import h5py

# Load all patient IDs
with h5py.File('data/processed/full.h5', 'r') as f:
    patient_ids = [pid.decode() for pid in f['patient_ids'][:]]

# Split by patient (no leakage)
train_patients, val_patients, test_patients = split_by_patient(
    patient_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    seed=42
)

print(f"Train: {len(train_patients)} patients")
print(f"Val: {len(val_patients)} patients")
print(f"Test: {len(test_patients)} patients")

# Create split manifests
manifest = {
    'train': train_patients,
    'val': val_patients,
    'test': test_patients
}

import json
with open('data/manifests/split_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
```

### Step 4: Update Configuration
```yaml
# configs/config.yaml
data:
  datasets_root: "./data/datasets"
  processed_dir: "./data/processed"
  primary_dataset: "intra"

stage1_detection:
  training:
    epochs: 200  # Adjust based on dataset size
    batch_size: 20
```

### Step 5: Train on Real Data
```bash
python scripts/train_stage1.py \
    --config configs/config.yaml \
    --experiment stage1_intra_v1 \
    --device cuda
```

---

## 🔬 Running Individual Stages

### Stage 1: Aneurysm Detection
```bash
python scripts/train_stage1.py \
    --config configs/config.yaml \
    --experiment stage1_v1 \
    --device cuda
```

**Monitor Training**:
```bash
# Watch training log in real-time
tail -f experiments/stage1_v1/training_log.jsonl

# Plot losses
python -c "
import json
log = [json.loads(l) for l in open('experiments/stage1_v1/training_log.jsonl')]
import matplotlib.pyplot as plt
plt.plot([l['train_loss'] for l in log], label='train')
plt.plot([l['val_loss'] for l in log], label='val')
plt.legend()
plt.savefig('training_curves.png')
"
```

### Stage 2: Physics-Informed NN
```bash
python scripts/train_stage2_pinn.py \
    --config configs/config.yaml \
    --experiment stage2_pinn_v1 \
    --num_epochs 14100 \
    --device cuda
```

**Key Outputs**:
```
experiments/stage2_pinn_v1/
├── checkpoints/
│   ├── checkpoint_epoch_0100.pt  # After Adam training
│   └── checkpoint_epoch_14100.pt # After L-BFGS-B
├── results/
│   ├── velocity_field.npy
│   ├── pressure_field.npy
│   ├── physics_residuals.json
│   └── hemodynamic_fields.h5
└── training_log.jsonl
```

### Stage 3: Rupture Prediction
```bash
python scripts/train_stage3_rupture.py \
    --config configs/config.yaml \
    --experiment stage3_rupture_v1 \
    --ablation_type "all" \  # Run all ablations
    --device cuda
```

**Ablation Results**:
```
experiments/stage3_rupture_v1/
├── ablation_1_geometry_only/
│   └── results.json  → AUC ~0.62
├── ablation_2_geometry_velocity/
│   └── results.json
├── ...
├── ablation_7_full_multichannel/
│   └── results.json  → AUC ≥0.75 (target)
└── ablation_summary.json  → Compare all ablations
```

---

## 🧪 Running Experimental Trials (T0-T13)

### All Trials at Once
```bash
python scripts/run_all_trials.py \
    --config configs/config.yaml \
    --output_dir ./results \
    --device cuda \
    --num_workers 4
```

**Expected Duration**: ~48-72 hours on V100 GPU

### Individual Trials
```bash
# T0: Data audit
python scripts/experiment_runners/experiment_T0_data_audit.py

# T1: Detector baseline
python scripts/experiment_runners/experiment_T1_detector_baseline.py

# T8: Rupture primary (multichannel)
python scripts/experiment_runners/experiment_T8_rupture_multichannel.py
```

---

## 📊 Viewing & Interpreting Results

### Basic Metrics
```python
import json
import numpy as np

# Load results
with open('experiments/stage1_v1/results/results.json') as f:
    results = json.load(f)

# Print key metrics
print(f"AUC: {results['test_metrics']['auc']:.4f}")
print(f"Accuracy: {results['test_metrics']['accuracy']:.4f}")
print(f"Sensitivity: {results['test_metrics']['sensitivity']:.4f}")
print(f"Specificity: {results['test_metrics']['specificity']:.4f}")
print(f"ECE: {results['test_metrics']['ece']:.4f}")

# Bootstrap confidence intervals
bootstrap_ci = results['test_metrics']['bootstrap_ci']
auc_ci = bootstrap_ci.get('auc', None)
print(f"AUC 95% CI: {auc_ci}")
```

### ROC Curve
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load predictions
data = np.load('experiments/stage1_v1/results/predictions.npz')
predictions = data['predictions']
targets = data['targets']

# Plot
fpr, tpr, _ = roc_curve(targets, predictions)
plt.plot(fpr, tpr, label=f'AUC={auc(fpr, tpr):.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('roc_curve.png')
```

### Calibration Plot
```python
from evaluation import CalibrationMetrics
import matplotlib.pyplot as plt

# Compute calibration curve
mean_preds, frac_pos = CalibrationMetrics.calibration_curve(predictions, targets)

# Plot
plt.plot(mean_preds, frac_pos, 'o-', label='Model')
plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.savefig('calibration.png')
```

---

## 🐛 Troubleshooting

### Installation Issues

**"No module named torch"**:
```bash
# Verify conda/pip environment is activated
conda activate neuroflow-diagnostics

# Reinstall if needed
pip install --force-reinstall torch==2.1.2
```

**"CUDA out of memory"**:
```bash
# Option 1: Reduce batch size
# In configs/config.yaml:
# stage1_detection.training.batch_size: 10

# Option 2: Reduce point count
# data.num_points: 4096

# Option 3: Use CPU
python scripts/train_stage1.py --device cpu
```

### Data Issues

**"Dataset file not found"**:
```bash
# Check data directory
ls -la data/processed/

# Check config path
grep "processed_dir" configs/config.yaml

# Create synthetic if needed
python -c "from data.preprocessing import create_synthetic_dataset; create_synthetic_dataset()"
```

**"Invalid HDF5 file"**:
```bash
# Check file integrity
python -c "import h5py; f = h5py.File('data/processed/train.h5'); print(list(f.keys()))"

# Recreate if corrupted
python process_data.py
```

### Training Issues

**"Model doesn't converge"**:
```bash
# Check loss curve
tail -20 experiments/stage1_v1/training_log.jsonl

# If loss is increasing: reduce learning rate
# In configs/config.yaml:
# stage1_detection.training.learning_rate: 1e-6

# If loss is stagnant: increase batch size
# stage1_detection.training.batch_size: 32
```

**"Out of disk space"**:
```bash
# Check disk usage
du -sh experiments/
du -sh data/

# Clean old checkpoints
find experiments/ -name "checkpoint_epoch_*.pt" -delete

# Keep only best models
find experiments/ -type f -not -name "best_model.pt" -delete
```

---

## 📈 Performance Optimization

### GPU Acceleration
```yaml
# configs/config.yaml
hardware:
  device: "cuda"
  gpu_ids: [0]
  pin_memory: true
  num_workers: 4
```

### Batch Processing
```yaml
stage1_detection:
  training:
    batch_size: 32  # Larger = faster, but more memory
    gradient_accumulation_steps: 2  # Effective batch: 64
```

### Mixed Precision Training (Optional)
```yaml
hardware:
  mixed_precision: true  # Reduces memory, speeds up (requires AMP)
```

---

## 📝 Reproducibility Checklist

Before publishing results:

- [ ] Record Python version: `python --version`
- [ ] Record package versions: `pip freeze > versions.txt`
- [ ] Record seed: In config.yaml reproducibility.seed
- [ ] Record device: CPU/GPU model in config
- [ ] Record exact data splits (data/manifests/)
- [ ] Save all configs used (copy configs/ to results/)
- [ ] Save trained models (checkpoints/)
- [ ] Save training logs (results/training_log.jsonl)
- [ ] Save predictions (results/predictions.npz)
- [ ] Document any modifications to code
- [ ] Run on multiple seeds to verify stability

---

## 🎓 What's Next?

1. **Explore the Code**: Review architecture in `models/`
2. **Understand Metrics**: See `evaluation/metrics.py` for all metrics
3. **Customize Training**: Modify `configs/config.yaml`
4. **Extend Models**: Add new architectures to `models/`
5. **Publish Results**: Generate report with `scripts/generate_report.py`

---

## 📚 Additional Resources

- **Paper References**: See README.md
- **Architecture Diagrams**: `reports/figures/architecture.png`
- **Detailed Metrics**: `STATUS.md`
- **Code API**: Docstrings in each module

---

**Ready to begin?** Start with: `python scripts/train_stage1.py --experiment quick_demo`

For more help, see STATUS.md or check specific module docstrings.
