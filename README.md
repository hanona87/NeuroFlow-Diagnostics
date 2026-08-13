# NeuroFlow-Diagnostics: Cerebral Aneurysm Detection & Hemodynamic Rupture Risk Assessment

**A research-grade pipeline combining deep learning and physics-informed neural networks for aneurysm detection and rupture risk prediction.**

**Status**: ISEF/Science Fair ready (Phase A stable). No clinical claims. Synthetic validation only.

---

## Project Overview

NeuroFlow-Diagnostics implements a **three-stage pipeline**:

1. **Stage 1 (Detection)**: PointNet++ classifier on vessel point clouds → binary aneurysm detection
2. **Stage 2 (Hemodynamics)**: Physics-Informed Neural Network (PINN) for incompressible Navier-Stokes → WSS, OSI, RRT indices
3. **Stage 3 (Rupture Risk)**: Multichannel PointNet++ fusing geometry + hemodynamic features → rupture probability

**Current Status**:
- ✅ Stage 1 & 2 architecture complete, tested on synthetic data
- ✅ Physics residuals (continuity, momentum) implemented correctly
- ✅ Hemodynamic indices (TAWSS, OSI, RRT) following standard definitions
- ✅ Patient-level leakage control in data splits
- ⚠️ Trained only on synthetic data (no clinical validation)
- ❌ Stage 3 not yet trained
- ❌ No external clinical dataset validation

For detailed status, see [PROJECT_STATUS_REAL.md](PROJECT_STATUS_REAL.md).

---

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo_url>
cd NeuroFlow-Diagnostics

# Create environment (recommended: Python 3.10+)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

**Required Dependencies** (key packages):
- `torch >= 2.2.0` — PyTorch
- `torch-geometric >= 2.5.0` — Graph neural networks
- `pytorch-lightning >= 2.2.0` — Training utilities
- `numpy, scipy, scikit-learn` — Scientific computing
- `monai >= 1.3.0` — Medical imaging (optional)

### Run Tests

```bash
# Test all imports and model instantiation
python test_project.py

# Test patient-level leakage control
python tests/test_leakage.py

# Test PINN physics residuals
python tests/test_physics_residuals.py
```

Expected output: All tests pass, residuals compute correctly.

---

## Run Experiments

### Stage 1: Synthetic Aneurysm Detection

Train PointNet++ on 200 synthetic point cloud samples with patient-level splits:

```bash
python scripts/train_stage1_synthetic.py
```

**Output**: `experiments/T1_detection_baseline/`
- `checkpoint_best.pt` — Best model weights
- `metrics.json` — ROC-AUC, PR-AUC, accuracy, sensitivity, specificity, F1
- `training_history.json` — Loss curves per epoch

**Expected metrics** (synthetic data):
- ROC-AUC: ~0.90–0.95 (data is well-separated, not clinically realistic)
- Accuracy: ~90%

### Stage 2: PINN Physics Training Smoke Test

Quick validation of PINN residual training:

```bash
python scripts/run_pinn_smoke.py
```

**Output**: `experiments/T3_pinn_smoke/`
- `residual_history.json` — Per-step continuity, momentum residuals
- `model_checkpoint.pt` — Trained PINN weights

**Validates**:
- Forward pass through PINN ✓
- Residual computation (continuity, momentum-x/y/z) ✓
- Gradient-based optimization ✓
- Hemodynamic calculators (TAWSS, OSI, RRT) ✓

---

## Project Structure

```
NeuroFlow-Diagnostics/
├── models/
│   ├── pointnet2.py              # PointNet++ classification/segmentation
│   ├── pinn.py                   # Physics-Informed Neural Network + Navier-Stokes
│   └── multichannel_pointnet2.py # Geometry + hemodynamics fusion
├── data/
│   └── preprocessing/
│       └── preprocessing.py      # Mesh to point cloud, normalization, augmentation
├── losses/
│   └── losses.py                 # Focal loss, physics loss, weighted CE
├── evaluation/
│   └── metrics.py                # Classification + calibration metrics
├── trainers/
│   └── trainer.py                # Base trainer, checkpoint management
├── scripts/
│   ├── train_stage1_synthetic.py # Train detection on synthetic data
│   └── run_pinn_smoke.py         # PINN smoke test
├── tests/
│   ├── test_leakage.py           # Patient-level split validation
│   ├── test_physics_residuals.py # PINN residual shapes/magnitudes
│   └── (others inherit from test_project.py)
├── configs/
│   └── config.yaml               # Hyperparameter defaults
├── utils.py                      # Device, seeding, normalization, split utilities
├── requirements.txt              # Python dependencies
└── [docs]
    ├── README.md                 # This file
    └── PROJECT_STATUS_REAL.md    # Detailed status (what works, what doesn't)
```

---

## Core Features

### PointNet++ Classification (Stage 1)

- **Architecture**: 4 Set Abstraction layers (2048→512→128→32 points) + classification head
- **SA radii**: 0.05, 0.1, 0.2, 0.4 m (research specification)
- **Input**: Point clouds (N, 6) with geometry + surface normals
- **Output**: Binary logits (B, 2) → softmax → probability

**Robustness**:
- Patient-level train/val/test split (no leakage)
- Data augmentation: rotation, jitter, dropout, scaling
- Batch normalization in feature encoding

### PINN Solver (Stage 2)

- **Network**: Tanh MLP (4→[64,64,64]→4)
- **Input**: (x, y, z, t) coordinates
- **Output**: (u, v, w, p) — velocity & pressure fields
- **Physics Constraints**:
  - Continuity: ∇·u = 0
  - Momentum: ∂u/∂t + (u·∇)u + ∇p/ρ - ν∇²u = 0
  - Domain: aneurysm vessel (~0.01 m scale)
  - Cycle: T = 0.8 s (cardiac frequency)

**Fluid Properties** (blood):
- Density ρ = 1050 kg/m³
- Dynamic viscosity μ = 3.85e-3 Pa·s
- Kinematic viscosity ν = μ/ρ = 3.66e-6 m²/s
- Reynolds number: Re ~ 100–200 (laminar, transitional)

**Hemodynamic Indices**:
- **TAWSS** (Time-Averaged WSS): (1/T) ∫ |τ_w| dt → rupture risk marker
- **OSI** (Oscillatory Shear Index): 0.5 (1 - |∫ WSS_vec dt| / ∫ |WSS_vec| dt) → flow quality
- **RRT** (Relative Residence Time): 1 / (TAWSS · (1 - 2·OSI)) → stagnation risk

### Multichannel Fusion (Stage 3)

- **Input Channels**: Geometry (x, y, z, nx, ny, nz) + hemodynamics (u, v, w, p, WSS, OSI, RRT)
- **Ablation Modes**: Geometry-only → +velocity → +pressure → +WSS → +OSI → full multichannel
- **Output**: Rupture probability per point (or per aneurysm)

---

## Important Notes

### ✅ What Works (Validated on Synthetic Data)

1. **Model instantiation & forward passes** — All components run without error
2. **Synthetic data generation** — Reproducible dataset creation with patient grouping
3. **Patient-level leakage control** — No overlap between train/val/test splits
4. **PointNet++ pipeline** — Training, validation, metric computation
5. **PINN physics residuals** — Continuity & momentum equations compute correctly
6. **Hemodynamic calculations** — TAWSS, OSI, RRT follow standard formulas
7. **Gradient-based optimization** — Residuals decrease during PINN training

### ⚠️ What Is Approximate

1. **Hemodynamic calculator**: Uses simplified WSS estimate (velocity magnitude × viscosity), not full velocity gradient
2. **PINN boundary conditions**: Not explicitly enforced; relies on data loss term (IC/BC sampled)
3. **Synthetic aneurysm shape**: Simplified geometry (bulge on cylinder), not realistic CFD
4. **Training hyperparameters**: Not tuned on real data; defaults are exploratory

### ❌ What Is Not Done

1. **No real patient dataset** — Only synthetic point clouds
2. **No external validation** — No held-out clinical cohort
3. **No Stage 3 training** — Multichannel model architecture only
4. **No comparison baselines** — No vanilla PointNet, DGCNN, or classical ML
5. **No uncertainty quantification** — No Bayesian or ensemble uncertainty
6. **No interpretability** — No saliency maps or attention visualization
7. **No production deployment** — No deployment script or inference API
8. **No regulatory path** — Not FDA/CE cleared; research only

---

## Configuration

### Hyperparameters (configs/config.yaml)

Key settings:
- **num_points**: 8192 (point cloud resolution)
- **batch_size**: 16–32
- **learning_rate**: 1e-3 (Adam)
- **epochs**: 30–100
- **dropout**: 0.5 (classification head)

For a quick test, use smaller batches and fewer epochs:
```yaml
num_points: 2048
batch_size: 8
epochs: 10
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ImportError: torch_scatter` | Install via `pip install torch-scatter` or use PyG prebuilt wheels |
| CUDA out of memory | Reduce `batch_size` or `num_points` |
| Slow data loading | Set `num_workers=0` (for debug), or adjust worker count |
| NaN in PINN loss | Reduce learning rate; check domain bounds in residual calculator |

---

## Contributing & Development

**To add a new experiment**:

1. Create `scripts/run_exp_<name>.py` with clear input/output documentation
2. Log metrics to JSON in `experiments/<tag>/<name>/`
3. Add a corresponding test in `tests/test_<name>.py`
4. Update this README with results

**To modify core models**:

- Update docstrings with input/output shapes
- Add unit tests in `tests/test_physics_residuals.py` for PINN changes
- Document any changes to fluid properties in comments

---

## Citation & Attribution

**Reference Publications** (not implemented here, but inspiration):

- PointNet++: Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space" (NIPS 2017)
- PINN: Raissi et al., "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems" (J. Comp. Phys. 2019)
- Aneurysm biomechanics: Cebral et al., "Hemodynamic characterization of cerebral aneurysms" (AJNR 2015)

---

## License

MIT (see LICENSE file)

---

## Contact & Questions

For questions about this ISEF project:
- Check [PROJECT_STATUS_REAL.md](PROJECT_STATUS_REAL.md) for detailed status
- Review docstrings in `models/pointnet2.py` and `models/pinn.py`
- Run smoke tests (`test_project.py`, `tests/test_physics_residuals.py`) for validation

**Disclaimer**: This is a research/education project. No clinical claims. Not for medical use without proper validation and regulatory approval.
