# Dataset Requirements Matrix for NeuroFlow Experiments

**Date**: August 13, 2026  
**Status**: Assessment based on dataset research  
**Format**: CSV-compatible matrix for experiment planning

---

## Overview

This matrix documents the scientific data requirements for each NeuroFlow experiment (T0-T13) and maps them against candidate datasets.

**Legend**:
- ✅ **YES** — Dataset has this capability; experiment can proceed
- ⚠️ **PARTIAL** — Dataset has limited version; experiment proceeds with caveats
- ❌ **NO** — Dataset lacks this capability; experiment blocked
- ❓ **UNKNOWN** — Verification required
- **N/A** — Experiment skipped (no real data, or scientific blocker)

---

## Experiment Requirements Matrix

### Core Data Requirements (All Experiments)

| Data Requirement | Type | IntrA | Aneumo | Synthetic |
|---|---|---|---|---|
| **Geometry** | 3D mesh files (STL/OBJ/VTK) | ✅ YES | ⚠️ PARTIAL | ✅ YES |
| **Patient ID** | Unique identifier per patient | ⚠️ LIKELY | ❓ UNKNOWN | ✅ YES |
| **Aneurysm ID** | Unique identifier per aneurysm | ⚠️ LIKELY | ❓ UNKNOWN | ✅ YES |
| **Study ID** | Unique identifier per imaging session | ⚠️ LIKELY | ❓ UNKNOWN | ✅ YES |
| **File Integrity** | Mesh validity (no disconnected components, no degeneracies) | ✅ LIKELY | ❓ UNKNOWN | ✅ YES |
| **Normalization** | Can be unit-sphere normalized | ✅ YES | ✅ YES | ✅ YES |
| **Point Cloud Sampling** | FPS to 8192 points possible | ✅ YES | ✅ YES | ✅ YES |
| **Normal Estimation** | Surface normals computable | ✅ YES | ✅ YES | ✅ YES |

---

## T0: Dataset Audit & Governance

| Capability | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **File Existence Check** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Verify all geometry files present |
| **File Integrity** | ✅ LIKELY | ❓ UNKNOWN | ✅ YES | Check for mesh defects |
| **Duplicate Detection** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Compare by patient/study/aneurysm ID |
| **Patient-Level Leakage** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Ensure patient groups don't split across train/val/test |
| **Class Balance** | ✅ LIKELY | ❓ UNKNOWN | ✅ YES | Analyze rupture status distribution (if available) |
| **Missing Data Check** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Document all absent fields |
| **Metadata Completeness** | ⚠️ PARTIAL | ❓ UNKNOWN | ✅ YES | Verify patient ID, study ID, aneurysm ID encoding |
| **Reproducibility Setup** | ✅ YES | ✅ YES | ✅ YES | Seed, manifest hash, split version |
| **Splitting Feasibility** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Train/val/test split possible without leakage |

**T0 Status**: 
- **If IntrA**: ✅ EXECUTABLE
- **If Synthetic only**: ✅ EXECUTABLE (but limited scope)

---

## T1: Real-Data Aneurysm Detection (PointNet++)

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Input Geometry** | ✅ YES | ✅ YES | ✅ YES | 3D vascular/aneurysm point clouds |
| **Supervised Label** | ✅ YES | ✅ YES | ✅ YES | Binary: aneurysm (1) vs. normal vessel (0) |
| **Label Definition** | ✅ CLEAR | ⚠️ UNCLEAR | ✅ CLEAR | Aneurysm sac detection task |
| **Sample Size** | ✅ YES (100-400 aneurysms) | ⚠️ SMALL? | ✅ YES (synthetic) | Sufficient for training (rough guidance: >100) |
| **Patient-Level Split** | ✅ YES | ⚠️ PARTIAL | ✅ YES | No patient overlap in train/val/test |
| **Balanced Classes** | ⚠️ LIKELY | ❓ UNKNOWN | ✅ YES | IntrA: mostly diseased; balance may be skewed |
| **Preprocessing Frozen** | ✅ YES | ✅ YES | ✅ YES | Same pipeline for all splits |
| **Test Set Lock** | ✅ YES | ✅ YES | ✅ YES | Separate, untouched during hyperparameter tuning |

**T1 Status**:
- **If IntrA**: ✅ READY
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T2: Robustness Analysis

| Robustness Type | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Geometric Perturbations** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Natural variation in real geometry |
| **Point Density Variation** | ✅ YES | ✅ YES | ✅ YES | Resample to different densities |
| **Coordinate Noise** | ✅ YES | ✅ YES | ✅ YES | Synthetic perturbation |
| **Boundary Perturbation** | ⚠️ POSSIBLE | ❓ UNKNOWN | ✅ YES | Requires segmentation or geometric warping |
| **Multiple Seeds** | ✅ YES | ✅ YES | ✅ YES | Retrain with different random seeds |
| **Cross-Validation** | ✅ YES | ✅ YES | ✅ YES | 5-fold or repeated splits |

**T2 Status**:
- **If IntrA**: ✅ READY (using natural geometric variation + synthetic perturbations)
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T3: PINN / Hemodynamic Baseline

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Geometry for Domain** | ✅ YES | ✅ YES | ✅ YES | Vascular geometry defines spatial domain |
| **PDE Formulation** | ✅ YES | ✅ YES | ✅ YES | Incompressible Navier-Stokes independent of data |
| **Boundary Conditions** | ⚠️ ASSUMED | ⚠️ ASSUMED | ✅ SYNTHETIC | Typically assumed (inlet flow, outlet, walls) |
| **Collocation Points** | ✅ YES | ✅ YES | ✅ YES | Sample from domain |
| **Reference Flow Data** | ❌ NO (needed for validation, not for baseline) | ❌ NO | ❌ NO | T3 is unsupervised PINN baseline |
| **Training Feasibility** | ✅ YES | ✅ YES | ✅ YES | Can train PINN on geometry alone |
| **Synthetic Validation** | ✅ YES | ✅ YES | ✅ YES | Check physics residuals decrease |

**T3 Status**:
- **If IntrA geometry**: ✅ READY (PINN baseline training)
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T4: PINN Reference Validation

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Reference Velocity Field** | ✅ YES (CFD) | ❓ UNKNOWN | ❌ NO | Critical blocker for real validation |
| **Reference Pressure Field** | ✅ YES (CFD) | ❓ UNKNOWN | ❌ NO | Critical blocker for real validation |
| **Reference WSS** | ✅ YES (derived from CFD) | ❓ UNKNOWN | ❌ NO | Can be computed from velocity fields |
| **Reference OSI** | ✅ YES (derived from CFD) | ❓ UNKNOWN | ❌ NO | Can be computed from velocity fields |
| **Reference RRT** | ✅ YES (derived from CFD) | ❓ UNKNOWN | ❌ NO | Can be computed from velocity fields |
| **Matching Geometry** | ✅ YES | ❓ UNKNOWN | ❌ NO | PINN domain must match CFD geometry |
| **Field Comparison Metrics** | ✅ YES | ✅ YES | ⚠️ SYNTHETIC | L2 error, MAE, correlation, divergence |

**T4 Status**:
- **If IntrA + CFD data present**: ✅ READY (real validation)
- **If IntrA but no CFD**: ❌ BLOCKED
- **If Synthetic**: ⚠️ SYNTHETICALLY_VALIDATED (fake reference)

---

## T5: PINN Ablation & Sensitivity

| Component | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **PDE Loss** | ✅ YES | ✅ YES | ✅ YES | Control contribution of physics residuals |
| **Boundary Condition Loss** | ✅ YES | ✅ YES | ✅ YES | Vary BC enforcement |
| **Data Loss** | ⚠️ PARTIAL | ⚠️ PARTIAL | ✅ YES | Requires reference flow (not always available) |
| **Collocation Density** | ✅ YES | ✅ YES | ✅ YES | Vary number of collocation points |
| **Geometry Encoding** | ✅ YES | ✅ YES | ✅ YES | Compare with/without geometry context |
| **Normalization** | ✅ YES | ✅ YES | ✅ YES | Test different scaling schemes |
| **Convergence Analysis** | ✅ YES | ✅ YES | ✅ YES | Track loss curves, residuals |

**T5 Status**:
- **If IntrA + CFD**: ✅ READY (full ablation with reference)
- **If IntrA, no CFD**: ✅ READY (ablation without data loss term)
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T6: Morphology-Only Rupture Baseline

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Rupture Labels** | ❓ UNKNOWN | ❓ UNKNOWN | ✅ YES | CRITICAL BLOCKER |
| **Morphological Features** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Size, aspect ratio, shape descriptors from geometry |
| **Feature Extraction** | ✅ YES | ✅ YES | ✅ YES | Compute from point clouds or mesh properties |
| **Patient-Level Split** | ✅ YES | ⚠️ PARTIAL | ✅ YES | No patient leakage |
| **Classification Target** | ⚠️ UNCLEAR | ❓ UNKNOWN | ✅ YES | Define rupture status for each aneurysm |
| **Sample Size for Training** | ⚠️ DEPENDS | ❓ UNKNOWN | ✅ YES | Need sufficient labeled cases |

**T6 Status**:
- **If IntrA + rupture labels**: ✅ READY
- **If IntrA, no rupture labels**: ❌ BLOCKED
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T7: Flow-Only Rupture Model

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Rupture Labels** | ❓ UNKNOWN | ❓ UNKNOWN | ✅ YES | CRITICAL BLOCKER (same as T6) |
| **Hemodynamic Features** | ✅ YES (CFD data) | ❓ UNKNOWN | ✅ YES | WSS, OSI, RRT from reference flow |
| **Feature Derivation** | ✅ YES | ❓ UNKNOWN | ✅ YES | Can compute from CFD fields |
| **Temporal Features** | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ PARTIAL | If only steady-state CFD available |
| **Patient-Level Consistency** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Same splitting as T6 |

**T7 Status**:
- **If IntrA + rupture labels + CFD**: ✅ READY
- **If IntrA, missing rupture or CFD**: ❌ BLOCKED
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T8: Geometry + Flow Multimodal Rupture Model

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Geometry** | ✅ YES | ✅ YES | ✅ YES | Point cloud or mesh representation |
| **Hemodynamics** | ✅ YES (CFD) | ❓ UNKNOWN | ✅ YES | Multi-channel features (velocity, WSS, OSI, etc.) |
| **Rupture Labels** | ❓ UNKNOWN | ❓ UNKNOWN | ✅ YES | CRITICAL BLOCKER |
| **Multi-Channel Fusion** | ✅ YES | ✅ YES | ✅ YES | Architecture handles concatenation |
| **Patient-Level Consistency** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Same train/val/test as T6/T7 |
| **Baseline Comparisons** | ✅ YES | ⚠️ YES | ✅ YES | Against T6 (morphology) and T7 (flow) |

**T8 Status**:
- **If IntrA + rupture labels + CFD**: ✅ READY
- **If IntrA, missing rupture or CFD**: ❌ BLOCKED
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T9: Biomarker / Feature Ablation

| Component | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Morphology Features** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Remove size, shape descriptors |
| **Velocity Features** | ✅ YES (from CFD) | ❓ UNKNOWN | ✅ YES | Remove u, v, w components |
| **Pressure Features** | ✅ YES (from CFD) | ❓ UNKNOWN | ✅ YES | Remove pressure channel |
| **WSS Features** | ✅ YES (from CFD) | ❓ UNKNOWN | ✅ YES | Remove wall shear stress |
| **OSI Features** | ✅ YES (from CFD) | ❓ UNKNOWN | ✅ YES | Remove oscillatory shear index |
| **RRT Features** | ✅ YES (from CFD) | ❓ UNKNOWN | ✅ YES | Remove relative residence time |
| **Systematic Comparison** | ✅ YES | ⚠️ PARTIAL | ✅ YES | Remove one feature group at a time |

**T9 Status**:
- **If IntrA + rupture labels + CFD**: ✅ READY
- **If IntrA, missing rupture or CFD**: ❌ BLOCKED
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T10: Architecture / Model Comparison

| Baseline | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Logistic Regression** | ✅ YES | ✅ YES | ✅ YES | Simple baseline for detection |
| **Random Forest** | ✅ YES | ✅ YES | ✅ YES | Requires tabular features |
| **Gradient Boosting** | ✅ YES | ✅ YES | ✅ YES | Requires tabular features |
| **Simple MLP** | ✅ YES | ✅ YES | ✅ YES | Baseline NN on flattened point clouds |
| **PointNet (original)** | ✅ YES | ✅ YES | ✅ YES | Simpler point-cloud model |
| **Graph Neural Networks** | ✅ YES | ✅ YES | ✅ YES | Mesh-based alternatives |
| **Consistent Evaluation** | ✅ YES | ✅ YES | ✅ YES | Same train/val/test splits across models |

**T10 Status**:
- **If IntrA**: ✅ READY (detection task has clear labels)
- **If Synthetic**: ✅ SYNTHETICALLY_VALIDATED

---

## T11: External Validation

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Independent Dataset** | ❌ NO (IntrA is single source) | ⚠️ PARTIAL (if Aneumo used) | ❌ NO | **CRITICAL**: Cannot use same dataset for internal + external |
| **Frozen Model** | ✅ YES | ✅ YES | ✅ YES | T1 model locked before external validation |
| **No Retraining** | ✅ YES | ✅ YES | ✅ YES | Apply frozen weights, no fine-tuning |
| **Independent Evaluation** | ✅ FRAMEWORK | ✅ FRAMEWORK | ✅ FRAMEWORK | Metrics computed separately |
| **Confidence Intervals** | ✅ YES | ✅ YES | ✅ YES | Bootstrap estimates on held-out data |

**T11 Status**:
- **If IntrA only**: ❌ BLOCKED (cannot use same data for internal + external)
- **If IntrA + Aneumo available**: ✅ READY (use Aneumo as external validation set)
- **If Synthetic**: ❌ NOT_APPLICABLE (no independent external data)

---

## T12: Decision Curve / Clinical Utility Analysis

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Rupture Labels** | ❓ UNKNOWN | ❓ UNKNOWN | ✅ YES | Needed to define clinical decision problem |
| **Clinical Context** | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ ASSUMED | Define treat threshold, net benefit |
| **Patient-Level Outcomes** | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ ASSUMED | Adverse events, follow-up data |
| **DCA Framework** | ✅ YES | ✅ YES | ✅ YES | Algorithm exists; need clinical parameters |
| **ROC/PR Curves** | ✅ YES | ✅ YES | ✅ YES | Threshold analysis |
| **Risk Stratification** | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ ASSUMED | Define risk tiers |

**T12 Status**:
- **If IntrA + rupture labels + clinical outcome data**: ✅ READY
- **If IntrA + rupture labels, no outcomes**: ⚠️ PARTIAL (DCA possible but limited clinical relevance)
- **If IntrA, no rupture labels**: ❌ BLOCKED
- **If Synthetic**: ⚠️ SYNTHETICALLY_VALIDATED (proof-of-concept only)

---

## T13: Longitudinal Analysis

| Requirement | IntrA | Aneumo | Synthetic | Notes |
|---|---|---|---|---|
| **Multiple Timepoints** | ❓ UNKNOWN (likely NO) | ❓ UNKNOWN | ✅ YES (synthetic) | CRITICAL: Need repeated scans per patient |
| **Patient Follow-Up** | ❓ UNKNOWN | ❓ UNKNOWN | ❌ NO | Clinical follow-up data for outcome tracking |
| **Temporal Labels** | ❓ UNKNOWN | ❓ UNKNOWN | ⚠️ SYNTHETIC | Aneurysm growth, morphology change, rupture occurrence |
| **Patient Grouping** | ✅ FRAMEWORK | ✅ FRAMEWORK | ✅ FRAMEWORK | Ensure repeated measurements grouped by patient |
| **Time-Series Features** | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ ASSUMED | Rate of change, trajectory |
| **Censoring Handling** | ⚠️ FRAMEWORK | ⚠️ FRAMEWORK | ⚠️ FRAMEWORK | For incomplete follow-up |

**T13 Status**:
- **If IntrA + longitudinal data**: ✅ READY (rare but scientifically valuable)
- **If IntrA + cross-sectional only**: ❌ BLOCKED
- **If Synthetic**: ⚠️ SYNTHETICALLY_VALIDATED (proof-of-concept)

---

## Summary Table: Experiment Feasibility

| Experiment | Scientific Question | IntrA Feasibility | Aneumo Feasibility | Synthetic Feasibility | Priority |
|---|---|---|---|---|---|
| **T0** | Data audit & governance | ✅ READY | ⚠️ PARTIAL | ✅ READY | HIGH |
| **T1** | Detection baseline | ✅ READY | ⚠️ PARTIAL | ✅ READY | HIGH |
| **T2** | Robustness | ✅ READY | ⚠️ PARTIAL | ✅ READY | MEDIUM |
| **T3** | PINN baseline | ✅ READY | ⚠️ PARTIAL | ✅ READY | MEDIUM |
| **T4** | PINN validation | ⚠️ IF CFD | ❓ UNKNOWN | ❌ BLOCKED | MEDIUM |
| **T5** | PINN ablation | ✅ READY | ⚠️ PARTIAL | ✅ READY | MEDIUM |
| **T6** | Morphology rupture | ❌ IF NO LABELS | ❓ UNKNOWN | ✅ READY | HIGH |
| **T7** | Flow rupture | ❌ IF NO LABELS | ❓ UNKNOWN | ✅ READY | HIGH |
| **T8** | Multimodal rupture | ❌ IF NO LABELS | ❓ UNKNOWN | ✅ READY | HIGH |
| **T9** | Feature ablation | ❌ IF NO LABELS | ❓ UNKNOWN | ✅ READY | MEDIUM |
| **T10** | Model comparison | ✅ READY | ⚠️ PARTIAL | ✅ READY | MEDIUM |
| **T11** | External validation | ❌ NEEDS SECOND DATASET | ⚠️ IF Aneumo | ❌ NOT_APPLICABLE | MEDIUM |
| **T12** | Clinical utility | ⚠️ PARTIAL | ❓ UNKNOWN | ⚠️ PARTIAL | LOW |
| **T13** | Longitudinal | ❓ UNKNOWN | ❓ UNKNOWN | ⚠️ PARTIAL | LOW |

---

## Decision Criteria

### GREEN LIGHT (Proceed)
- Geometry available ✅
- Supervised labels clear ✅
- Sample size adequate ✅
- Patient-level IDs available ✅
- No leakage possible ✅

### YELLOW LIGHT (Proceed with Caveats)
- Partial data availability (e.g., some missing metadata)
- Sample size borderline
- Limited external validation options
- Document limitations clearly

### RED LIGHT (Blocked)
- Critical data missing (e.g., rupture labels for T6-T9)
- License incompatible
- Data integrity issues
- Patient-level leakage risk

---

## Action Plan

1. **VERIFY IntrA Availability**
   - [ ] Check if repository accessible
   - [ ] Confirm geometry files present
   - [ ] Confirm CFD data available
   - [ ] Look for rupture labels
   - [ ] Verify file structure and format

2. **If IntrA Available & Complete**:
   - Execute T0-T5 (detection + hemodynamics)
   - Execute T6-T10 only if rupture labels found
   - Execute T11 only if second dataset available
   - Execute T12-T13 with caveats

3. **If IntrA Incomplete**:
   - Evaluate Aneumo as alternative or supplement
   - Proceed with maximum feasible subset
   - Document all blockers

4. **If No Public Data Available**:
   - Mark experiments SYNTHETICALLY_VALIDATED
   - Proceed with synthetic-only pipeline
   - Clearly distinguish from clinical evidence

---

**Status**: AWAITING DATA VERIFICATION  
**Owner**: Research Lead  
**Last Updated**: August 13, 2026
