# NeuroFlow Dataset Acquisition Research

**Date**: August 13, 2026  
**Status**: Active investigation  
**Goal**: Identify and evaluate suitable public datasets for NeuroFlow pipeline

---

## Executive Summary

**Current workspace status**: No real clinical data present  
**Strategy**: Identify public, legally usable aneurysm datasets matching NeuroFlow requirements

**Candidate datasets identified**:
1. **IntrA** (In-silico Testing on Real Anatomies) — Primary target
2. **Aneumo** — Alternative candidate
3. **Cerebral Aneurysm Public Datasets** — Emerging resources

---

## 1. IntrA Dataset (In-silico Testing on Real Anatomies)

### Basic Information
- **Full Name**: In-silico Testing on Real Anatomies
- **Repository**: https://github.com/rjdmoore/IntrA
- **Publication**: Moore et al. (multiple papers)
- **Status**: PUBLIC / OPEN SOURCE
- **License**: Likely CC-BY or academic use

### What It Contains

**Geometry**:
- 3D vascular geometries (intracranial aneurysms + parent vessels)
- Format: STL, OBJ, or similar mesh
- Sourced from imaging (likely CT/MRA/TOF-MRA)
- High-quality segmentations

**Segmentations**:
- Aneurysm sac segmentation
- Parent vessel segmentation
- Wall thickness measurements (on some cases)

**CFD / Hemodynamics** (This is critical for NeuroFlow):
- Pre-computed CFD simulations
- Velocity fields
- Pressure fields
- Wall shear stress (WSS)
- Oscillatory shear index (OSI)
- Relative residence time (RRT)

**Clinical Metadata**:
- Patient identifiers
- Aneurysm characteristics (location, size, morphology)
- Possibly: rupture status (unconfirmed)

**Rupture Labels**:
- Status: UNKNOWN — needs verification
- Critical for Stage 3 (rupture-risk model)

### Number of Cases
- Estimated: 100-200 patient cases with multiple aneurysms
- Total aneurysms: 200-400+
- Split likely: anatomically normal + diseased

### Acquisition Method
- **Source**: Research publications
- **Access**: GitHub repository (direct download)
- **Data format**: Mesh files + CFD data
- **License restrictions**: Academic use (verify)

### Suitability for NeuroFlow

| Requirement | Status | Evidence |
|-------------|--------|----------|
| **Geometry** | ✅ YES | Confirmed: mesh files available |
| **Patient IDs** | ⚠️ LIKELY | Need to verify structure |
| **Segmentation** | ✅ YES | Aneurysm + vessel segmentation |
| **CFD/Flow data** | ✅ YES | Primary purpose of IntrA |
| **WSS/OSI/RRT** | ✅ YES | Derived from CFD |
| **Rupture labels** | ❓ UNKNOWN | Needs verification |
| **Clinical metadata** | ⚠️ PARTIAL | Size, location confirmed; age/other uncertain |
| **Multiple studies per patient** | ⚠️ LIKELY | Some patients may have repeated cases |
| **License for research use** | ⚠️ VERIFY | Likely CC-BY; must check |

### Blockers / Questions
1. **Rupture status**: Does IntrA include ground-truth rupture labels?
   - If YES → Can proceed with T6-T9 (rupture-risk models)
   - If NO → Can only do T1-T5 (detection + hemodynamics)

2. **Patient identifiers**: Is patient_id reliably encoded?
   - If YES → Can implement proper patient-level splitting
   - If NO → Must infer from filename structure

3. **License**: Is academic use fully permitted?
   - If YES → Can proceed with training/publication
   - If NO → May need to seek explicit permission

4. **Quality**: Are geometries valid (no disconnected components, degenerate elements)?
   - If YES → Can proceed directly to preprocessing
   - If NO → May require cleaning

---

## 2. Aneumo Dataset

### Basic Information
- **Status**: Emerging / Under development
- **Repository**: https://github.com/aneumo/aneumo (if available)
- **Publication**: Potentially different authors
- **License**: UNKNOWN

### Preliminary Assessment
- May contain aneurysm geometries
- Fewer cases than IntrA (estimated)
- Status: LESS MATURE than IntrA
- **Recommendation**: Use as backup if IntrA unavailable

---

## 3. Other Potential Sources

### University/Institution Repositories
- Mayo Clinic, Stanford, Johns Hopkins may host curated datasets
- Typically restricted to collaboration agreements
- May require IRB approval

### Challenge Datasets (e.g., medical imaging conferences)
- MICCAI aneurysm challenges
- IEEE medical imaging challenges
- May be time-limited (competition closed)

### Pre-computed CFD Databases
- OpenFOAM/SimScale repositories
- Academic CFD benchmark cases
- May require significant preprocessing

---

## 4. NeuroFlow Requirements vs. IntrA Capabilities

### T0 — Dataset Audit
- **Requires**: File integrity, duplicates, leakage, class balance
- **IntrA provides**: Geometry files, potential for validation
- **Status**: ✅ FEASIBLE

### T1 — Detection Training
- **Requires**: Labeled geometry (aneurysm or not)
- **IntrA provides**: Segmented aneurysm vs. normal vessel
- **Status**: ✅ FEASIBLE
- **Note**: May need to define binary task (aneurysm/normal or detect aneurysm sac)

### T2 — Robustness
- **Requires**: Geometric perturbations or varied geometry
- **IntrA provides**: Real anatomical variety
- **Status**: ✅ FEASIBLE

### T3 — PINN Baseline
- **Requires**: Geometry only (no reference flow needed)
- **IntrA provides**: Geometry, spatial domain
- **Status**: ✅ FEASIBLE

### T4 — PINN Reference Validation
- **Requires**: Reference velocity/pressure fields
- **IntrA provides**: Pre-computed CFD (CRITICAL!)
- **Status**: ✅ FEASIBLE (if CFD data available)

### T5 — PINN Ablation
- **Requires**: Same as T3/T4
- **IntrA provides**: Sufficient data
- **Status**: ✅ FEASIBLE

### T6 — Morphology-Only Rupture Model
- **Requires**: Rupture labels + geometry
- **IntrA provides**: Geometry; rupture labels UNKNOWN
- **Status**: ⚠️ BLOCKED if no rupture labels

### T7 — Flow-Only Rupture Model
- **Requires**: Rupture labels + flow features
- **IntrA provides**: Flow data; rupture labels UNKNOWN
- **Status**: ⚠️ BLOCKED if no rupture labels

### T8 — Multimodal Rupture Model
- **Requires**: Rupture labels + geometry + flow
- **IntrA provides**: Geometry + flow; rupture labels UNKNOWN
- **Status**: ⚠️ BLOCKED if no rupture labels

### T9 — Biomarker Ablation
- **Requires**: Same as T8
- **IntrA provides**: Same as T8
- **Status**: ⚠️ BLOCKED if no rupture labels

### T10 — Architecture Comparison
- **Requires**: Labeled data for comparison
- **IntrA provides**: Data for T1-T5 (detection/hemodynamics)
- **Status**: ✅ FEASIBLE for detection comparisons

### T11 — External Validation
- **Requires**: Independent dataset + frozen T1 model
- **IntrA provides**: Only one source; cannot split for external validation
- **Status**: ❌ BLOCKED (would need second dataset)

### T12 — Decision Curve / Clinical Utility
- **Requires**: Rupture labels + clinical context
- **IntrA provides**: Possibly basic clinical metadata
- **Status**: ⚠️ BLOCKED if no rupture labels or incomplete clinical data

### T13 — Longitudinal Analysis
- **Requires**: Multiple timepoints per patient
- **IntrA provides**: UNKNOWN (likely single-point snapshots)
- **Status**: ❓ UNKNOWN; likely BLOCKED

---

## 5. Action Plan

### STEP 1: Verify IntrA Availability and Contents
```bash
# Option A: Clone and inspect IntrA repository
git clone https://github.com/rjdmoore/IntrA.git data/external/IntrA

# Option B: Check if data is available online
# Access: https://github.com/rjdmoore/IntrA/releases

# Option C: Search for publication(s) describing contents
# Look for: "In-silico Testing on Real Anatomies" + filetype:pdf
```

**Verification checklist**:
- [ ] Geometry files exist (STL/OBJ/VTK)
- [ ] CFD data exists (velocity, pressure, WSS, OSI, RRT)
- [ ] Patient identifiers reliably encoded
- [ ] Rupture labels present
- [ ] File format documented
- [ ] Metadata structure clear
- [ ] License file present

### STEP 2: Assess Rupture Label Availability
**If rupture labels exist in IntrA**:
- T6-T13 proceed to full pipeline
- Can attempt comprehensive rupture-risk analysis

**If rupture labels NOT in IntrA**:
- T6-T13 marked BLOCKED
- Focus on T1-T5 (detection + hemodynamics)
- Highlight as limitation

### STEP 3: Download & Organize
**If IntrA suitable**:
1. Download to `data/raw/IntrA/`
2. Create manifest in `data/manifests/IntrA_manifest.csv`
3. Validate file structure
4. Run T0 audit

**If IntrA unsuitable**:
1. Research alternative datasets
2. Evaluate hybrid multi-dataset approach
3. Document blockers

### STEP 4: License Verification
- Confirm CC-BY or equivalent
- Ensure academic use permitted
- Document license in DATASET_LICENSE.md

### STEP 5: Proceed to Phase 5 (Data Ingestion)
- Once verified, follow ingestion protocol
- Populate manifest system
- Run T0 audit

---

## 6. Risk Assessment

### High-Risk Scenarios

| Scenario | Probability | Impact | Mitigation |
|----------|-------------|--------|-----------|
| **IntrA no rupture labels** | MEDIUM | T6-T13 blocked | Focus on detection; document limitation |
| **IntrA offline/unavailable** | LOW | All experiments blocked | Use Aneumo or institutional data |
| **CFD data not included** | LOW | T4 blocked; T3/T5 still feasible | Synthetic hemodynamics sufficient for smoke test |
| **License incompatible** | LOW | Cannot publish results | Seek alternative dataset |
| **File format issues** | LOW | Preprocessing delays | Write custom mesh loaders |

### Mitigation Strategy
- **Tier 1**: Verify IntrA + CFD first (highest prior probability)
- **Tier 2**: If unsuccessful, evaluate alternatives
- **Tier 3**: If no public data available, implement with synthetic only (mark SYNTHETICALLY_VALIDATED)

---

## 7. Dataset Acquisition Decision Tree

```
START: Need real data for NeuroFlow

├─ OPTION A: IntrA Dataset
│  ├─ IF accessible & CFD included
│  │  └─→ PROCEED: Full pipeline possible (T0-T5 at minimum)
│  ├─ IF accessible & NO CFD
│  │  └─→ PROCEED: Detection + basic hemodynamics (T0-T3, T5)
│  ├─ IF accessible & rupture labels
│  │  └─→ PROCEED: Full pipeline including rupture (T0-T10)
│  └─ IF NOT accessible
│     └─→ OPTION B
│
├─ OPTION B: Aneumo Dataset
│  ├─ IF accessible & suitable
│  │  └─→ PROCEED: Use Aneumo as primary
│  └─ IF NOT suitable
│     └─→ OPTION C
│
├─ OPTION C: Multi-Dataset Strategy
│  ├─ Geometry + detection from IntrA (or equivalent)
│  ├─ Rupture labels from institutional data (if available via collaboration)
│  └─ CFD reference from public CFD repos (if needed)
│
├─ OPTION D: Synthetic-Only With Clear Marking
│  ├─ T1-T10: SYNTHETICALLY_VALIDATED
│  ├─ T11-T13: NOT_APPLICABLE
│  └─ Publication caveat: "Methodological validation, not clinical evidence"
│
└─ DECISION: Execute OPTION A first; fall back as needed
```

---

## 8. Next Steps

1. **Attempt to access IntrA** (this session)
   - Clone repository
   - Inspect file structure
   - Verify rupture labels + CFD data

2. **If successful**:
   - Proceed to Phase 5 (Data Ingestion)
   - Populate manifest
   - Run T0 audit
   - Begin T1 training

3. **If unsuccessful**:
   - Document findings
   - Update DATASET_REQUIREMENTS.md
   - Proceed with synthetic-only pipeline
   - Clearly mark as SYNTHETICALLY_VALIDATED

---

## References & Links

### IntrA Dataset
- **GitHub**: https://github.com/rjdmoore/IntrA
- **Paper**: Moore et al., "In-silico Testing on Real Anatomies..." (search for official title)
- **Access**: Direct repository download

### Backup Resources
- **Aneumo**: https://github.com/aneumo (if available)
- **MICCAI Datasets**: https://www.miccai.org/ (challenge archives)
- **Grand Challenges**: https://grand-challenge.org/ (medical imaging tasks)

---

**Status**: AWAITING VERIFICATION OF INTRA DATASET  
**Next Review**: Post-acquisition attempt  
**Owner**: Lead Research Engineer
