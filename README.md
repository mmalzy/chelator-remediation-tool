# Chelator ML Remediation Tool

**Geochemical Simulation-Trained Machine Learning for Chelator-Assisted Heavy Metal Remediation: A Decision-Support Framework**

Mallory Malz

---

## Overview

This repository contains the code, training data, and decision-support interface for a machine learning framework that predicts chelator effectiveness for heavy metal remediation in contaminated soils. The approach uses PHREEQC geochemical simulations as a systematic data generation engine for Gradient Boosting regression models, with parameters calibrated for Rhode Island coastal environments.

A methodology paper describing this framework is currently in preparation.

## Key Features

- **12,636 PHREEQC simulation scenarios** spanning four metals (Pb, Cu, Zn, Cd), five chelators (EDTA, NTA, citrate, humic acid, fulvic acid), and realistic Rhode Island soil conditions
- **Gradient Boosting models** with cross-validated R² of 0.979–1.000 for all four metals
- **Streamlit decision-support interface** for field practitioners
- **Internal validation** against eight chemical logic rules confirming geochemical consistency

## Repository Structure

```
chelator-remediation-tool/
├── README.md                       ← This file
├── LICENSE                         ← MIT License
├── chelator_app.py                 ← Streamlit interface (run from repo root)
│
├── data/
│   └── complete_training_data_with_baseline.csv   ← Master dataset (12,636 rows × 36 columns)
│
├── models/
│   ├── pb_percent_free_model.joblib
│   ├── cu_percent_free_model.joblib
│   ├── zn_percent_free_model.joblib
│   ├── cd_percent_free_model.joblib
│   ├── label_encoders.joblib
│   ├── feature_info.json
│   └── training_report.json
│
├── python_scripts/
│   ├── chelator_app_v5.py              ← Latest app version (same as chelator_app.py in root)
│   ├── train_chelator_model.py         ← Model training pipeline
│   ├── generate_final_RI_training_data.py  ← PHREEQC scenario generation
│   ├── generate_baseline_no_chelator.py    ← Baseline scenario generation
│   └── merge_baseline_into_training.py     ← Dataset assembly
│
├── phreeqc_inputs/
│   └── (example .phr input files)
│
└── figures/
    └── (prediction and feature importance plots for each metal)
```

## Quick Start

### Requirements

- Python 3.9+
- Required packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`

Install dependencies:
```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### Run the Decision-Support Interface

```bash
streamlit run chelator_app.py
```

The interface will open in your browser at `http://localhost:8501`.

### Training Data

The master dataset (`data/complete_training_data_with_baseline.csv`) contains 12,636 rows with 36 columns:

**Input features (14 used by model):** ph, pb_mg_L, cu_mg_L, zn_mg_L, cd_mg_L, doc_mg_L, ca_mg_L, mg_mg_L, na_mg_L, cl_mg_L, dose_mg_L, hfo_sites, pe, chelator_encoded

**Target variables:** pb_percent_free, cu_percent_free, zn_percent_free, cd_percent_free (percent free dissolved metal ion in soil pore water)

### Reproducing Results

To retrain models from the training data:
```bash
cd python_scripts
python3 train_chelator_model.py
```

## Model Performance

| Metal | R² (test) | CV R² (5-fold) | RMSE (pp) |
|-------|-----------|----------------|-----------|
| Pb    | 0.9990    | 0.9788         | 0.83      |
| Cu    | 0.9997    | 0.9481         | 0.59      |
| Zn    | 0.9998    | 0.9972         | 0.33      |
| Cd    | 1.0000    | 0.9999         | 0.15      |

## Geochemical Modeling

Simulations were run in PHREEQC 3.5.0 (USGS) with the minteq.v4.dat thermodynamic database, which includes stability constants for EDTA, NTA, and citrate complexation with all target metals, as well as surface complexation parameters for iron hydroxide (Hfo) surfaces.

**Parameter space:** 5 pH levels (5.5–7.5) × 3 metal contamination levels × 5 chelators × 3 doses × 3 soil textures × 3 moisture/redox conditions × 3 ionic strength levels × 2 Ca/Mg competition levels, plus 486 no-chelator baselines.

## Key Scientific Findings

- **pH is the dominant control** on chelator effectiveness for all four metals
- **Optimal chelator varies by metal:** Citrate for Pb, NTA for Cu/Zn/Cd — EDTA is not always best
- **Zinc is resistant to chelation** (mean 84% free across all scenarios)
- **High ionic strength (coastal salinity) reduces free Cd** by 57 percentage points through chloride complexation
- **Underdosing with weak chelators can increase free Pb** relative to no treatment (competitive desorption)

## Citation

If you use this code or data, please cite this repository:

```
Malz, M. (2026). Chelator ML Remediation Tool: Geochemical Simulation-Trained
Machine Learning for Heavy Metal Remediation. GitHub repository:
https://github.com/mmalzy/chelator-remediation-tool
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
