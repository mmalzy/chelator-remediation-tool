# Chelator ML Remediation Tool

**Geochemical Simulation-Trained Machine Learning for Chelator-Assisted Heavy Metal Remediation: A Decision-Support Framework**

Mallory Malz — University of Rhode Island

---

## Overview

This repository contains the code, training data, and decision-support interface for a machine learning framework that predicts chelator effectiveness for heavy metal remediation in contaminated soils. The approach uses PHREEQC geochemical simulations as a systematic data generation engine for Gradient Boosting regression models, with parameters calibrated for Rhode Island coastal environments.

**Associated Publication:**  
Malz, M. (2026). Geochemical Simulation-Trained Machine Learning for Chelator-Assisted Heavy Metal Remediation: A Decision-Support Framework. *Journal of Hazardous Materials* (submitted).

## Key Features

- **12,636 PHREEQC simulation scenarios** spanning four metals (Pb, Cu, Zn, Cd), five chelators (EDTA, NTA, citrate, humic acid, fulvic acid), and realistic Rhode Island soil conditions
- **Gradient Boosting models** with cross-validated R² of 0.979–1.000 for all four metals
- **Streamlit decision-support interface** for field practitioners
- **Three-tier validation framework**: internal chemical logic, literature benchmarking, and planned bench-scale experiments

## Repository Structure

```
chelator_ml_project/
├── README.md                          ← This file
├── LICENSE                            ← MIT License
│
├── data/
│   ├── complete_training_data_with_baseline.csv   ← Master dataset (12,636 rows × 36 columns)
│   └── literature_benchmark_data.csv              ← Tier 2 validation data (when available)
│
├── models/
│   ├── pb_percent_free_model.joblib   ← Trained Gradient Boosting model for Pb
│   ├── cu_percent_free_model.joblib   ← Trained Gradient Boosting model for Cu
│   ├── zn_percent_free_model.joblib   ← Trained Gradient Boosting model for Zn
│   ├── cd_percent_free_model.joblib   ← Trained Gradient Boosting model for Cd
│   ├── label_encoders.joblib          ← Scikit-learn label encoders for categorical features
│   ├── feature_info.json              ← Feature column names and order
│   └── training_report.json           ← Full training metrics
│
├── python_scripts/
│   ├── chelator_app_v5.py             ← Streamlit decision-support interface
│   ├── train_chelator_model.py        ← Model training pipeline
│   ├── tier1_validation_chemical_logic.py  ← Internal consistency validation
│   ├── generate_final_RI_training_data.py  ← PHREEQC scenario generation
│   ├── generate_baseline_no_chelator.py    ← Baseline scenario generation
│   └── merge_baseline_into_training.py     ← Dataset assembly
│
├── phreeqc_inputs/
│   └── (example .phr input files)     ← Representative PHREEQC input files
│
└── figures/
    ├── pb_percent_free_predictions.png
    ├── cu_percent_free_predictions.png
    ├── zn_percent_free_predictions.png
    ├── cd_percent_free_predictions.png
    ├── pb_percent_free_feature_importance.png
    ├── cu_percent_free_feature_importance.png
    ├── zn_percent_free_feature_importance.png
    └── cd_percent_free_feature_importance.png
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
cd python_scripts
python3 -m streamlit run chelator_app_v5.py
```

The interface will open in your browser at `http://localhost:8501`.

### Training Data Format

The master dataset (`data/complete_training_data_with_baseline.csv`) contains 12,636 rows with 36 columns:

**Input features (14 used by model):** ph, pb_mg_L, cu_mg_L, zn_mg_L, cd_mg_L, doc_mg_L, ca_mg_L, mg_mg_L, na_mg_L, cl_mg_L, dose_mg_L, hfo_sites, pe, chelator_encoded

**Target variables:** pb_percent_free, cu_percent_free, zn_percent_free, cd_percent_free (percent free dissolved metal ion in soil pore water)

### Reproducing Results

To retrain models from the training data:
```bash
cd python_scripts
python3 train_chelator_model.py
```

This will produce new model files in `models/` and prediction/feature importance plots in `figures/`.

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

## Citation

If you use this code or data, please cite:

```
Malz, M. (2026). Geochemical Simulation-Trained Machine Learning for Chelator-Assisted
Heavy Metal Remediation: A Decision-Support Framework. Journal of Hazardous Materials.
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact

Mallory Malz — University of Rhode Island
