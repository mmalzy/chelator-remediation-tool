# Chelator Remediation Decision Support Tool

A machine learning-powered decision support system for heavy metal remediation in contaminated soils, calibrated for Rhode Island coastal environments.

## What It Does

Environmental practitioners input site-specific soil parameters (pH, metal concentrations, soil texture, moisture, salinity) and receive predictions for chelator effectiveness across four heavy metals (Pb, Cu, Zn, Cd) with five chelating agents (EDTA, NTA, Citrate, Humic Acid, Fulvic Acid).

## How It Works

The underlying Gradient Boosting models were trained on 12,636 geochemical simulations run in PHREEQC 3.5.0 using the minteq.v4.dat thermodynamic database. The target variable is **% free dissolved metal** — the bioavailable, mobile fraction posing the greatest environmental and health risk.

### Model Performance

| Metal | R² | CV R² | RMSE |
|-------|-----|-------|------|
| Lead (Pb) | 0.9990 | 0.9788 | 0.83% |
| Copper (Cu) | 0.9997 | 0.9481 | 0.59% |
| Zinc (Zn) | 0.9998 | 0.9972 | 0.33% |
| Cadmium (Cd) | 1.0000 | 0.9999 | 0.15% |

## Run Locally

```bash
pip install -r requirements.txt
streamlit run chelator_app.py
```

## Files

```
chelator_app.py        — Streamlit interface
requirements.txt       — Python dependencies
models/                — Trained ML models and encoders
  ├── pb_percent_free_model.joblib
  ├── cu_percent_free_model.joblib
  ├── zn_percent_free_model.joblib
  ├── cd_percent_free_model.joblib
  ├── label_encoders.joblib
  └── feature_info.json
```

## Author

Mallory Malz — University of Rhode Island

Geochemical modeling: PHREEQC 3.5.0 (USGS) with minteq.v4.dat
Machine learning: scikit-learn Gradient Boosting Regressor
