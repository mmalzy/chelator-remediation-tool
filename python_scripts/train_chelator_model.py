#!/usr/bin/env python3
"""
train_chelator_model.py
========================
Trains machine learning models to predict % free metal fraction
based on soil conditions and chelator treatment.

Trains both Random Forest and Gradient Boosting, compares them,
and saves the best performer for each metal.

Input:  complete_training_data_with_baseline.csv (12,636 rows)
Output: Trained models in models/ folder + evaluation report

Usage:
    python3 train_chelator_model.py

Author: Mallory Malz (with AI co-creator)
Project: Chelator ML Remediation - Rhode Island Coastal Soils
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
import time
from datetime import datetime

# Check for required packages and give helpful install messages
required_packages = {
    'sklearn': 'scikit-learn',
    'joblib': 'joblib',
    'matplotlib': 'matplotlib',
}

missing = []
for import_name, install_name in required_packages.items():
    try:
        __import__(import_name)
    except ImportError:
        missing.append(install_name)

if missing:
    print("=" * 60)
    print("MISSING PACKAGES - Need to install before training")
    print("=" * 60)
    print(f"\nRun this command to install them:\n")
    print(f"  pip3 install --user {' '.join(missing)}")
    print(f"\nThen re-run this script:")
    print(f"  python3 train_chelator_model.py")
    sys.exit(1)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Input file
DATA_FILE = os.path.join(DATA_DIR, "complete_training_data_with_baseline.csv")

# Target variables (what we're predicting)
TARGETS = [
    "pb_percent_free",
    "cu_percent_free",
    "zn_percent_free",
    "cd_percent_free",
]

# Features to USE for training (numeric + chelator)
# Dropping redundant categorical labels, keeping numeric equivalents
NUMERIC_FEATURES = [
    "ph",
    "pb_mg_L",
    "cu_mg_L",
    "zn_mg_L",
    "cd_mg_L",
    "doc_mg_L",
    "ca_mg_L",
    "mg_mg_L",
    "na_mg_L",
    "cl_mg_L",
    "dose_mg_L",
    "hfo_sites",
    "pe",
]

CATEGORICAL_FEATURES = [
    "chelator",    # EDTA, NTA, Citrate, Humic, Fulvic, None
]

# Random seed for reproducibility
RANDOM_STATE = 42


def load_and_prepare_data():
    """Load CSV and prepare features for ML training."""
    print("--- Loading Data ---")
    
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print(f"Make sure you ran merge_baseline_into_training.py first!")
        sys.exit(1)
    
    df = pd.read_csv(DATA_FILE)
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Check that targets exist and have no nulls
    for t in TARGETS:
        if t not in df.columns:
            print(f"ERROR: Target column '{t}' not found!")
            sys.exit(1)
        nulls = df[t].isna().sum()
        if nulls > 0:
            print(f"  Warning: {nulls} null values in {t}, dropping those rows")
            df = df.dropna(subset=[t])
    
    print(f"  After cleaning: {df.shape[0]} rows")
    
    # --- Encode categorical features ---
    print("\n--- Encoding Categorical Features ---")
    label_encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df[col + "_encoded"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Build feature list (numeric + encoded categoricals)
    feature_columns = NUMERIC_FEATURES + [col + "_encoded" for col in CATEGORICAL_FEATURES]
    
    print(f"\n  Feature columns ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns):
        print(f"    {i+1}. {col}")
    
    X = df[feature_columns].copy()
    y_dict = {t: df[t].copy() for t in TARGETS}
    
    return X, y_dict, label_encoders, df


def train_and_evaluate(X, y, target_name):
    """
    Train both Random Forest and Gradient Boosting for one target variable.
    Returns the best model and its metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Training models for: {target_name}")
    print(f"{'='*60}")
    
    # Split data: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"  Train set: {X_train.shape[0]} rows")
    print(f"  Test set:  {X_test.shape[0]} rows")
    print(f"  Target range: {y.min():.1f}% to {y.max():.1f}%")
    
    results = {}
    
    # --- Random Forest ---
    print(f"\n  Training Random Forest...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=200,       # 200 trees (good balance of accuracy/speed)
        max_depth=20,           # Prevent overly deep trees
        min_samples_split=5,    # Require 5 samples to split a node
        min_samples_leaf=2,     # Each leaf must have at least 2 samples
        max_features='sqrt',    # Use sqrt(n_features) at each split
        random_state=RANDOM_STATE,
        n_jobs=-1,              # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0
    
    rf_pred = rf.predict(X_test)
    rf_metrics = {
        "r2": r2_score(y_test, rf_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "mae": mean_absolute_error(y_test, rf_pred),
        "train_time": rf_time,
    }
    print(f"    R² = {rf_metrics['r2']:.4f}  |  RMSE = {rf_metrics['rmse']:.2f}%  "
          f"|  MAE = {rf_metrics['mae']:.2f}%  |  Time: {rf_time:.1f}s")
    
    # Cross-validation for Random Forest (5-fold)
    print(f"  Running 5-fold cross-validation for RF...")
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring='r2', n_jobs=-1)
    rf_metrics["cv_r2_mean"] = rf_cv.mean()
    rf_metrics["cv_r2_std"] = rf_cv.std()
    print(f"    CV R² = {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
    
    results["RandomForest"] = {"model": rf, "metrics": rf_metrics, "predictions": rf_pred}
    
    # --- Gradient Boosting ---
    print(f"\n  Training Gradient Boosting...")
    t0 = time.time()
    gb = GradientBoostingRegressor(
        n_estimators=200,       # 200 boosting stages
        max_depth=6,            # Shallower trees (boosting compensates)
        learning_rate=0.1,      # Standard learning rate
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,          # Use 80% of data per tree (reduces overfitting)
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train, y_train)
    gb_time = time.time() - t0
    
    gb_pred = gb.predict(X_test)
    gb_metrics = {
        "r2": r2_score(y_test, gb_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, gb_pred)),
        "mae": mean_absolute_error(y_test, gb_pred),
        "train_time": gb_time,
    }
    print(f"    R² = {gb_metrics['r2']:.4f}  |  RMSE = {gb_metrics['rmse']:.2f}%  "
          f"|  MAE = {gb_metrics['mae']:.2f}%  |  Time: {gb_time:.1f}s")
    
    # Cross-validation for Gradient Boosting
    print(f"  Running 5-fold cross-validation for GB...")
    gb_cv = cross_val_score(gb, X, y, cv=5, scoring='r2', n_jobs=-1)
    gb_metrics["cv_r2_mean"] = gb_cv.mean()
    gb_metrics["cv_r2_std"] = gb_cv.std()
    print(f"    CV R² = {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")
    
    results["GradientBoosting"] = {"model": gb, "metrics": gb_metrics, "predictions": gb_pred}
    
    # --- Pick winner based on CV R² ---
    if rf_metrics["cv_r2_mean"] >= gb_metrics["cv_r2_mean"]:
        winner = "RandomForest"
    else:
        winner = "GradientBoosting"
    
    print(f"\n  >>> WINNER for {target_name}: {winner} "
          f"(CV R² = {results[winner]['metrics']['cv_r2_mean']:.4f})")
    
    return results, winner, X_test, y_test


def get_feature_importance(model, feature_names, target_name, model_name):
    """Extract and display feature importance."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\n  Feature Importance ({model_name} → {target_name}):")
    print(f"  {'Rank':<5} {'Feature':<25} {'Importance':<12} {'Bar'}")
    print(f"  {'-'*5} {'-'*25} {'-'*12} {'-'*20}")
    
    importance_data = []
    for rank, idx in enumerate(indices):
        bar = "█" * int(importances[idx] * 50)
        print(f"  {rank+1:<5} {feature_names[idx]:<25} {importances[idx]:<12.4f} {bar}")
        importance_data.append({
            "rank": rank + 1,
            "feature": feature_names[idx],
            "importance": float(importances[idx]),
        })
    
    return importance_data


def plot_results(y_test, predictions, target_name, model_name, save_path):
    """Create actual vs predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: actual vs predicted
    ax1 = axes[0]
    ax1.scatter(y_test, predictions, alpha=0.3, s=10, color='steelblue')
    ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual % Free', fontsize=12)
    ax1.set_ylabel('Predicted % Free', fontsize=12)
    ax1.set_title(f'{target_name}\n{model_name} - Actual vs Predicted', fontsize=13)
    ax1.legend()
    ax1.set_xlim(-5, 105)
    ax1.set_ylim(-5, 105)
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[1]
    residuals = predictions - y_test.values
    ax2.scatter(predictions, residuals, alpha=0.3, s=10, color='coral')
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xlabel('Predicted % Free', fontsize=12)
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax2.set_title(f'{target_name}\nResiduals', fontsize=13)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")


def plot_feature_importance(importance_data, target_name, model_name, save_path):
    """Create horizontal bar chart of feature importance."""
    features = [d['feature'] for d in importance_data]
    importances = [d['importance'] for d in importance_data]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = range(len(features) - 1, -1, -1)  # Reverse so top feature is at top
    
    colors = ['#2196F3' if imp > 0.1 else '#90CAF9' if imp > 0.05 
              else '#E0E0E0' for imp in importances]
    
    ax.barh(y_pos, importances, color=colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance: {target_name}\n({model_name})', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {save_path}")


def main():
    """Main training pipeline."""
    start_time = time.time()
    
    print("=" * 60)
    print("CHELATOR ML MODEL TRAINING")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    X, y_dict, label_encoders, df = load_and_prepare_data()
    feature_names = list(X.columns)
    
    # Store all results for the summary report
    all_results = {}
    
    # Train models for each target metal
    for target in TARGETS:
        metal = target.split("_")[0].upper()  # e.g., "PB", "CU"
        
        results, winner, X_test, y_test = train_and_evaluate(X, y_dict[target], target)
        
        # Get the winning model
        best_model = results[winner]["model"]
        best_metrics = results[winner]["metrics"]
        best_predictions = results[winner]["predictions"]
        
        # Feature importance
        importance_data = get_feature_importance(
            best_model, feature_names, target, winner
        )
        
        # Save the winning model
        model_filename = f"{target}_model.joblib"
        model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(best_model, model_path)
        print(f"\n  Model saved: {model_path}")
        
        # Save plots
        plot_results(
            y_test, best_predictions, target, winner,
            os.path.join(MODEL_DIR, f"{target}_predictions.png")
        )
        plot_feature_importance(
            importance_data, target, winner,
            os.path.join(MODEL_DIR, f"{target}_feature_importance.png")
        )
        
        # Store results
        all_results[target] = {
            "winner": winner,
            "metrics": {
                "RandomForest": results["RandomForest"]["metrics"],
                "GradientBoosting": results["GradientBoosting"]["metrics"],
            },
            "feature_importance": importance_data,
        }
    
    # --- Save label encoders (needed by the interface) ---
    encoders_path = os.path.join(MODEL_DIR, "label_encoders.joblib")
    joblib.dump(label_encoders, encoders_path)
    print(f"\n  Label encoders saved: {encoders_path}")
    
    # --- Save feature list (needed by the interface) ---
    feature_info = {
        "feature_columns": feature_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "targets": TARGETS,
    }
    feature_path = os.path.join(MODEL_DIR, "feature_info.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"  Feature info saved: {feature_path}")
    
    # ============================================================
    # FINAL SUMMARY REPORT
    # ============================================================
    elapsed = time.time() - start_time
    
    print(f"\n\n{'='*60}")
    print(f"TRAINING COMPLETE - SUMMARY REPORT")
    print(f"{'='*60}")
    print(f"Total training time: {elapsed:.0f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Training data: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"\n{'Metal':<8} {'Best Model':<20} {'R²':<8} {'CV R²':<12} {'RMSE':<8} {'MAE':<8}")
    print(f"{'-'*8} {'-'*20} {'-'*8} {'-'*12} {'-'*8} {'-'*8}")
    
    for target in TARGETS:
        metal = target.replace("_percent_free", "").upper()
        r = all_results[target]
        winner = r["winner"]
        m = r["metrics"][winner]
        cv_str = f"{m['cv_r2_mean']:.4f}±{m['cv_r2_std']:.4f}"
        print(f"{metal:<8} {winner:<20} {m['r2']:<8.4f} {cv_str:<12} "
              f"{m['rmse']:<8.2f} {m['mae']:<8.2f}")
    
    print(f"\n  Files saved to: {MODEL_DIR}/")
    print(f"    - 4 model files (*_model.joblib)")
    print(f"    - 4 prediction plots (*_predictions.png)")
    print(f"    - 4 feature importance plots (*_feature_importance.png)")
    print(f"    - label_encoders.joblib")
    print(f"    - feature_info.json")
    
    # Interpretation guide
    print(f"\n{'='*60}")
    print(f"HOW TO INTERPRET THESE RESULTS")
    print(f"{'='*60}")
    print(f"  R² (R-squared): How well the model explains the data")
    print(f"    > 0.95 = Excellent  |  > 0.90 = Very Good  |  > 0.80 = Good")
    print(f"  RMSE: Average prediction error in percentage points")
    print(f"    < 5% = Excellent  |  < 10% = Good  |  > 15% = Needs work")
    print(f"  CV R²: Cross-validated R² (more trustworthy than plain R²)")
    print(f"    Close to plain R² = model is NOT overfitting (good!)")
    print(f"    Much lower than R² = model IS overfitting (bad)")
    print(f"\n  Top features to look for:")
    print(f"    - pH should be #1 or #2 for all metals")
    print(f"    - chelator and dose should be important")
    print(f"    - hfo_sites (soil surface) should matter for sorption")
    
    # Save full report as JSON for reference
    report = {
        "training_date": datetime.now().isoformat(),
        "data_file": DATA_FILE,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_columns": feature_names,
        "results": {},
    }
    for target in TARGETS:
        r = all_results[target]
        report["results"][target] = {
            "winner": r["winner"],
            "metrics": {
                model_name: {k: float(v) for k, v in metrics.items()}
                for model_name, metrics in r["metrics"].items()
            },
            "top_3_features": [
                d["feature"] for d in r["feature_importance"][:3]
            ],
        }
    
    report_path = os.path.join(MODEL_DIR, "training_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  Full report saved: {report_path}")
    
    print(f"\n{'='*60}")
    print(f"NEXT STEP: Build the Streamlit interface!")
    print(f"  The models are ready to power the recommendation tool.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
