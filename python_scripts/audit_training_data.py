#!/usr/bin/env python3
"""
audit_training_data.py
=======================
READ-ONLY audit of the complete training dataset.
Does NOT modify any files. Reports potential issues for review.

Run this BEFORE ML training to catch data quality issues.

Usage:
    python3 audit_training_data.py
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA_DIR = os.path.join(BASE_DIR, "data")


def audit_file(filepath):
    """Run comprehensive audit on a training data CSV."""
    
    print(f"\n{'=' * 70}")
    print(f"DATA AUDIT REPORT")
    print(f"File: {filepath}")
    print(f"{'=' * 70}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found!")
        return
    
    df = pd.read_csv(filepath)
    
    # ---- 1. BASIC SHAPE ----
    print(f"\n--- 1. BASIC INFO ---")
    print(f"  Rows: {df.shape[0]}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # ---- 2. COLUMN NAMES (check for duplicates) ----
    print(f"\n--- 2. COLUMN NAMES ---")
    col_counts = pd.Series(df.columns).value_counts()
    dupes = col_counts[col_counts > 1]
    if len(dupes) > 0:
        print(f"  ⚠️  DUPLICATE COLUMN NAMES FOUND:")
        for col, count in dupes.items():
            print(f"      '{col}' appears {count} times")
        print(f"  (pandas may have auto-renamed to '{col}.1', etc.)")
    else:
        print(f"  ✓ No duplicate column names")
    
    # List all columns
    print(f"\n  All columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        nulls = df[col].isna().sum()
        null_str = f" [⚠️ {nulls} nulls]" if nulls > 0 else ""
        print(f"    {i+1:2d}. {col:25s} ({dtype}){null_str}")
    
    # ---- 3. MISSING VALUES ----
    print(f"\n--- 3. MISSING VALUES ---")
    total_nulls = df.isna().sum().sum()
    if total_nulls == 0:
        print(f"  ✓ No missing values")
    else:
        print(f"  ⚠️  Total missing values: {total_nulls}")
        null_cols = df.isna().sum()
        for col, count in null_cols[null_cols > 0].items():
            print(f"      {col}: {count} missing ({100*count/len(df):.1f}%)")
    
    # ---- 4. TARGET VARIABLE DISTRIBUTIONS ----
    print(f"\n--- 4. TARGET VARIABLES ---")
    targets = ['pb_percent_free', 'cu_percent_free', 'zn_percent_free', 'cd_percent_free']
    for t in targets:
        if t in df.columns:
            col = df[t]
            print(f"\n  {t}:")
            print(f"    Mean:   {col.mean():.2f}%")
            print(f"    Median: {col.median():.2f}%")
            print(f"    Std:    {col.std():.2f}%")
            print(f"    Min:    {col.min():.2f}%")
            print(f"    Max:    {col.max():.2f}%")
            # Check for concerning patterns
            pct_zero = (col == 0).sum() / len(col) * 100
            pct_hundred = (col >= 99.9).sum() / len(col) * 100
            if pct_zero > 10:
                print(f"    ⚠️  {pct_zero:.1f}% of values are exactly 0")
            if pct_hundred > 10:
                print(f"    ⚠️  {pct_hundred:.1f}% of values are ≥99.9%")
            if col.min() < 0:
                print(f"    ⚠️  NEGATIVE VALUES found (physically impossible)")
            if col.max() > 100:
                print(f"    ⚠️  VALUES > 100% found (physically impossible)")
    
    # ---- 5. CATEGORICAL VARIABLE DISTRIBUTIONS ----
    print(f"\n--- 5. CATEGORICAL FEATURES ---")
    cat_cols = ['chelator', 'texture', 'moisture', 'metal_level', 
                'ionic_level', 'ca_mg_level']
    for col in cat_cols:
        if col in df.columns:
            print(f"\n  {col}:")
            vc = df[col].value_counts()
            for val, count in vc.items():
                print(f"    {val:15s}: {count:6d} ({100*count/len(df):.1f}%)")
            if vc.std() / vc.mean() > 0.5:
                print(f"    ⚠️  Imbalanced distribution")
    
    # ---- 6. NUMERIC FEATURE RANGES ----
    print(f"\n--- 6. NUMERIC FEATURE RANGES ---")
    num_cols = ['ph', 'pb_mg_L', 'cu_mg_L', 'zn_mg_L', 'cd_mg_L',
                'doc_mg_L', 'ca_mg_L', 'mg_mg_L', 'na_mg_L', 'cl_mg_L',
                'dose_mg_L', 'hfo_sites', 'pe']
    for col in num_cols:
        if col in df.columns:
            unique = sorted(df[col].unique())
            print(f"  {col:15s}: {unique}")
    
    # ---- 7. COLLINEARITY CHECK ----
    print(f"\n--- 7. COLLINEARITY CHECK ---")
    print("  Checking for perfectly correlated feature pairs...")
    
    # Check if categorical labels perfectly predict numeric values
    colinear_pairs = [
        ('texture', 'hfo_sites', "Texture determines HFO sites"),
        ('texture', 'doc_mg_L', "Texture determines DOC (if tied)"),
        ('moisture', 'pe', "Moisture determines pe"),
        ('metal_level', 'pb_mg_L', "Metal level determines Pb conc"),
    ]
    for cat_col, num_col, reason in colinear_pairs:
        if cat_col in df.columns and num_col in df.columns:
            groups = df.groupby(cat_col)[num_col].nunique()
            if (groups == 1).all():
                print(f"  ⚠️  {cat_col} ↔ {num_col}: PERFECTLY CORRELATED ({reason})")
            else:
                print(f"  ✓  {cat_col} ↔ {num_col}: not perfectly correlated")
    
    # ---- 8. CHELATOR BASELINE CHECK ----
    print(f"\n--- 8. BASELINE (NO CHELATOR) CHECK ---")
    if 'chelator' in df.columns:
        has_none = (df['chelator'] == 'None').any()
        has_baseline = (df['chelator'].str.lower() == 'none').any()
        if has_none or has_baseline:
            n_baseline = ((df['chelator'] == 'None') | 
                         (df['chelator'].str.lower() == 'none')).sum()
            print(f"  ✓ Found {n_baseline} no-chelator baseline rows")
        else:
            print(f"  ⚠️  NO BASELINE (no-chelator) scenarios found!")
            print(f"      Chelator values present: {df['chelator'].unique().tolist()}")
            print(f"      Run generate_baseline_no_chelator.py to add baselines")
    
    # ---- 9. SCENARIO BALANCE ----
    print(f"\n--- 9. SCENARIO BALANCE ---")
    if 'chelator' in df.columns and 'ph' in df.columns:
        cross = pd.crosstab(df['chelator'], df['ph'])
        print(f"  Chelator × pH cross-tabulation:")
        print(cross.to_string())
    
    # ---- 10. DATA QUALITY SUMMARY ----
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    issues = []
    if total_nulls > 0:
        issues.append(f"Missing values ({total_nulls})")
    if len(dupes) > 0:
        issues.append(f"Duplicate column names ({len(dupes)})")
    for t in targets:
        if t in df.columns:
            if df[t].min() < 0:
                issues.append(f"Negative values in {t}")
            if df[t].max() > 100:
                issues.append(f"Values >100% in {t}")
    if not ((df.get('chelator', pd.Series()) == 'None').any()):
        issues.append("No baseline (no-chelator) scenarios")
    
    if issues:
        print(f"  Issues found ({len(issues)}):")
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print(f"  ✓ No critical issues found!")
    
    print(f"\n  Recommendation for ML training:")
    print(f"    - Drop redundant categorical columns (keep numeric equivalents)")
    print(f"    - Keep: ph, pb/cu/zn/cd_mg_L, doc_mg_L, ca/mg/na/cl_mg_L,")
    print(f"            chelator, dose_mg_L, hfo_sites, pe")
    print(f"    - Drop: metal_level, texture, moisture, ionic_level, ca_mg_level")
    print(f"            (these are perfectly predicted by their numeric counterparts)")
    print(f"    - Exception: keep 'texture' if you want it in the interface")


def main():
    # Audit whichever file exists
    files_to_check = [
        os.path.join(DATA_DIR, "complete_training_data_with_baseline.csv"),
        os.path.join(DATA_DIR, "complete_training_data.csv"),
    ]
    
    for f in files_to_check:
        if os.path.exists(f):
            audit_file(f)
            return
    
    print(f"No training data CSV found in {DATA_DIR}")
    print(f"Looked for:")
    for f in files_to_check:
        print(f"  {f}")


if __name__ == "__main__":
    main()
