#!/usr/bin/env python3
"""
merge_baseline_into_training.py
================================
Merges the baseline no-chelator scenarios into the complete training dataset.

Creates a new master file: complete_training_data_with_baseline.csv
Also backs up the original complete_training_data.csv

Run AFTER generate_baseline_no_chelator.py has completed successfully.

Usage:
    python3 merge_baseline_into_training.py
"""

import pandas as pd
import os
import shutil

BASE_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA_DIR = os.path.join(BASE_DIR, "data")

ORIGINAL_FILE = os.path.join(DATA_DIR, "complete_training_data.csv")
BASELINE_FILE = os.path.join(DATA_DIR, "baseline_no_chelator.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "complete_training_data_with_baseline.csv")
BACKUP_FILE = os.path.join(DATA_DIR, "complete_training_data_BACKUP.csv")


def main():
    # Verify files exist
    if not os.path.exists(ORIGINAL_FILE):
        print(f"ERROR: Original training data not found: {ORIGINAL_FILE}")
        return
    if not os.path.exists(BASELINE_FILE):
        print(f"ERROR: Baseline data not found: {BASELINE_FILE}")
        print("  Run generate_baseline_no_chelator.py first!")
        return

    # Load datasets
    print("Loading datasets...")
    original = pd.read_csv(ORIGINAL_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    print(f"  Original training data: {original.shape[0]} rows × {original.shape[1]} columns")
    print(f"  Baseline (no chelator): {baseline.shape[0]} rows × {baseline.shape[1]} columns")

    # Check for column alignment
    # The baseline may have a 'scenario_id' column the original doesn't
    # Drop it if present for merging
    if 'scenario_id' in baseline.columns and 'scenario_id' not in original.columns:
        baseline = baseline.drop(columns=['scenario_id'])
    
    # Find any columns in original but not in baseline, and vice versa
    orig_cols = set(original.columns)
    base_cols = set(baseline.columns)
    
    missing_in_baseline = orig_cols - base_cols
    extra_in_baseline = base_cols - orig_cols
    
    if missing_in_baseline:
        print(f"\n  Columns in original but not baseline: {missing_in_baseline}")
        print("  These will be filled with appropriate defaults.")
        for col in missing_in_baseline:
            if col == 'scenario_id':
                continue
            baseline[col] = None  # Will be handled case-by-case
    
    if extra_in_baseline:
        print(f"\n  Extra columns in baseline (dropping): {extra_in_baseline}")
        baseline = baseline.drop(columns=list(extra_in_baseline))

    # Align column order to match original
    common_cols = [c for c in original.columns if c in baseline.columns]
    baseline = baseline[common_cols]

    # Backup original
    print(f"\n  Backing up original to: {BACKUP_FILE}")
    shutil.copy2(ORIGINAL_FILE, BACKUP_FILE)

    # Merge
    combined = pd.concat([original, baseline], ignore_index=True)

    print(f"\n  Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns")
    print(f"    Original chelator rows: {original.shape[0]}")
    print(f"    Baseline (None) rows:   {baseline.shape[0]}")
    print(f"    Total:                  {combined.shape[0]}")

    # Verify chelator distribution
    print(f"\n  Chelator distribution in combined data:")
    print(combined['chelator'].value_counts().to_string())

    # Save
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved to: {OUTPUT_FILE}")

    # Summary statistics for baseline scenarios
    print(f"\n{'=' * 60}")
    print("BASELINE SCENARIO SUMMARY (no chelator)")
    print(f"{'=' * 60}")
    for metal in ['pb', 'cu', 'zn', 'cd']:
        col = f"{metal}_percent_free"
        if col in baseline.columns:
            print(f"  {metal.upper()} % free: "
                  f"mean={baseline[col].mean():.1f}%, "
                  f"min={baseline[col].min():.1f}%, "
                  f"max={baseline[col].max():.1f}%")

    print(f"\nDone! Use '{OUTPUT_FILE}' for ML training.")
    print("This file includes both chelator AND no-chelator scenarios.")


if __name__ == "__main__":
    main()
