#!/usr/bin/env python3
"""
DAY 1: Generate summary statistics and core tables for methodology paper.
Outputs:
  - Table 3: Summary statistics of free metal fractions
  - Table 4: ML model performance metrics (GB and RF)
  - Table for Tier 1 validation results
  - All saved to data/paper_tables/
"""

import pandas as pd
import numpy as np
import json
import os

# === PATHS ===
BASE = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA = os.path.join(BASE, "data", "complete_training_data_with_baseline.csv")
MODELS = os.path.join(BASE, "models")
TIER1 = os.path.join(BASE, "data", "tier1_validation_report_v2.csv")
OUTDIR = os.path.join(BASE, "data", "paper_tables")

os.makedirs(OUTDIR, exist_ok=True)

print("=" * 70)
print("  DAY 1: Summary Statistics & Core Tables for Methodology Paper")
print("=" * 70)

# =====================================================================
# TABLE 3: Summary Statistics of Simulated Free Metal Fractions
# =====================================================================
print("\n--- TABLE 3: Free Metal Fraction Summary Statistics ---\n")

df = pd.read_csv(DATA)
print(f"  Loaded {len(df)} rows from training data\n")

targets = ['pb_percent_free', 'cu_percent_free', 'zn_percent_free', 'cd_percent_free']
metal_names = ['Pb', 'Cu', 'Zn', 'Cd']

table3_rows = []
for col, name in zip(targets, metal_names):
    stats = {
        'Metal': name,
        'Mean (%)': round(df[col].mean(), 2),
        'Std Dev (%)': round(df[col].std(), 2),
        'Min (%)': round(df[col].min(), 2),
        'Q25 (%)': round(df[col].quantile(0.25), 2),
        'Median (%)': round(df[col].median(), 2),
        'Q75 (%)': round(df[col].quantile(0.75), 2),
        'Max (%)': round(df[col].max(), 2),
    }
    table3_rows.append(stats)

table3 = pd.DataFrame(table3_rows)
table3.to_csv(os.path.join(OUTDIR, "table3_summary_statistics.csv"), index=False)

print("  Table 3: Summary Statistics of Simulated Free Metal Fractions")
print("  " + "-" * 66)
print(table3.to_string(index=False))

# Also compute stats split by chelator vs no-chelator
print("\n  --- Breakdown: With Chelator vs No Treatment ---\n")
has_chelator = df[df['dose_mg_L'] > 0]
no_chelator = df[df['dose_mg_L'] == 0]

for col, name in zip(targets, metal_names):
    chel_mean = has_chelator[col].mean()
    none_mean = no_chelator[col].mean()
    reduction = none_mean - chel_mean
    print(f"  {name}: No Treatment = {none_mean:.1f}%  |  With Chelator = {chel_mean:.1f}%  |  Reduction = {reduction:.1f} pp")

# =====================================================================
# TABLE 4: ML Model Performance Metrics
# =====================================================================
print("\n\n--- TABLE 4: ML Model Performance Metrics ---\n")

# Load training report
report_path = os.path.join(MODELS, "training_report.json")
if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    results = report.get('results', report)
    table4_rows = []
    for name in metal_names:
        key = f"{name.lower()}_percent_free"
        if key in results:
            r = results[key]
            metrics = r.get('metrics', {})
            gb = metrics.get('GradientBoosting', {})
            rf = metrics.get('RandomForest', {})
            row = {
                'Metal': name,
                'Best Model': r.get('winner', 'GradientBoosting'),
                'GB R² (test)': round(gb.get('r2', 0), 4),
                'GB CV R² (5-fold)': round(gb.get('cv_r2_mean', 0), 4),
                'GB CV R² Std': round(gb.get('cv_r2_std', 0), 4),
                'GB RMSE (%)': round(gb.get('rmse', 0), 2),
                'RF R² (test)': round(rf.get('r2', 0), 4),
                'RF CV R²': round(rf.get('cv_r2_mean', 0), 4),
                'Top 3 Features': ', '.join(r.get('top_3_features', [])),
            }
            table4_rows.append(row)
    
    if table4_rows:
        table4 = pd.DataFrame(table4_rows)
        table4.to_csv(os.path.join(OUTDIR, "table4_model_performance.csv"), index=False)
        print("  Table 4: Model Performance Metrics")
        print("  " + "-" * 66)
        print(table4.to_string(index=False))
    else:
        print("  WARNING: Could not parse model results from training_report.json")
        print(f"  Keys found: {list(report.keys())}")
        print(f"  Dumping full content for inspection:\n")
        print(json.dumps(report, indent=2)[:2000])
else:
    print(f"  WARNING: training_report.json not found at {report_path}")
    print("  Checking what files exist in models/:")
    for f in os.listdir(MODELS):
        print(f"    {f}")

# =====================================================================
# FEATURE INFO (hyperparameters)
# =====================================================================
print("\n\n--- HYPERPARAMETERS ---\n")

feature_path = os.path.join(MODELS, "feature_info.json")
if os.path.exists(feature_path):
    with open(feature_path, 'r') as f:
        features = json.load(f)
    print(f"  Feature info loaded. Keys: {list(features.keys())}")
    if 'hyperparameters' in features:
        print(f"  Hyperparameters: {json.dumps(features['hyperparameters'], indent=4)}")
    else:
        print(f"  Full content:")
        print(json.dumps(features, indent=2)[:2000])
else:
    print(f"  feature_info.json not found at {feature_path}")

# =====================================================================
# TIER 1 VALIDATION TABLE
# =====================================================================
print("\n\n--- TIER 1 VALIDATION RESULTS TABLE ---\n")

if os.path.exists(TIER1):
    tier1 = pd.read_csv(TIER1)
    
    # Auto-detect columns
    print(f"  Tier 1 CSV columns: {list(tier1.columns)}")
    tier1_paper = tier1.copy()
    tier1_paper.to_csv(os.path.join(OUTDIR, "tier1_validation_summary.csv"), index=False)
    
    print("\n  Tier 1: Chemical Logic Validation Summary")
    print("  " + "-" * 66)
    print(tier1_paper.to_string(index=False))
    
    # Try to find pass rate column
    rate_col = None
    for c in tier1.columns:
        if 'rate' in c.lower() or 'pct' in c.lower() or 'percent' in c.lower():
            rate_col = c
            break
    
    if rate_col:
        perfect = len(tier1_paper[tier1_paper[rate_col] >= 99.99])
        total = len(tier1_paper)
        print(f"\n  {perfect}/{total} rules pass at 100%")
        print(f"  Lowest pass rate: {tier1_paper[rate_col].min():.1f}%")
else:
    print(f"  WARNING: Tier 1 report not found at {TIER1}")
    print("  Looking for alternatives...")
    data_dir = os.path.join(BASE, "data")
    for f in os.listdir(data_dir):
        if 'tier1' in f.lower() or 'validation' in f.lower():
            print(f"    Found: {f}")

# =====================================================================
# ADDITIONAL USEFUL STATS FOR THE PAPER
# =====================================================================
print("\n\n--- ADDITIONAL STATISTICS FOR PAPER TEXT ---\n")

# Dataset composition
print("  Dataset composition:")
print(f"    Total scenarios: {len(df)}")
print(f"    With chelator: {len(has_chelator)}")
print(f"    No-chelator baselines: {len(no_chelator)}")
print(f"    Unique pH levels: {sorted(df['ph'].unique())}")
print(f"    Chelator types: {sorted(df['chelator'].dropna().unique().tolist())}")
print(f"    Dose levels: {sorted(df['dose_mg_L'].unique())}")
print(f"    Textures: {sorted(df['texture'].unique().tolist())}")
print(f"    Moisture conditions: {sorted(df['moisture'].unique().tolist())}")
print(f"    Ionic levels: {sorted(df['ionic_level'].unique().tolist())}")

# Mean % free by chelator for each metal (useful for Section 3.1.3)
print("\n  Mean % free by chelator (for chelator ranking discussion):")
chel_means = df.groupby('chelator')[targets].mean()
chel_means.columns = metal_names
print(chel_means.round(1).to_string())

# Mean % free by pH (for Section 3.1.2)
print("\n  Mean % free by pH (for pH dominance discussion):")
ph_means = df.groupby('ph')[targets].mean()
ph_means.columns = metal_names
print(ph_means.round(1).to_string())

# Mean % free by texture (for Section 3.1.5)
print("\n  Mean % free by texture (for texture effects discussion):")
tex_means = df.groupby('texture')[targets].mean()
tex_means.columns = metal_names
print(tex_means.round(1).to_string())

# Mean % free by ionic level (for Section 3.1.4)
print("\n  Mean % free by ionic level (for ionic strength discussion):")
ion_means = df.groupby('ionic_level')[targets].mean()
ion_means.columns = metal_names
print(ion_means.round(1).to_string())

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("  DAY 1 COMPLETE")
print("=" * 70)
print(f"\n  Files saved to: {OUTDIR}/")
for f in sorted(os.listdir(OUTDIR)):
    print(f"    {f}")
print(f"\n  These tables are ready to reference while writing the paper.")
print(f"  Next: Day 2 — generate publication-quality figures.")
