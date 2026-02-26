#!/usr/bin/env python3
"""
Generate professionally formatted paper tables (clean column names, proper casing).
Saves to data/paper_tables/ as CSVs ready to paste into Word.
"""

import pandas as pd
import numpy as np
import json
import os

BASE = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA = os.path.join(BASE, "data", "complete_training_data_with_baseline.csv")
MODELS = os.path.join(BASE, "models")
TIER1 = os.path.join(BASE, "data", "tier1_validation_report_v2.csv")
OUTDIR = os.path.join(BASE, "data", "paper_tables")
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(DATA)

print("=" * 70)
print("  PROFESSIONAL PAPER TABLES")
print("=" * 70)

# =====================================================================
# TABLE 1: Parameter Space Design
# =====================================================================
print("\n--- TABLE 1: Simulation Parameter Space ---\n")

table1 = pd.DataFrame([
    {"Parameter": "pH", "Environmental Proxy": "Soil acidity", "Values": "5.5, 6.0, 6.5, 7.0, 7.5", "Units": "—", "Rationale": "Acidic to neutral range common in contaminated soils"},
    {"Parameter": "Metal concentrations (Pb, Cu, Zn, Cd)", "Environmental Proxy": "Contamination severity", "Values": "Low, Medium, High", "Units": "mg/L", "Rationale": "1x to 10x EPA screening level exceedances"},
    {"Parameter": "Chelator type", "Environmental Proxy": "Treatment agent", "Values": "EDTA, NTA, Citrate, Humic acid, Fulvic acid", "Units": "—", "Rationale": "Industry standard plus biodegradable alternatives"},
    {"Parameter": "Chelator dose", "Environmental Proxy": "Treatment intensity", "Values": "50, 150, 300", "Units": "mg/L", "Rationale": "Sub-stoichiometric to excess relative to metals"},
    {"Parameter": "Soil texture", "Environmental Proxy": "Surface area, sorption capacity", "Values": "Sand (0.1), Loam (0.5), Clay (1.5)", "Units": "mol Hfo_wOH", "Rationale": "Iron oxide content scaled by texture class"},
    {"Parameter": "Dissolved organic carbon", "Environmental Proxy": "Organic matter content", "Values": "10, 25, 40", "Units": "mg/L", "Rationale": "Tied to texture: Sand = 10, Loam = 25, Clay = 40"},
    {"Parameter": "pe (electron activity)", "Environmental Proxy": "Moisture/redox condition", "Values": "12 (Dry), 8 (Mesic), 3 (Wet)", "Units": "—", "Rationale": "Oxidizing to reducing conditions"},
    {"Parameter": "Na/Cl concentration", "Environmental Proxy": "Ionic strength/salinity", "Values": "Low, Medium, High", "Units": "mg/L", "Rationale": "Non-saline to coastal/road-salt-impacted (RI-specific)"},
    {"Parameter": "Ca/Mg concentration", "Environmental Proxy": "Competing cations", "Values": "Low (20/10), High (100/50)", "Units": "mg/L", "Rationale": "Competition for chelator binding sites"},
    {"Parameter": "No chelator (baseline)", "Environmental Proxy": "Untreated reference", "Values": "Dose = 0", "Units": "—", "Rationale": "Required for calculating chelator effectiveness"},
])

table1.to_csv(os.path.join(OUTDIR, "table1_parameter_space.csv"), index=False)
print(table1.to_string(index=False))

# =====================================================================
# TABLE 3: Summary Statistics
# =====================================================================
print("\n\n--- TABLE 3: Summary Statistics of Simulated Free Metal Fractions ---\n")

targets = ['pb_percent_free', 'cu_percent_free', 'zn_percent_free', 'cd_percent_free']
metal_labels = ['Pb', 'Cu', 'Zn', 'Cd']

table3_rows = []
for col, name in zip(targets, metal_labels):
    table3_rows.append({
        'Metal': name,
        'Mean (%)': round(df[col].mean(), 1),
        'Std. Dev. (%)': round(df[col].std(), 1),
        'Min. (%)': round(df[col].min(), 1),
        '25th Pctl. (%)': round(df[col].quantile(0.25), 1),
        'Median (%)': round(df[col].median(), 1),
        '75th Pctl. (%)': round(df[col].quantile(0.75), 1),
        'Max. (%)': round(df[col].max(), 1),
    })

table3 = pd.DataFrame(table3_rows)
table3.to_csv(os.path.join(OUTDIR, "table3_summary_statistics.csv"), index=False)
print(table3.to_string(index=False))

# =====================================================================
# TABLE 4: Model Performance
# =====================================================================
print("\n\n--- TABLE 4: Model Performance Metrics ---\n")

report_path = os.path.join(MODELS, "training_report.json")
with open(report_path, 'r') as f:
    report = json.load(f)

results = report.get('results', {})
table4_rows = []
for name in metal_labels:
    key = f"{name.lower()}_percent_free"
    if key in results:
        r = results[key]
        gb = r.get('metrics', {}).get('GradientBoosting', {})
        rf = r.get('metrics', {}).get('RandomForest', {})
        table4_rows.append({
            'Target': f'{name} % Free',
            'GB R\u00b2 (Test)': f"{gb.get('r2', 0):.4f}",
            'GB R\u00b2 (CV, 5-Fold)': f"{gb.get('cv_r2_mean', 0):.4f}",
            'GB RMSE (%)': f"{gb.get('rmse', 0):.2f}",
            'RF R\u00b2 (Test)': f"{rf.get('r2', 0):.4f}",
            'RF R\u00b2 (CV, 5-Fold)': f"{rf.get('cv_r2_mean', 0):.4f}",
            'Top 3 Features': ', '.join(r.get('top_3_features', [])),
        })

# Clean up feature names in Top 3
feature_display = {
    'ph': 'pH',
    'chelator_encoded': 'Chelator type',
    'dose_mg_L': 'Chelator dose',
    'pe': 'pe (redox)',
    'cd_mg_L': 'Cd concentration',
    'cl_mg_L': 'Cl concentration',
    'na_mg_L': 'Na concentration',
    'hfo_sites': 'Surface sites (Hfo)',
    'doc_mg_L': 'DOC',
    'pb_mg_L': 'Pb concentration',
    'cu_mg_L': 'Cu concentration',
    'zn_mg_L': 'Zn concentration',
    'ca_mg_L': 'Ca concentration',
    'mg_mg_L': 'Mg concentration',
}

for row in table4_rows:
    raw_features = row['Top 3 Features'].split(', ')
    clean_features = [feature_display.get(f.strip(), f.strip()) for f in raw_features]
    row['Top 3 Features'] = ', '.join(clean_features)

table4 = pd.DataFrame(table4_rows)
table4.to_csv(os.path.join(OUTDIR, "table4_model_performance.csv"), index=False)
print(table4.to_string(index=False))

# =====================================================================
# TABLE 5: Tier 1 Validation (paper-ready, concise)
# =====================================================================
print("\n\n--- TABLE 5: Internal Consistency Validation (Tier 1) ---\n")

tier1 = pd.read_csv(TIER1)

# Build clean paper version with short scientific explanations
explanations = {
    'Higher pH decreases % free metal': '—',
    'Chelator (EDTA/NTA/Citrate) reduces % free vs baseline': 'Low-dose chelator at pH 5.5 causes competitive desorption from surface sites',
    'Higher dose decreases % free metal': '—',
    'More surface sites decreases % free metal': '—',
    'EDTA outperforms NTA for Pb and Cu': 'NTA less affected by protonation at pH 5.5; differential binding kinetics',
    'Zn harder to chelate than Cu (higher % free)': 'Cu-organic matter affinity (Irving-Williams series) reverses order with Humic/Fulvic at pH 5.5',
    'Chelator produces lower free% at higher pH': '—',
    'High ionic strength reduces free Pb/Cu': 'Effect reverses for no-chelator baseline at pH 7.5 where metals already precipitated',
}

table5_rows = []
for _, row in tier1.iterrows():
    rule = row['rule']
    table5_rows.append({
        'Chemical Logic Rule': rule,
        'Status': row['status'],
        'Pass Rate (%)': round(row['pass_rate'], 1),
        'Tests (n)': int(row['total_tests']),
        'Violations (n)': int(row['violations']),
        'Explanation of Violations': explanations.get(rule, '—'),
    })

table5 = pd.DataFrame(table5_rows)
table5.to_csv(os.path.join(OUTDIR, "table5_tier1_validation.csv"), index=False)
print(table5.to_string(index=False))

# =====================================================================
# TABLE 6: Mean % Free by Chelator and Metal (for heatmap / Section 3.1.3)
# =====================================================================
print("\n\n--- TABLE 6: Mean Percent Free Metal by Chelator (at all conditions) ---\n")

chelator_order = ['EDTA', 'NTA', 'Citrate', 'Humic', 'Fulvic']
# Include no-chelator baseline
chel_means = df.groupby('chelator')[targets].mean()

table6_rows = []
# Add no-treatment first
no_treat = df[df['dose_mg_L'] == 0]
if len(no_treat) > 0:
    table6_rows.append({
        'Chelator': 'No Treatment',
        'Pb (% Free)': round(no_treat['pb_percent_free'].mean(), 1),
        'Cu (% Free)': round(no_treat['cu_percent_free'].mean(), 1),
        'Zn (% Free)': round(no_treat['zn_percent_free'].mean(), 1),
        'Cd (% Free)': round(no_treat['cd_percent_free'].mean(), 1),
    })

for chel in chelator_order:
    if chel in chel_means.index:
        table6_rows.append({
            'Chelator': chel,
            'Pb (% Free)': round(chel_means.loc[chel, 'pb_percent_free'], 1),
            'Cu (% Free)': round(chel_means.loc[chel, 'cu_percent_free'], 1),
            'Zn (% Free)': round(chel_means.loc[chel, 'zn_percent_free'], 1),
            'Cd (% Free)': round(chel_means.loc[chel, 'cd_percent_free'], 1),
        })

table6 = pd.DataFrame(table6_rows)
table6.to_csv(os.path.join(OUTDIR, "table6_chelator_comparison.csv"), index=False)
print(table6.to_string(index=False))

# =====================================================================
# DONE
# =====================================================================
print("\n" + "=" * 70)
print("  ALL PROFESSIONAL TABLES SAVED")
print("=" * 70)
print(f"\n  Location: {OUTDIR}/")
for f in sorted(os.listdir(OUTDIR)):
    fpath = os.path.join(OUTDIR, f)
    print(f"    {f}  ({os.path.getsize(fpath)} bytes)")
print("\n  Open these CSVs in Excel/Numbers, then copy-paste into your Word document.")
print("  All column names are publication-ready with proper formatting.")
