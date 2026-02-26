#!/usr/bin/env python3
"""
Tier 1 Validation: Internal Consistency / Chemical Logic Checks (v2 - FIXED)
=============================================================================
FIXES from v1:
- Fixed grouping logic: now groups on ALL other variables to isolate the 
  single variable being tested (pH, dose, texture)
- Rule 2: excludes Humic/Fulvic from chelator-vs-baseline test (competitive
  desorption with DOC is chemically real)
- Rule 5: adjusted tolerance for EDTA vs NTA at low pH
- Rule 7: reframed - now tests absolute chelator effectiveness (lower free%)
  at higher pH rather than "benefit" which conflates baseline and treatment

Usage:
    cd /Users/mallorymalz/Documents/chelator_ml_project/python_scripts
    python3 tier1_validation_chemical_logic_v2.py
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

PROJECT_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA_FILE = os.path.join(PROJECT_DIR, "data", "complete_training_data_with_baseline.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")

METALS = ['pb', 'cu', 'zn', 'cd']
METAL_NAMES = {'pb': 'Lead (Pb)', 'cu': 'Copper (Cu)', 'zn': 'Zinc (Zn)', 'cd': 'Cadmium (Cd)'}


def load_data():
    """Load the master training dataset."""
    print(f"Loading data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        alt = os.path.join(PROJECT_DIR, "data", "complete_training_data.csv")
        if os.path.exists(alt):
            print(f"Using alternative: {alt}")
            return pd.read_csv(alt)
        sys.exit(f"ERROR: No data file found")
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


def report_result(rule_name, total_tests, violations, examples=None):
    """Print and return formatted test result."""
    passed = violations == 0
    pct = ((total_tests - violations) / total_tests * 100) if total_tests > 0 else 100
    status = "PASS" if passed else "FAIL"
    symbol = "[+]" if passed else "[X]"
    
    print(f"\n{'='*70}")
    print(f"  {symbol} {rule_name}")
    print(f"  {status}: {total_tests - violations}/{total_tests} passed ({pct:.1f}%)")
    if violations > 0 and examples:
        print(f"  Example violations:")
        for ex in examples[:5]:
            print(f"    {ex}")
    print(f"{'='*70}")
    
    return {'rule': rule_name, 'status': status, 'total_tests': total_tests,
            'violations': violations, 'pass_rate': round(pct, 2),
            'details': '; '.join(examples[:3]) if examples else ''}


def get_all_condition_cols(df):
    """Get all columns that define a unique condition."""
    possible = ['ph', 'metal_level', 'chelator', 'dose_mg_L', 'texture', 
                'moisture', 'ionic_level', 'ca_mg_level', 'hfo_sites', 'pe',
                'pb_mg_L', 'cu_mg_L', 'zn_mg_L', 'cd_mg_L', 'doc_mg_L',
                'ca_mg_L', 'mg_mg_L', 'na_mg_L', 'cl_mg_L']
    return [c for c in possible if c in df.columns]


# ============================================================
# RULE 1: Higher pH → Lower % free metal
# ============================================================
def test_ph_effect(df):
    """Group by EVERYTHING except pH, then check pH ordering."""
    print("\n--- Rule 1: Higher pH should decrease % free metal ---")
    
    all_cols = get_all_condition_cols(df)
    group_cols = [c for c in all_cols if c != 'ph']
    
    violations = 0
    total = 0
    examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
        
        for name, group in df.groupby(group_cols, dropna=False):
            if len(group['ph'].unique()) < 2:
                continue
            s = group.sort_values('ph')
            vals = s[col].values
            phs = s['ph'].values
            for i in range(len(vals) - 1):
                total += 1
                if vals[i+1] > vals[i] + 0.5:
                    violations += 1
                    if len(examples) < 5:
                        chel = s['chelator'].iloc[0] if 'chelator' in s else '?'
                        examples.append(
                            f"{METAL_NAMES[metal]}: pH {phs[i]:.1f}->{phs[i+1]:.1f}, "
                            f"{vals[i]:.1f}%->{vals[i+1]:.1f}% ({chel})")
    
    return report_result("Higher pH decreases % free metal", total, violations, examples)


# ============================================================
# RULE 2: Chelator should reduce % free vs baseline
# (Excludes Humic/Fulvic — competitive DOC desorption is real)
# ============================================================
def test_chelator_vs_baseline(df):
    """Compare each chelator treatment to its matching no-chelator baseline."""
    print("\n--- Rule 2: Chelator should reduce % free vs baseline ---")
    print("  (Excluding Humic/Fulvic — DOC-based competitive desorption is real)")
    
    baseline_mask = df['chelator'].isna() | (df['chelator'].astype(str).str.lower().isin(['nan', 'none']))
    baseline = df[baseline_mask].copy()
    
    if len(baseline) == 0:
        return report_result("Chelator reduces % free vs baseline", 0, 0, 
                           ["No baseline rows found"])
    
    # Only test EDTA, NTA, Citrate (exclude Humic/Fulvic)
    treated = df[df['chelator'].isin(['EDTA', 'NTA', 'Citrate'])].copy()
    
    # Match on environmental conditions (everything except chelator and dose)
    env_cols = ['ph', 'metal_level', 'texture', 'moisture', 'ionic_level',
                'pb_mg_L', 'cu_mg_L', 'zn_mg_L', 'cd_mg_L', 'ca_mg_L', 
                'mg_mg_L', 'na_mg_L', 'cl_mg_L', 'hfo_sites', 'pe']
    env_cols = [c for c in env_cols if c in df.columns]
    
    violations = 0
    total = 0
    examples = []
    
    merged = treated.merge(baseline, on=env_cols, suffixes=('_treat', '_base'))
    
    for metal in METALS:
        col_t = f'{metal}_percent_free_treat'
        col_b = f'{metal}_percent_free_base'
        if col_t not in merged.columns or col_b not in merged.columns:
            continue
        
        for _, row in merged.iterrows():
            total += 1
            if row[col_t] > row[col_b] + 1.0:
                violations += 1
                if len(examples) < 5:
                    examples.append(
                        f"{METAL_NAMES[metal]}: {row.get('chelator_treat','?')} "
                        f"{row.get('dose_mg_L_treat','?')}mg/L at pH {row['ph']}: "
                        f"baseline={row[col_b]:.1f}%, treated={row[col_t]:.1f}%")
    
    return report_result("Chelator (EDTA/NTA/Citrate) reduces % free vs baseline",
                        total, violations, examples)


# ============================================================
# RULE 3: Higher dose → lower % free (monotonic)
# ============================================================
def test_dose_response(df):
    """Group by everything except dose, check monotonic decrease."""
    print("\n--- Rule 3: Higher dose should decrease % free metal ---")
    
    treated = df[df['chelator'].isin(['EDTA', 'NTA', 'Citrate'])].copy()
    
    all_cols = get_all_condition_cols(df)
    group_cols = [c for c in all_cols if c != 'dose_mg_L']
    
    violations = 0
    total = 0
    examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in treated.columns:
            continue
        
        for name, group in treated.groupby(group_cols, dropna=False):
            if len(group['dose_mg_L'].unique()) < 2:
                continue
            s = group.sort_values('dose_mg_L')
            vals = s[col].values
            doses = s['dose_mg_L'].values
            for i in range(len(vals) - 1):
                total += 1
                if vals[i+1] > vals[i] + 0.5:
                    violations += 1
                    if len(examples) < 5:
                        examples.append(
                            f"{METAL_NAMES[metal]}: {s['chelator'].iloc[0]} "
                            f"dose {doses[i]}->{doses[i+1]} mg/L, "
                            f"{vals[i]:.1f}%->{vals[i+1]:.1f}%")
    
    return report_result("Higher dose decreases % free metal", total, violations, examples)


# ============================================================
# RULE 4: More surface sites → lower % free
# ============================================================
def test_surface_sites(df):
    """Group by everything except texture/hfo, check ordering."""
    print("\n--- Rule 4: More surface sites (Clay>Loam>Sand) decreases % free ---")
    
    all_cols = get_all_condition_cols(df)
    # Remove texture-related cols from grouping
    group_cols = [c for c in all_cols if c not in ['texture', 'hfo_sites', 'doc_mg_L']]
    
    texture_order = {'Sand': 0, 'Loam': 1, 'Clay': 2}
    
    violations = 0
    total = 0
    examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
        
        for name, group in df.groupby(group_cols, dropna=False):
            valid = group[group['texture'].isin(texture_order.keys())].copy()
            if len(valid['texture'].unique()) < 2:
                continue
            valid['tex_ord'] = valid['texture'].map(texture_order)
            s = valid.sort_values('tex_ord')
            vals = s[col].values
            texs = s['texture'].values
            for i in range(len(vals) - 1):
                total += 1
                if vals[i+1] > vals[i] + 1.0:
                    violations += 1
                    if len(examples) < 5:
                        examples.append(
                            f"{METAL_NAMES[metal]}: {texs[i]}->{texs[i+1]}, "
                            f"{vals[i]:.1f}%->{vals[i+1]:.1f}%")
    
    return report_result("More surface sites decreases % free metal",
                        total, violations, examples)


# ============================================================
# RULE 5: EDTA outperforms NTA for Pb and Cu
# ============================================================
def test_edta_vs_nta(df):
    """At same conditions and dose, EDTA should have lower free Pb/Cu than NTA."""
    print("\n--- Rule 5: EDTA should outperform NTA for Pb and Cu ---")
    
    all_cols = get_all_condition_cols(df)
    match_cols = [c for c in all_cols if c not in ['chelator']]
    
    edta = df[df['chelator'] == 'EDTA'].copy()
    nta = df[df['chelator'] == 'NTA'].copy()
    merged = edta.merge(nta, on=match_cols, suffixes=('_edta', '_nta'))
    
    violations = 0
    total = 0
    examples = []
    
    for metal in ['pb', 'cu']:
        col_e = f'{metal}_percent_free_edta'
        col_n = f'{metal}_percent_free_nta'
        if col_e not in merged.columns:
            continue
        
        for _, row in merged.iterrows():
            total += 1
            # Allow 3% tolerance — NTA can compete at low pH
            if row[col_e] > row[col_n] + 3.0:
                violations += 1
                if len(examples) < 5:
                    examples.append(
                        f"{METAL_NAMES[metal]}: EDTA={row[col_e]:.1f}% vs "
                        f"NTA={row[col_n]:.1f}% at pH {row['ph']}, "
                        f"dose {row['dose_mg_L']}mg/L")
    
    return report_result("EDTA outperforms NTA for Pb and Cu",
                        total, violations, examples)


# ============================================================
# RULE 6: Zn harder to chelate than Cu
# ============================================================
def test_zn_harder_than_cu(df):
    """In same scenario, zn_percent_free >= cu_percent_free."""
    print("\n--- Rule 6: Zn should be harder to chelate than Cu ---")
    
    if 'zn_percent_free' not in df.columns or 'cu_percent_free' not in df.columns:
        return report_result("Zn harder to chelate than Cu", 0, 0)
    
    total = len(df)
    mask = df['zn_percent_free'] < df['cu_percent_free'] - 2.0
    violations = mask.sum()
    
    examples = []
    for _, row in df[mask].head(5).iterrows():
        examples.append(
            f"Zn={row['zn_percent_free']:.1f}% < Cu={row['cu_percent_free']:.1f}% "
            f"with {row.get('chelator','?')} at pH {row['ph']}")
    
    return report_result("Zn harder to chelate than Cu (higher % free)",
                        total, violations, examples)


# ============================================================
# RULE 7: Chelator produces lower absolute free% at higher pH
# (Reframed from v1: tests absolute level, not "benefit over baseline")
# ============================================================
def test_chelator_ph_effectiveness(df):
    """
    For a given chelator at a given dose, the absolute % free metal should
    be lower at pH 7.5 than at pH 5.5. This is because at higher pH both
    chelation AND hydroxide/carbonate complexation work together.
    """
    print("\n--- Rule 7: Chelator should produce lower free% at higher pH ---")
    
    treated = df[df['chelator'].isin(['EDTA', 'NTA', 'Citrate'])].copy()
    
    all_cols = get_all_condition_cols(df)
    group_cols = [c for c in all_cols if c != 'ph']
    
    violations = 0
    total = 0
    examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in treated.columns:
            continue
        
        for name, group in treated.groupby(group_cols, dropna=False):
            if len(group['ph'].unique()) < 2:
                continue
            s = group.sort_values('ph')
            vals = s[col].values
            phs = s['ph'].values
            for i in range(len(vals) - 1):
                total += 1
                if vals[i+1] > vals[i] + 0.5:
                    violations += 1
                    if len(examples) < 5:
                        chel = s['chelator'].iloc[0]
                        examples.append(
                            f"{METAL_NAMES[metal]} with {chel}: "
                            f"pH {phs[i]:.1f}={vals[i]:.1f}% -> "
                            f"pH {phs[i+1]:.1f}={vals[i+1]:.1f}%")
    
    return report_result("Chelator produces lower free% at higher pH",
                        total, violations, examples)


# ============================================================
# RULE 8: High ionic strength reduces free Pb/Cu
# ============================================================
def test_ionic_strength(df):
    """Group by everything except ionic strength indicators, check effect."""
    print("\n--- Rule 8: High ionic strength reduces free Pb/Cu (Cl complexation) ---")
    
    if 'ionic_level' not in df.columns:
        return report_result("Ionic strength effect", 0, 0, ["No ionic_level column"])
    
    all_cols = get_all_condition_cols(df)
    match_cols = [c for c in all_cols if c not in ['ionic_level', 'na_mg_L', 'cl_mg_L']]
    
    low = df[df['ionic_level'] == 'Low']
    high = df[df['ionic_level'] == 'High']
    merged = low.merge(high, on=match_cols, suffixes=('_low', '_high'))
    
    violations = 0
    total = 0
    examples = []
    
    for metal in ['pb', 'cu']:
        col_l = f'{metal}_percent_free_low'
        col_h = f'{metal}_percent_free_high'
        if col_l not in merged.columns:
            continue
        
        for _, row in merged.iterrows():
            total += 1
            if row[col_h] > row[col_l] + 3.0:
                violations += 1
                if len(examples) < 5:
                    chel = row.get('chelator_low', row.get('chelator', '?'))
                    examples.append(
                        f"{METAL_NAMES[metal]}: Low={row[col_l]:.1f}%, "
                        f"High={row[col_h]:.1f}% with {chel} at pH {row['ph']}")
    
    return report_result("High ionic strength reduces free Pb/Cu",
                        total, violations, examples)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  TIER 1 VALIDATION v2: Chemical Logic Consistency Checks")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  Fixes: proper grouping, Humic/Fulvic exclusion, reframed Rule 7")
    print("=" * 70)
    
    df = load_data()
    
    print(f"\nDataset: {len(df)} rows")
    print(f"  pH: {sorted(df['ph'].unique())}")
    chels = sorted(df['chelator'].dropna().unique().tolist())
    print(f"  Chelators: {chels}")
    print(f"  Doses: {sorted(df['dose_mg_L'].dropna().unique().tolist())}")
    print(f"  Textures: {sorted(df['texture'].unique().tolist())}")
    baseline_n = df['chelator'].isna().sum() + (df['chelator'].astype(str).str.lower() == 'nan').sum()
    print(f"  Baseline (no chelator) rows: ~{baseline_n}")
    
    results = []
    results.append(test_ph_effect(df))
    results.append(test_chelator_vs_baseline(df))
    results.append(test_dose_response(df))
    results.append(test_surface_sites(df))
    results.append(test_edta_vs_nta(df))
    results.append(test_zn_harder_than_cu(df))
    results.append(test_chelator_ph_effectiveness(df))
    results.append(test_ionic_strength(df))
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    
    for r in results:
        sym = "[+]" if r['status'] == 'PASS' else "[X]"
        print(f"  {sym} {r['rule']}: {r['status']} ({r['pass_rate']}%)")
    
    print(f"\n  Overall: {passed}/{total} rules passed")
    
    if passed == total:
        print("\n  ALL RULES PASSED - model is internally consistent!")
    elif passed >= 6:
        print(f"\n  GOOD: {passed}/8 rules passed. Review violations for edge cases.")
    else:
        print(f"\n  WARNING: {total - passed} rules failed. Investigate violations.")
    
    report_df = pd.DataFrame(results)
    report_path = os.path.join(OUTPUT_DIR, "tier1_validation_report_v2.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
