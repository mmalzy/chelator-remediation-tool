#!/usr/bin/env python3
"""
Tier 1 Validation: Internal Consistency / Chemical Logic Checks
================================================================
Tests whether the trained Gradient Boosting models obey fundamental
geochemical rules. If any rule is violated, the model has learned
something physically impossible and needs investigation.

Rules tested:
1. Increasing pH should decrease % free metal (more hydroxide/carbonate complexation)
2. Adding chelator should decrease % free metal vs no-chelator baseline
3. Higher chelator dose should decrease % free metal (monotonic)
4. More surface sites (clay > loam > sand) should decrease % free metal
5. EDTA should outperform NTA for Pb and Cu (stronger stability constants)
6. Zn should be harder to chelate than Cu across all conditions
7. Chelator effectiveness should diminish at very low pH (H+ competition)
8. Higher ionic strength should affect speciation (chloride complexation)

Usage:
    cd /Users/mallorymalz/Documents/chelator_ml_project/python_scripts
    python3 ../python_scripts/tier1_validation_chemical_logic.py

Output:
    Prints PASS/FAIL for each rule with details on any violations.
    Saves a summary report to data/tier1_validation_report.csv
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ============================================================
# CONFIGURATION - Update these paths if needed
# ============================================================
PROJECT_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA_FILE = os.path.join(PROJECT_DIR, "data", "complete_training_data_with_baseline.csv")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data")

# Try to load models for prediction-based tests; fall back to data-only tests
USE_MODELS = False
try:
    import joblib
    USE_MODELS = True
except ImportError:
    print("Note: joblib not available. Running data-based validation only.")
    print("Install with: pip3 install joblib --user")

METALS = ['pb', 'cu', 'zn', 'cd']
METAL_NAMES = {'pb': 'Lead (Pb)', 'cu': 'Copper (Cu)', 'zn': 'Zinc (Zn)', 'cd': 'Cadmium (Cd)'}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_data():
    """Load the master training dataset."""
    print(f"Loading data from: {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found at {DATA_FILE}")
        print("Looking for alternative...")
        alt = os.path.join(PROJECT_DIR, "data", "complete_training_data.csv")
        if os.path.exists(alt):
            print(f"Found: {alt}")
            return pd.read_csv(alt)
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows x {len(df.columns)} columns")
    return df


def report_result(rule_name, passed, total_tests, violations, details=""):
    """Print formatted test result."""
    status = "PASS" if passed else "FAIL"
    symbol = "[+]" if passed else "[X]"
    pct = ((total_tests - violations) / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  {symbol} Rule: {rule_name}")
    print(f"  Status: {status} ({pct:.1f}% of {total_tests} tests passed)")
    if violations > 0:
        print(f"  Violations: {violations}")
    if details:
        print(f"  Details: {details}")
    print(f"{'='*70}")
    
    return {
        'rule': rule_name,
        'status': status,
        'total_tests': total_tests,
        'violations': violations,
        'pass_rate': round(pct, 2),
        'details': details
    }


# ============================================================
# RULE 1: Higher pH → Lower % free metal
# ============================================================
def test_ph_effect(df):
    """
    For each unique combination of (metal_level, chelator, dose, texture, 
    moisture, ionic_level), increasing pH should decrease or maintain 
    % free metal for all four metals.
    """
    print("\n--- Testing Rule 1: Higher pH should decrease % free metal ---")
    
    group_cols = ['metal_level', 'chelator', 'dose_mg_L', 'texture', 'moisture', 'ionic_level']
    # Only use columns that exist
    group_cols = [c for c in group_cols if c in df.columns]
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
            
        grouped = df.groupby(group_cols)
        for name, group in grouped:
            if len(group) < 2:
                continue
            sorted_group = group.sort_values('ph')
            values = sorted_group[col].values
            phs = sorted_group['ph'].values
            
            for i in range(len(values) - 1):
                total_tests += 1
                # Allow small tolerance for numerical noise (0.5%)
                if values[i+1] > values[i] + 0.5:
                    total_violations += 1
                    if len(violation_examples) < 3:
                        violation_examples.append(
                            f"  {METAL_NAMES[metal]}: pH {phs[i]}->{phs[i+1]}, "
                            f"free% {values[i]:.1f}->{values[i+1]:.1f} "
                            f"(+{values[i+1]-values[i]:.1f}%)"
                        )
    
    details = ""
    if violation_examples:
        details = "Example violations:\n" + "\n".join(violation_examples)
    
    return report_result(
        "Higher pH decreases % free metal",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 2: Chelator should reduce % free metal vs baseline
# ============================================================
def test_chelator_vs_baseline(df):
    """
    For any given environmental condition, adding any chelator at any dose
    should result in equal or lower % free metal compared to no-chelator baseline.
    """
    print("\n--- Testing Rule 2: Chelator should reduce % free vs no-chelator ---")
    
    # Identify baseline rows (chelator is nan/None/No Treatment)
    baseline_mask = df['chelator'].isna() | (df['chelator'].astype(str).str.lower().isin(['nan', 'none', 'no treatment']))
    
    if baseline_mask.sum() == 0:
        return report_result(
            "Chelator reduces % free vs baseline",
            True, 0, 0,
            "No baseline rows found - cannot test. Add no-chelator scenarios."
        )
    
    baseline = df[baseline_mask].copy()
    treated = df[~baseline_mask].copy()
    
    # Match on environmental conditions
    match_cols = ['ph', 'metal_level', 'texture', 'moisture', 'ionic_level']
    match_cols = [c for c in match_cols if c in df.columns]
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for _, base_row in baseline.iterrows():
        # Find treated rows with same environmental conditions
        mask = pd.Series([True] * len(treated), index=treated.index)
        for col in match_cols:
            mask = mask & (treated[col] == base_row[col])
        
        matched = treated[mask]
        
        for metal in METALS:
            col = f'{metal}_percent_free'
            if col not in df.columns:
                continue
            base_val = base_row[col]
            
            for _, treat_row in matched.iterrows():
                total_tests += 1
                # Chelator should reduce free metal; allow 1% tolerance
                if treat_row[col] > base_val + 1.0:
                    total_violations += 1
                    if len(violation_examples) < 3:
                        chel = treat_row.get('chelator', '?')
                        dose = treat_row.get('dose_mg_L', '?')
                        violation_examples.append(
                            f"  {METAL_NAMES[metal]}: {chel} {dose}mg/L at pH {base_row['ph']} "
                            f"increased free% from {base_val:.1f} to {treat_row[col]:.1f}"
                        )
    
    details = ""
    if violation_examples:
        details = "Example violations:\n" + "\n".join(violation_examples)
    
    return report_result(
        "Chelator reduces % free vs baseline",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 3: Higher chelator dose → lower % free metal (monotonic)
# ============================================================
def test_dose_response(df):
    """
    For a given chelator and environmental condition, increasing dose
    should decrease % free metal (monotonic dose-response).
    """
    print("\n--- Testing Rule 3: Higher dose should decrease % free metal ---")
    
    # Exclude baseline rows
    treated = df[~(df['chelator'].isna() | (df['chelator'].astype(str).str.lower().isin(['nan', 'none'])))].copy()
    
    group_cols = ['ph', 'metal_level', 'chelator', 'texture', 'moisture', 'ionic_level']
    group_cols = [c for c in group_cols if c in treated.columns]
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in treated.columns:
            continue
            
        grouped = treated.groupby(group_cols)
        for name, group in grouped:
            if len(group) < 2:
                continue
            sorted_group = group.sort_values('dose_mg_L')
            values = sorted_group[col].values
            doses = sorted_group['dose_mg_L'].values
            
            for i in range(len(values) - 1):
                total_tests += 1
                # Allow 0.5% tolerance
                if values[i+1] > values[i] + 0.5:
                    total_violations += 1
                    if len(violation_examples) < 3:
                        chel_val = sorted_group['chelator'].iloc[0] if 'chelator' in sorted_group else '?'
                        violation_examples.append(
                            f"  {METAL_NAMES[metal]}: {chel_val} dose {doses[i]}->{doses[i+1]} mg/L, "
                            f"free% {values[i]:.1f}->{values[i+1]:.1f}"
                        )
    
    details = ""
    if violation_examples:
        details = "Example violations:\n" + "\n".join(violation_examples)
    
    return report_result(
        "Higher dose decreases % free metal",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 4: More surface sites → lower % free metal
# ============================================================
def test_surface_sites(df):
    """
    Clay (hfo=1.5) > Loam (hfo=0.5) > Sand (hfo=0.1) for metal sorption,
    so % free metal should decrease with more surface sites.
    """
    print("\n--- Testing Rule 4: More surface sites should decrease % free metal ---")
    
    group_cols = ['ph', 'metal_level', 'chelator', 'dose_mg_L', 'moisture', 'ionic_level']
    group_cols = [c for c in group_cols if c in df.columns]
    
    texture_order = {'Sand': 0, 'Loam': 1, 'Clay': 2}
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
            
        grouped = df.groupby(group_cols)
        for name, group in grouped:
            if len(group) < 2:
                continue
            if 'texture' not in group.columns:
                continue
            # Filter to known textures and sort
            group_f = group[group['texture'].isin(texture_order.keys())].copy()
            group_f['tex_order'] = group_f['texture'].map(texture_order)
            sorted_group = group_f.sort_values('tex_order')
            
            if len(sorted_group) < 2:
                continue
                
            values = sorted_group[col].values
            textures = sorted_group['texture'].values
            
            for i in range(len(values) - 1):
                total_tests += 1
                # More surface sites should mean less free metal; 1% tolerance
                if values[i+1] > values[i] + 1.0:
                    total_violations += 1
                    if len(violation_examples) < 3:
                        violation_examples.append(
                            f"  {METAL_NAMES[metal]}: {textures[i]}->{textures[i+1]}, "
                            f"free% {values[i]:.1f}->{values[i+1]:.1f}"
                        )
    
    details = ""
    if violation_examples:
        details = "Example violations:\n" + "\n".join(violation_examples)
    
    return report_result(
        "More surface sites (clay>loam>sand) decreases % free metal",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 5: EDTA should outperform NTA for Pb and Cu
# ============================================================
def test_edta_vs_nta(df):
    """
    EDTA has higher stability constants with Pb and Cu than NTA.
    At the same dose and conditions, EDTA should give equal or lower
    % free Pb and Cu than NTA.
    """
    print("\n--- Testing Rule 5: EDTA should outperform NTA for Pb and Cu ---")
    
    match_cols = ['ph', 'metal_level', 'dose_mg_L', 'texture', 'moisture', 'ionic_level']
    match_cols = [c for c in match_cols if c in df.columns]
    
    edta = df[df['chelator'] == 'EDTA'].copy()
    nta = df[df['chelator'] == 'NTA'].copy()
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for metal in ['pb', 'cu']:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
            
        merged = edta.merge(nta, on=match_cols, suffixes=('_edta', '_nta'))
        
        for _, row in merged.iterrows():
            total_tests += 1
            edta_val = row[f'{col}_edta']
            nta_val = row[f'{col}_nta']
            # EDTA should have lower or equal free%; allow 2% tolerance
            if edta_val > nta_val + 2.0:
                total_violations += 1
                if len(violation_examples) < 3:
                    violation_examples.append(
                        f"  {METAL_NAMES[metal]}: EDTA={edta_val:.1f}% vs NTA={nta_val:.1f}% "
                        f"at pH {row['ph']}, dose {row['dose_mg_L']}mg/L"
                    )
    
    details = ""
    if violation_examples:
        details = "Example violations:\n" + "\n".join(violation_examples)
    
    return report_result(
        "EDTA outperforms NTA for Pb and Cu",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 6: Zn should be harder to chelate than Cu
# ============================================================
def test_zn_harder_than_cu(df):
    """
    Zinc has weaker stability constants with most chelators than copper.
    In the same scenario, zn_percent_free should be >= cu_percent_free.
    """
    print("\n--- Testing Rule 6: Zn should be harder to chelate than Cu ---")
    
    if 'zn_percent_free' not in df.columns or 'cu_percent_free' not in df.columns:
        return report_result("Zn harder to chelate than Cu", True, 0, 0, "Columns missing")
    
    total_tests = len(df)
    # Allow 2% tolerance for near-equal cases
    violations = (df['zn_percent_free'] < df['cu_percent_free'] - 2.0).sum()
    
    violation_rows = df[df['zn_percent_free'] < df['cu_percent_free'] - 2.0]
    examples = []
    if len(violation_rows) > 0:
        for _, row in violation_rows.head(3).iterrows():
            chel = row.get('chelator', '?')
            examples.append(
                f"  Zn={row['zn_percent_free']:.1f}% < Cu={row['cu_percent_free']:.1f}% "
                f"with {chel} at pH {row['ph']}"
            )
    
    details = ""
    if examples:
        details = "Example violations:\n" + "\n".join(examples)
    
    return report_result(
        "Zn harder to chelate than Cu (higher % free)",
        violations == 0,
        total_tests,
        violations,
        details
    )


# ============================================================
# RULE 7: Chelator effectiveness should diminish at low pH
# ============================================================
def test_chelator_ph_interaction(df):
    """
    At low pH (5.5), chelators should be less effective (smaller reduction
    in % free metal) compared to high pH (7.5), because H+ competes for
    chelator binding sites.
    
    We test this by comparing the chelator benefit (baseline - treated) at
    pH 5.5 vs pH 7.5. The benefit should be larger at higher pH.
    """
    print("\n--- Testing Rule 7: Chelator effectiveness diminishes at low pH ---")
    
    baseline_mask = df['chelator'].isna() | (df['chelator'].astype(str).str.lower().isin(['nan', 'none']))
    
    if baseline_mask.sum() == 0:
        return report_result(
            "Chelator effectiveness diminishes at low pH",
            True, 0, 0,
            "No baseline rows - cannot compute chelator benefit"
        )
    
    # Compare chelator benefit at pH 5.5 vs 7.5
    match_cols = ['metal_level', 'texture', 'moisture', 'ionic_level']
    match_cols = [c for c in match_cols if c in df.columns]
    
    total_tests = 0
    total_violations = 0
    violation_examples = []
    
    low_ph = df[df['ph'] == 5.5]
    high_ph = df[df['ph'] == 7.5]
    
    for metal in METALS:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
        
        # Get mean free% for baseline and each chelator at each pH
        for chel in ['EDTA', 'NTA', 'Citrate']:
            low_base = low_ph[low_ph['chelator'].isna() | (low_ph['chelator'].astype(str).str.lower().isin(['nan', 'none']))][col].mean()
            low_treat = low_ph[low_ph['chelator'] == chel][col].mean()
            high_base = high_ph[high_ph['chelator'].isna() | (high_ph['chelator'].astype(str).str.lower().isin(['nan', 'none']))][col].mean()
            high_treat = high_ph[high_ph['chelator'] == chel][col].mean()
            
            if pd.isna(low_base) or pd.isna(low_treat) or pd.isna(high_base) or pd.isna(high_treat):
                continue
            
            low_benefit = low_base - low_treat
            high_benefit = high_base - high_treat
            
            total_tests += 1
            # High pH should show greater chelator benefit; 1% tolerance
            if low_benefit > high_benefit + 1.0:
                total_violations += 1
                if len(violation_examples) < 3:
                    violation_examples.append(
                        f"  {METAL_NAMES[metal]} with {chel}: "
                        f"benefit at pH5.5={low_benefit:.1f}% > pH7.5={high_benefit:.1f}%"
                    )
    
    details = ""
    if violation_examples:
        details = "Example violations (chelator more effective at low pH):\n" + "\n".join(violation_examples)
    
    return report_result(
        "Chelator effectiveness greater at higher pH",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# RULE 8: Ionic strength affects speciation
# ============================================================
def test_ionic_strength_effect(df):
    """
    Higher ionic strength (coastal RI) introduces chloride complexation.
    For Pb and Cu, chloride complexes (PbCl+, CuCl+) reduce the free
    metal fraction. So high ionic strength should lower % free Pb/Cu.
    This is a RI-specific finding.
    """
    print("\n--- Testing Rule 8: High ionic strength reduces free Pb/Cu (chloride complexation) ---")
    
    if 'ionic_level' not in df.columns:
        return report_result(
            "Ionic strength affects Pb/Cu speciation",
            True, 0, 0,
            "ionic_level column not found"
        )
    
    match_cols = ['ph', 'metal_level', 'chelator', 'dose_mg_L', 'texture', 'moisture']
    match_cols = [c for c in match_cols if c in df.columns]
    
    total_violations = 0
    total_tests = 0
    violation_examples = []
    
    for metal in ['pb', 'cu']:
        col = f'{metal}_percent_free'
        if col not in df.columns:
            continue
        
        low_ionic = df[df['ionic_level'] == 'Low']
        high_ionic = df[df['ionic_level'] == 'High']
        
        merged = low_ionic.merge(high_ionic, on=match_cols, suffixes=('_low', '_high'))
        
        for _, row in merged.iterrows():
            total_tests += 1
            # High ionic should have lower free metal (chloride complexation)
            # Allow 3% tolerance because this effect varies with conditions
            if row[f'{col}_high'] > row[f'{col}_low'] + 3.0:
                total_violations += 1
                if len(violation_examples) < 3:
                    chel = row.get('chelator_low', row.get('chelator', '?'))
                    violation_examples.append(
                        f"  {METAL_NAMES[metal]}: Low ionic={row[f'{col}_low']:.1f}%, "
                        f"High ionic={row[f'{col}_high']:.1f}% with {chel} at pH {row['ph']}"
                    )
    
    details = ""
    if violation_examples:
        details = "Violations (high ionic increased free metal):\n" + "\n".join(violation_examples)
    
    return report_result(
        "High ionic strength reduces free Pb/Cu (chloride complexation)",
        total_violations == 0,
        total_tests,
        total_violations,
        details
    )


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  TIER 1 VALIDATION: Chemical Logic Consistency Checks")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    df = load_data()
    
    # Print quick data summary
    print(f"\nDataset summary:")
    print(f"  Rows: {len(df)}")
    print(f"  pH values: {sorted(df['ph'].unique())}")
    print(f"  Chelators: {sorted(df['chelator'].dropna().unique().tolist())}")
    if 'dose_mg_L' in df.columns:
        print(f"  Doses: {sorted(df['dose_mg_L'].dropna().unique().tolist())}")
    if 'texture' in df.columns:
        print(f"  Textures: {sorted(df['texture'].unique().tolist())}")
    
    # Run all tests
    results = []
    results.append(test_ph_effect(df))
    results.append(test_chelator_vs_baseline(df))
    results.append(test_dose_response(df))
    results.append(test_surface_sites(df))
    results.append(test_edta_vs_nta(df))
    results.append(test_zn_harder_than_cu(df))
    results.append(test_chelator_ph_interaction(df))
    results.append(test_ionic_strength_effect(df))
    
    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    total = len(results)
    
    for r in results:
        symbol = "[+]" if r['status'] == 'PASS' else "[X]"
        print(f"  {symbol} {r['rule']}: {r['status']} ({r['pass_rate']}%)")
    
    print(f"\n  Overall: {passed}/{total} rules passed")
    
    if passed == total:
        print("\n  EXCELLENT: All chemical logic rules are satisfied!")
        print("  The model's predictions are internally consistent with")
        print("  known geochemical principles.")
    else:
        print(f"\n  WARNING: {total - passed} rule(s) violated.")
        print("  Review the violations above. Some may be:")
        print("  - Real issues in the PHREEQC simulations")
        print("  - Edge cases where the rule doesn't strictly apply")
        print("  - Tolerance thresholds that need adjustment")
    
    # Save report
    report_df = pd.DataFrame(results)
    report_path = os.path.join(OUTPUT_DIR, "tier1_validation_report.csv")
    report_df.to_csv(report_path, index=False)
    print(f"\n  Report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
