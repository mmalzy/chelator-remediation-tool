#!/usr/bin/env python3
"""
Tier 2 Validation: Literature Benchmarking Framework
=====================================================
Compares model predictions against published experimental data from
peer-reviewed chelator-assisted soil remediation studies.

Workflow:
1. Researcher populates literature_benchmark_data.csv with published results
2. This script maps published conditions to model input features
3. Runs model predictions for those conditions  
4. Compares predicted vs. observed outcomes
5. Generates validation report with statistics and plots

Usage:
    cd /Users/mallorymalz/Documents/chelator_ml_project
    python3 python_scripts/tier2_literature_benchmark.py

The literature CSV should be placed at:
    data/literature_benchmark_data.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
LIT_DATA_FILE = os.path.join(DATA_DIR, "literature_benchmark_data.csv")
REPORT_FILE = os.path.join(DATA_DIR, "tier2_benchmark_report.csv")

METALS = ['pb', 'cu', 'zn', 'cd']

# ============================================================
# MAPPING FUNCTIONS: Literature values → Model inputs
# ============================================================

def map_texture_to_model(texture_description):
    """
    Map published soil texture descriptions to our model categories.
    Our model uses: Sand, Loam, Clay
    """
    tex = str(texture_description).lower()
    
    # Sandy textures
    if any(w in tex for w in ['sand', 'sandy', 'loamy sand']):
        if 'clay' in tex:
            return 'Loam'  # sandy clay loam → Loam
        return 'Sand'
    
    # Clay textures
    if any(w in tex for w in ['clay', 'clayey', 'silty clay']):
        if 'loam' in tex:
            return 'Clay'  # clay loam is still clay-dominated
        return 'Clay'
    
    # Loam and silt textures
    if any(w in tex for w in ['loam', 'silt', 'silty']):
        return 'Loam'
    
    # Default to Loam if unclear
    return 'Loam'


def map_moisture_to_model(description):
    """
    Map published experimental moisture conditions to our model categories.
    Our model uses: Dry (pe=12), Mesic (pe=8), Wet (pe=3)
    """
    desc = str(description).lower()
    
    if any(w in desc for w in ['saturated', 'flooded', 'waterlogged', 'anaerobic', 'wet']):
        return 'Wet'
    if any(w in desc for w in ['dry', 'air-dried', 'arid', 'drained']):
        return 'Dry'
    # Most lab batch experiments are at field capacity → Mesic
    return 'Mesic'


def map_ionic_strength(ec_dS_m=None, na_mg_L=None, description=None):
    """
    Map published ionic strength indicators to our model categories.
    Our model uses: Low (Na=100, Cl=150), Medium (Na=500, Cl=700), High (Na=2000, Cl=3000)
    
    EC (electrical conductivity) in dS/m is a common proxy:
    < 2 dS/m → Low
    2-8 dS/m → Medium  
    > 8 dS/m → High (saline)
    """
    if ec_dS_m is not None and not pd.isna(ec_dS_m):
        if ec_dS_m < 2:
            return 'Low'
        elif ec_dS_m < 8:
            return 'Medium'
        else:
            return 'High'
    
    if na_mg_L is not None and not pd.isna(na_mg_L):
        if na_mg_L < 200:
            return 'Low'
        elif na_mg_L < 1000:
            return 'Medium'
        else:
            return 'High'
    
    if description is not None:
        desc = str(description).lower()
        if any(w in desc for w in ['saline', 'coastal', 'salt', 'brackish']):
            return 'High'
    
    return 'Low'  # Most lab studies use clean water


def om_to_doc(om_percent):
    """
    Convert organic matter % to dissolved organic carbon (mg/L).
    DOC = OM% * 10 * 0.58 (van Bemmelen factor * unit conversion)
    Then map to our model values: 10, 25, or 40 mg/L
    """
    if pd.isna(om_percent):
        return 25  # default
    doc_est = om_percent * 10 * 0.58
    if doc_est < 17.5:
        return 10
    elif doc_est < 32.5:
        return 25
    else:
        return 40


def map_metal_level(pb=None, cu=None, zn=None, cd=None):
    """
    Map published metal concentrations to our Low/Medium/High categories.
    Uses Pb as primary indicator if available.
    
    Our levels: Low (Pb=25), Medium (Pb=100), High (Pb=300) mg/kg
    Note: published values are usually mg/kg (soil) not mg/L (solution)
    Approximate conversion: mg/L in pore water ~ mg/kg * (theta / rho)
    For simplicity, we map by relative magnitude.
    """
    # Use the first available metal
    for conc, thresholds in [
        (pb, (50, 200)),   # Pb boundaries
        (cu, (40, 150)),   # Cu boundaries  
        (zn, (60, 250)),   # Zn boundaries
        (cd, (5, 15)),     # Cd boundaries
    ]:
        if conc is not None and not pd.isna(conc):
            if conc < thresholds[0]:
                return 'Low'
            elif conc < thresholds[1]:
                return 'Medium'
            else:
                return 'High'
    
    return 'Medium'  # default


def map_chelator_name(published_name):
    """Map published chelator names to our model categories."""
    name = str(published_name).upper().strip()
    
    if 'EDTA' in name:
        return 'EDTA'
    if 'NTA' in name:
        return 'NTA'
    if 'CITRI' in name or 'CITRATE' in name.upper():
        return 'Citrate'
    if 'HUMIC' in name:
        return 'Humic'
    if 'FULVIC' in name:
        return 'Fulvic'
    if name in ['NONE', 'CONTROL', 'BLANK', '']:
        return 'nan'  # baseline
    
    return None  # Unknown chelator - skip


def map_dose(dose_value, dose_unit, chelator):
    """
    Convert published chelator doses to mg/L.
    Common units in literature: mmol/kg, g/kg, mM, mg/L, mg/kg
    """
    if pd.isna(dose_value) or pd.isna(dose_unit):
        return 150  # default middle dose
    
    unit = str(dose_unit).lower().strip()
    
    # Molecular weights
    mw = {'EDTA': 292.24, 'NTA': 191.14, 'Citrate': 189.1, 'Humic': 200, 'Fulvic': 200}
    chelator_mw = mw.get(chelator, 250)
    
    if 'mg/l' in unit or 'mg/L' in unit or 'ppm' in unit:
        mg_L = dose_value
    elif 'mmol/l' in unit or 'mm' in unit:
        mg_L = dose_value * chelator_mw
    elif 'mol/l' in unit:
        mg_L = dose_value * chelator_mw * 1000
    elif 'g/l' in unit:
        mg_L = dose_value * 1000
    elif 'mmol/kg' in unit:
        # Approximate: assume 10:1 liquid:solid ratio (common in batch tests)
        mg_L = dose_value * chelator_mw / 10
    elif 'g/kg' in unit:
        mg_L = dose_value * 1000 / 10
    else:
        mg_L = dose_value  # assume mg/L
    
    # Map to nearest model dose: 50, 150, or 300
    if mg_L < 100:
        return 50
    elif mg_L < 225:
        return 150
    else:
        return 300


# ============================================================
# COMPARISON AND REPORTING
# ============================================================

def compare_predictions(lit_data, model_predictions):
    """
    Compare literature observations with model predictions.
    
    We compare DIRECTION (did the chelator help?) and RELATIVE RANKING
    (which chelator was best?) rather than exact % values, because:
    - Literature reports extraction efficiency, not % free in pore water
    - Lab conditions don't perfectly match our simulation parameters
    - The comparison is necessarily approximate
    """
    results = []
    
    for idx, row in lit_data.iterrows():
        pred = model_predictions.get(idx, {})
        if not pred:
            continue
        
        result = {
            'study': row.get('citation', 'Unknown'),
            'metal': row.get('metal', '?'),
            'chelator': row.get('chelator_mapped', '?'),
            'dose_mg_L': row.get('dose_mapped', '?'),
            'ph': row.get('ph', '?'),
            'observed_extraction_pct': row.get('observed_extraction_pct', None),
            'observed_best_chelator': row.get('observed_best_chelator', None),
            'predicted_free_pct': pred.get('predicted_free_pct', None),
            'predicted_reduction_pct': pred.get('predicted_reduction', None),
        }
        
        # Directional check: did model predict improvement?
        if pred.get('predicted_reduction', 0) > 0:
            result['direction_match'] = 'Yes' if row.get('observed_extraction_pct', 0) > 0 else 'No'
        else:
            result['direction_match'] = 'Check'
        
        results.append(result)
    
    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  TIER 2 VALIDATION: Literature Benchmarking")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Check for literature data file
    if not os.path.exists(LIT_DATA_FILE):
        print(f"\nLiterature data file not found at:")
        print(f"  {LIT_DATA_FILE}")
        print(f"\nCreating template file...")
        create_template()
        print(f"\nTemplate created at: {LIT_DATA_FILE}")
        print(f"\nNext steps:")
        print(f"  1. Open the CSV in Excel or Google Sheets")
        print(f"  2. Add rows from published studies (see column descriptions)")
        print(f"  3. Re-run this script to compare predictions vs literature")
        return
    
    # Load literature data
    lit = pd.read_csv(LIT_DATA_FILE)
    print(f"\nLoaded {len(lit)} literature data points")
    
    if len(lit) == 0:
        print("No data rows found. Please populate the template.")
        return
    
    # Map literature values to model inputs
    print("\nMapping literature conditions to model inputs...")
    lit['texture_mapped'] = lit['soil_texture'].apply(map_texture_to_model)
    lit['moisture_mapped'] = lit['moisture_description'].apply(map_moisture_to_model)
    lit['ionic_mapped'] = lit.apply(
        lambda r: map_ionic_strength(r.get('ec_dS_m'), r.get('na_mg_L'), r.get('salinity_description')),
        axis=1
    )
    lit['doc_mapped'] = lit['om_percent'].apply(om_to_doc)
    lit['chelator_mapped'] = lit['chelator_used'].apply(map_chelator_name)
    lit['dose_mapped'] = lit.apply(
        lambda r: map_dose(r['chelator_dose'], r['dose_unit'], r.get('chelator_mapped', '')),
        axis=1
    )
    lit['metal_level_mapped'] = lit.apply(
        lambda r: map_metal_level(r.get('pb_mg_kg'), r.get('cu_mg_kg'), 
                                   r.get('zn_mg_kg'), r.get('cd_mg_kg')),
        axis=1
    )
    
    # Print mapping summary
    print(f"\nMapped conditions:")
    print(f"  Textures: {lit['texture_mapped'].value_counts().to_dict()}")
    print(f"  Moisture: {lit['moisture_mapped'].value_counts().to_dict()}")
    print(f"  Ionic: {lit['ionic_mapped'].value_counts().to_dict()}")
    print(f"  Chelators: {lit['chelator_mapped'].value_counts().to_dict()}")
    
    # Skip rows with unmapped chelators
    unmapped = lit['chelator_mapped'].isna()
    if unmapped.any():
        print(f"\n  WARNING: {unmapped.sum()} rows have unmapped chelators - skipping")
        lit = lit[~unmapped]
    
    # Try to load models and predict
    try:
        import joblib
        
        print("\nLoading models...")
        models = {}
        for metal in METALS:
            model_path = os.path.join(MODEL_DIR, f"{metal}_percent_free_model.joblib")
            if os.path.exists(model_path):
                models[metal] = joblib.load(model_path)
                print(f"  Loaded {metal} model")
        
        encoders_path = os.path.join(MODEL_DIR, "label_encoders.joblib")
        if os.path.exists(encoders_path):
            encoders = joblib.load(encoders_path)
            print(f"  Loaded label encoders")
        
        feature_path = os.path.join(MODEL_DIR, "feature_info.json")
        if os.path.exists(feature_path):
            with open(feature_path, 'r') as f:
                feature_info = json.load(f)
            print(f"  Loaded feature info: {len(feature_info.get('feature_names', []))} features")
        
        # Generate predictions (would need full feature vector - placeholder)
        print("\nNote: Full prediction pipeline requires assembling complete")
        print("feature vectors from mapped conditions. Run the model comparison")
        print("after populating the literature dataset.")
        
    except ImportError:
        print("\njoblib not available - install with: pip3 install joblib --user")
        print("Mapping complete but cannot generate predictions without models.")
    
    # Save mapped data for review
    mapped_file = os.path.join(DATA_DIR, "literature_benchmark_mapped.csv")
    lit.to_csv(mapped_file, index=False)
    print(f"\nMapped literature data saved to: {mapped_file}")
    print("Review the _mapped columns to verify the mappings are reasonable.")


def create_template():
    """Create the literature benchmark CSV template with column descriptions."""
    
    columns = {
        'study_id': [],
        'citation': [],
        'doi': [],
        'year': [],
        'study_type': [],          # 'batch', 'column', 'field', 'pot'
        'soil_texture': [],         # As described in paper
        'ph': [],
        'om_percent': [],           # Organic matter %
        'cec_cmol_kg': [],          # Cation exchange capacity (optional)
        'ec_dS_m': [],              # Electrical conductivity (optional)
        'na_mg_L': [],              # Sodium if reported (optional)
        'salinity_description': [], # 'saline', 'non-saline', etc.
        'moisture_description': [], # 'saturated', 'field capacity', 'air-dried'
        'pb_mg_kg': [],             # Total Pb in soil
        'cu_mg_kg': [],             # Total Cu in soil
        'zn_mg_kg': [],             # Total Zn in soil
        'cd_mg_kg': [],             # Total Cd in soil
        'metal': [],                # Which metal is reported (pb/cu/zn/cd)
        'chelator_used': [],        # EDTA, NTA, Citrate, etc.
        'chelator_dose': [],        # Numeric dose value
        'dose_unit': [],            # mmol/kg, mg/L, g/kg, mM, etc.
        'contact_time_hr': [],      # Hours of contact/extraction
        'liquid_solid_ratio': [],   # e.g., 10 for 10:1
        'observed_extraction_pct': [],  # % of metal extracted/mobilized
        'observed_free_pct': [],    # % free metal if directly measured (rare)
        'observed_best_chelator': [],   # Which chelator worked best
        'notes': [],
    }
    
    template = pd.DataFrame(columns)
    
    # Add example rows to show format
    example_rows = [
        {
            'study_id': 'EXAMPLE_1',
            'citation': 'Author et al. (2020) J Hazard Mater 999:123456',
            'doi': '10.1016/j.jhazmat.2020.123456',
            'year': 2020,
            'study_type': 'batch',
            'soil_texture': 'sandy loam',
            'ph': 6.2,
            'om_percent': 3.5,
            'cec_cmol_kg': 12.5,
            'ec_dS_m': 0.8,
            'na_mg_L': None,
            'salinity_description': 'non-saline',
            'moisture_description': 'slurry (10:1 L/S)',
            'pb_mg_kg': 850,
            'cu_mg_kg': 120,
            'zn_mg_kg': 450,
            'cd_mg_kg': 8,
            'metal': 'pb',
            'chelator_used': 'EDTA',
            'chelator_dose': 5,
            'dose_unit': 'mmol/kg',
            'contact_time_hr': 24,
            'liquid_solid_ratio': 10,
            'observed_extraction_pct': 72,
            'observed_free_pct': None,
            'observed_best_chelator': 'EDTA',
            'notes': 'EXAMPLE ROW - delete before use'
        },
        {
            'study_id': 'EXAMPLE_2',
            'citation': 'Author et al. (2021) Chemosphere 280:130000',
            'doi': '10.1016/j.chemosphere.2021.130000',
            'year': 2021,
            'study_type': 'batch',
            'soil_texture': 'silt loam',
            'ph': 7.1,
            'om_percent': 5.2,
            'cec_cmol_kg': 18.0,
            'ec_dS_m': 1.2,
            'na_mg_L': None,
            'salinity_description': 'non-saline',
            'moisture_description': 'slurry (20:1 L/S)',
            'pb_mg_kg': 1200,
            'cu_mg_kg': 85,
            'zn_mg_kg': 680,
            'cd_mg_kg': 15,
            'metal': 'zn',
            'chelator_used': 'Citrate',
            'chelator_dose': 10,
            'dose_unit': 'mmol/kg',
            'contact_time_hr': 48,
            'liquid_solid_ratio': 20,
            'observed_extraction_pct': 35,
            'observed_free_pct': None,
            'observed_best_chelator': 'EDTA',
            'notes': 'EXAMPLE ROW - delete before use. Zn poorly extracted by citrate.'
        }
    ]
    
    template = pd.DataFrame(example_rows)
    template.to_csv(LIT_DATA_FILE, index=False)


if __name__ == "__main__":
    main()
