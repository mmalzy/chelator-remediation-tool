#!/usr/bin/env python3
"""
Supplemental script - generates ONLY Dry moisture scenarios
to complete the full 12,150 scenario dataset
"""

import os
import subprocess
import itertools
import pandas as pd
import re
import time
from datetime import datetime

print("=" * 80)
print("SUPPLEMENTAL: DRY MOISTURE SCENARIOS")
print("Generating missing 4,050 dry scenarios")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Same parameters as main script
params = {
    'ph': [5.5, 6.0, 6.5, 7.0, 7.5],
    
    'metal_levels': {
        'Low': {'Pb': 25, 'Cu': 20, 'Zn': 30, 'Cd': 2},
        'Medium': {'Pb': 100, 'Cu': 80, 'Zn': 120, 'Cd': 8},
        'High': {'Pb': 300, 'Cu': 250, 'Zn': 400, 'Cd': 25}
    },
    
    'textures': {
        'Sand': {'hfo': 0.1, 'doc': 10},
        'Loam': {'hfo': 0.5, 'doc': 25},
        'Clay': {'hfo': 1.5, 'doc': 40}
    },
    
    'ca_mg_levels': {
        'Low': {'Ca': 20, 'Mg': 10},
        'High': {'Ca': 100, 'Mg': 50}
    },
    
    'ionic_strength': {
        'Low': {'Na': 100, 'Cl': 150},
        'Medium': {'Na': 500, 'Cl': 700},
        'High': {'Na': 2000, 'Cl': 3000}
    },
    
    'chelators': {
        'EDTA': [50, 150, 300],
        'NTA': [50, 150, 300],
        'Citrate': [50, 150, 300],
        'Humic': [50, 150, 300],
        'Fulvic': [50, 150, 300]
    },
    
    # ONLY DRY THIS TIME
    'moisture': {
        'Dry': 12
    }
}

MW = {
    'Pb': 207.2, 'Cu': 63.55, 'Zn': 65.38, 'Cd': 112.41,
    'Ca': 40.08, 'Mg': 24.31, 'Na': 22.99, 'Cl': 35.45,
    'EDTA': 292.24, 'NTA': 191.14, 'Citrate': 189.1,
    'Humic': 1000, 'Fulvic': 800
}

def mg_to_mol(mg_per_L, element):
    return (mg_per_L / 1000) / MW[element]

def create_phreeqc_input(scenario_id, ph, metal_level, texture, ca_mg_level,
                         ionic_level, chelator, dose, moisture):
    
    metals = params['metal_levels'][metal_level]
    texture_params = params['textures'][texture]
    ca_mg = params['ca_mg_levels'][ca_mg_level]
    ionic = params['ionic_strength'][ionic_level]
    pe = params['moisture'][moisture]
    
    pb_mol = mg_to_mol(metals['Pb'], 'Pb')
    cu_mol = mg_to_mol(metals['Cu'], 'Cu')
    zn_mol = mg_to_mol(metals['Zn'], 'Zn')
    cd_mol = mg_to_mol(metals['Cd'], 'Cd')
    ca_mol = mg_to_mol(ca_mg['Ca'], 'Ca')
    mg_mol = mg_to_mol(ca_mg['Mg'], 'Mg')
    na_mol = mg_to_mol(ionic['Na'], 'Na')
    cl_mol = mg_to_mol(ionic['Cl'], 'Cl')
    doc_mol = mg_to_mol(texture_params['doc'], 'Pb')
    
    if chelator == 'EDTA':
        chelator_mol = mg_to_mol(dose, 'EDTA')
        chelator_line = f"    Edta      {chelator_mol:.4e}  # {dose} mg/L EDTA"
    elif chelator == 'NTA':
        chelator_mol = mg_to_mol(dose, 'NTA')
        chelator_line = f"    Nta       {chelator_mol:.4e}  # {dose} mg/L NTA"
    elif chelator == 'Citrate':
        chelator_mol = mg_to_mol(dose, 'Citrate')
        chelator_line = f"    Citrate   {chelator_mol:.4e}  # {dose} mg/L Citrate"
    elif chelator == 'Humic':
        chelator_mol = mg_to_mol(dose, 'Humic')
        doc_mol += chelator_mol
        chelator_line = f"    # Humic acid: {dose} mg/L (as additional DOC)"
    elif chelator == 'Fulvic':
        chelator_mol = mg_to_mol(dose, 'Fulvic')
        doc_mol += chelator_mol * 0.8
        chelator_line = f"    # Fulvic acid: {dose} mg/L (as additional DOC)"
    
    input_text = f"""TITLE Dry Scenario {scenario_id}: pH={ph}, {metal_level}, {chelator}={dose}mg/L, {texture}, Dry, Ionic={ionic_level}
SOLUTION 1  RI contaminated soil pore water - DRY CONDITIONS
    temp      25
    pH        {ph}
    pe        {pe}
    units     mol/L
    Pb        {pb_mol:.4e}
    Cu        {cu_mol:.4e}
    Zn        {zn_mol:.4e}
    Cd        {cd_mol:.4e}
    Ca        {ca_mol:.4e}
    Mg        {mg_mol:.4e}
    Na        {na_mol:.4e}
    Cl        {cl_mol:.4e}
    C(4)      {doc_mol:.4e}
{chelator_line}
    
SURFACE 1
    Hfo_wOH   {texture_params['hfo']}  600  0.09
    -equil 1
    
END
"""
    
    filename = f"../phreeqc_inputs/dry_{scenario_id:05d}.phr"
    with open(filename, 'w') as f:
        f.write(input_text)
    
    return {
        'scenario_id': scenario_id,
        'ph': ph,
        'metal_level': metal_level,
        'pb_mg_L': metals['Pb'],
        'cu_mg_L': metals['Cu'],
        'zn_mg_L': metals['Zn'],
        'cd_mg_L': metals['Cd'],
        'doc_mg_L': texture_params['doc'],
        'ca_mg_L': ca_mg['Ca'],
        'mg_mg_L': ca_mg['Mg'],
        'na_mg_L': ionic['Na'],
        'cl_mg_L': ionic['Cl'],
        'chelator': chelator,
        'dose_mg_L': dose,
        'texture': texture,
        'hfo_sites': texture_params['hfo'],
        'moisture': moisture,
        'pe': pe,
        'ca_mg_level': ca_mg_level,
        'ionic_level': ionic_level
    }

def run_phreeqc(scenario_id):
    input_file = f"../phreeqc_inputs/dry_{scenario_id:05d}.phr"
    output_file = f"../phreeqc_outputs/dry_{scenario_id:05d}.txt"
    database = "/usr/local/share/phreeqc_databases/minteq.v4.dat"
    cmd = ['phreeqc', input_file, output_file, database]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_file
    except Exception:
        return None

def parse_output(output_file):
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        results = {}
        for metal in ['Pb', 'Cu', 'Zn', 'Cd']:
            pattern = rf'{metal}\s+(\d\.\d+e[+-]\d+).*?{metal}\+2\s+(\d\.\d+e[+-]\d+)'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                total = float(match.group(1))
                free = float(match.group(2))
                percent_free = (free / total) * 100 if total > 0 else 0
                results[f'{metal.lower()}_total_mol'] = total
                results[f'{metal.lower()}_free_mol'] = free
                results[f'{metal.lower()}_percent_free'] = percent_free
            sorbed_match = re.search(rf'Hfo_wO{metal}\+\s+(\d\.\d+e[+-]\d+)', content)
            results[f'{metal.lower()}_sorbed_mol'] = float(sorbed_match.group(1)) if sorbed_match else 0
        return results
    except Exception:
        return None

# Generate scenarios
print("\nGenerating dry scenarios...")
scenarios = []
scenario_id = 1

for (ph, metal_level, texture, ca_mg_level, ionic_level) in itertools.product(
    params['ph'],
    params['metal_levels'].keys(),
    params['textures'].keys(),
    params['ca_mg_levels'].keys(),
    params['ionic_strength'].keys()
):
    for chelator, dose_list in params['chelators'].items():
        for dose in dose_list:
            scenario = create_phreeqc_input(
                scenario_id, ph, metal_level, texture, ca_mg_level,
                ionic_level, chelator, dose, 'Dry'
            )
            scenarios.append(scenario)
            scenario_id += 1

total = len(scenarios)
print(f"✓ Created {total:,} dry scenario input files")

# Run simulations
print(f"\nRunning PHREEQC simulations (~45-60 minutes)...")
print("-" * 80)

results = []
start_time = time.time()
failed = 0

for i, scenario in enumerate(scenarios, 1):
    output_file = run_phreeqc(scenario['scenario_id'])
    if output_file:
        parsed = parse_output(output_file)
        if parsed:
            results.append({**scenario, **parsed})
        else:
            failed += 1
    else:
        failed += 1
    
    if i % 100 == 0 or i == total:
        elapsed = time.time() - start_time
        rate = i / elapsed
        remaining = (total - i) / rate if rate > 0 else 0
        print(f"Progress: {i:5d}/{total} ({i/total*100:5.1f}%) | "
              f"Success: {len(results):5d} | Failed: {failed:3d} | "
              f"Time left: ~{remaining/60:.1f} min")

# Save dry scenarios
print("\nSaving dry scenarios...")
df_dry = pd.DataFrame(results)
df_dry.to_csv('../data/dry_scenarios.csv', index=False)

# Merge with existing data
print("Merging with existing dataset...")
df_existing = pd.read_csv('../data/RI_final_training_data.csv')
df_combined = pd.concat([df_existing, df_dry], ignore_index=True)
df_combined.to_csv('../data/complete_training_data.csv', index=False)

elapsed_total = time.time() - start_time

print(f"\n{'='*80}")
print(f"✓ Dry scenarios complete: {len(results):,} examples")
print(f"✓ Combined dataset: {len(df_combined):,} total scenarios")
print(f"✓ Time: {elapsed_total/60:.1f} minutes")
print(f"✓ Saved to: ../data/complete_training_data.csv")
print("=" * 80)

# Quick stats on dry vs mesic/wet
print("\nMoisture condition comparison (mean % free):")
print(df_combined.groupby('moisture')[['pb_percent_free', 'cu_percent_free',
                                        'zn_percent_free', 'cd_percent_free']].mean().round(1))
print("\n" + "=" * 80)
print(f"FINAL DATASET: {len(df_combined):,} scenarios ready for ML training!")
print("=" * 80)
