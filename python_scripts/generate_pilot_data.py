#!/usr/bin/env python3
"""
Generate pilot training data for chelator ML model
Creates ~120 PHREEQC scenarios, runs them, and parses results
"""

import os
import subprocess
import itertools
import pandas as pd
import re

# Define parameter ranges
ph_values = [5.5, 6.5, 7.5]
pb_concentrations = [50, 150]  # mg/L
chelators = {
    'None': 0,
    'EDTA': [25, 100],  # mg/L
    'Citrate': [25, 100]  # mg/L
}
textures = {
    'Loam': 0.5,
    'Clay': 1.5
}
moisture_levels = {
    'Mesic': 8,
    'Wet': 3
}

# Conversion factors
PB_MW = 207.2  # g/mol
EDTA_MW = 292.24  # g/mol (as disodium salt)
CITRATE_MW = 189.1  # g/mol (citric acid)

def mg_to_mol(mg_per_L, mw):
    """Convert mg/L to mol/L"""
    return (mg_per_L / 1000) / mw

def create_phreeqc_input(scenario_id, ph, pb_mg, chelator, dose_mg, texture_name, texture_val, moisture_name, pe):
    """Create a PHREEQC input file for one scenario"""
    
    # Convert concentrations
    pb_mol = mg_to_mol(pb_mg, PB_MW)
    
    # Chelator setup
    if chelator == 'None':
        chelator_line = ""
        chelator_mol = 0
    elif chelator == 'EDTA':
        chelator_mol = mg_to_mol(dose_mg, EDTA_MW)
        chelator_line = f"    Edta      {chelator_mol:.4e}  # {dose_mg} mg/L EDTA"
    elif chelator == 'Citrate':
        chelator_mol = mg_to_mol(dose_mg, CITRATE_MW)
        chelator_line = f"    Citrate   {chelator_mol:.4e}  # {dose_mg} mg/L Citrate"
    
    input_text = f"""TITLE Scenario {scenario_id}: pH={ph}, Pb={pb_mg}mg/L, {chelator}={dose_mg}mg/L, {texture_name}, {moisture_name}
SOLUTION 1
    temp      25
    pH        {ph}
    pe        {pe}
    units     mol/L
    Pb        {pb_mol:.4e}
    Ca        1.25e-3
    Mg        8.23e-4
    Na        4.35e-3
    Cl        4.23e-3
    C(4)      1.0e-3
{chelator_line}
    
SURFACE 1
    Hfo_wOH   {texture_val}  600  0.09
    -equil 1
    
END
"""
    
    filename = f"../phreeqc_inputs/scenario_{scenario_id:03d}.phr"
    with open(filename, 'w') as f:
        f.write(input_text)
    
    return filename, {
        'scenario_id': scenario_id,
        'ph': ph,
        'pb_mg_L': pb_mg,
        'chelator': chelator,
        'dose_mg_L': dose_mg,
        'texture': texture_name,
        'moisture': moisture_name,
        'pe': pe,
        'hfo_sites': texture_val
    }

def run_phreeqc(input_file, output_file):
    """Run PHREEQC on an input file"""
    database = "/usr/local/share/phreeqc_databases/minteq.v4.dat"
    cmd = ['phreeqc', input_file, output_file, database]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {input_file}: {e}")
        return False

def parse_phreeqc_output(output_file):
    """Extract key results from PHREEQC output"""
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        # Extract free Pb+2
        pb_match = re.search(r'Pb\s+(\d\.\d+e[+-]\d+).*?Pb\+2\s+(\d\.\d+e[+-]\d+)', content, re.DOTALL)
        
        if pb_match:
            total_pb = float(pb_match.group(1))
            free_pb = float(pb_match.group(2))
            percent_free = (free_pb / total_pb) * 100
            
            # Extract sorbed Pb
            sorbed_match = re.search(r'Hfo_wOPb\+\s+(\d\.\d+e[+-]\d+)', content)
            sorbed_pb = float(sorbed_match.group(1)) if sorbed_match else 0
            
            return {
                'total_pb_mol': total_pb,
                'free_pb_mol': free_pb,
                'percent_free': percent_free,
                'sorbed_pb_mol': sorbed_pb
            }
    except Exception as e:
        print(f"Error parsing {output_file}: {e}")
    
    return None

# Generate all scenarios
print("Generating scenarios...")
scenarios = []
scenario_id = 1

for ph, pb_mg, texture_name, moisture_name in itertools.product(
    ph_values, pb_concentrations, textures.keys(), moisture_levels.keys()
):
    texture_val = textures[texture_name]
    pe = moisture_levels[moisture_name]
    
    # No chelator scenario
    _, params = create_phreeqc_input(
        scenario_id, ph, pb_mg, 'None', 0, 
        texture_name, texture_val, moisture_name, pe
    )
    scenarios.append(params)
    scenario_id += 1
    
    # EDTA scenarios
    for dose in chelators['EDTA']:
        _, params = create_phreeqc_input(
            scenario_id, ph, pb_mg, 'EDTA', dose,
            texture_name, texture_val, moisture_name, pe
        )
        scenarios.append(params)
        scenario_id += 1
    
    # Citrate scenarios
    for dose in chelators['Citrate']:
        _, params = create_phreeqc_input(
            scenario_id, ph, pb_mg, 'Citrate', dose,
            texture_name, texture_val, moisture_name, pe
        )
        scenarios.append(params)
        scenario_id += 1

print(f"Created {len(scenarios)} scenarios")

# Run PHREEQC on all scenarios
print("\nRunning PHREEQC simulations...")
results = []

for i, params in enumerate(scenarios, 1):
    input_file = f"../phreeqc_inputs/scenario_{params['scenario_id']:03d}.phr"
    output_file = f"../phreeqc_outputs/scenario_{params['scenario_id']:03d}.txt"
    
    if run_phreeqc(input_file, output_file):
        parsed = parse_phreeqc_output(output_file)
        if parsed:
            result = {**params, **parsed}
            results.append(result)
            
            if i % 10 == 0:
                print(f"  Completed {i}/{len(scenarios)} scenarios")
    else:
        print(f"  Failed: scenario {params['scenario_id']}")

# Save to CSV
print("\nSaving results to CSV...")
df = pd.DataFrame(results)
df.to_csv('../data/pilot_training_data.csv', index=False)

print(f"\n✓ Complete! Generated {len(results)} training examples")
print(f"✓ Saved to: ../data/pilot_training_data.csv")
print(f"\nPreview:")
print(df.head(10))
print(f"\nSummary statistics:")
print(df[['percent_free']].describe())
