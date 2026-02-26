#!/usr/bin/env python3
"""
Generate full-scale training data with multiple metals competing
~1000+ PHREEQC scenarios
"""

import os
import subprocess
import itertools
import pandas as pd
import re
import time

# Define parameter ranges
ph_values = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]  # 6 values
metal_levels = {
    'Low': {'Pb': 25, 'Cu': 20, 'Zn': 30},      # mg/L
    'Medium': {'Pb': 100, 'Cu': 80, 'Zn': 120},
    'High': {'Pb': 300, 'Cu': 250, 'Zn': 400}
}
chelators = {
    'None': [0],
    'EDTA': [10, 50, 150],     # mg/L - wider range
    'Citrate': [10, 50, 150]
}
textures = {
    'Sand': 0.1,
    'Loam': 0.5,
    'Clay': 1.5
}
moisture_levels = {
    'Dry': 12,
    'Mesic': 8,
    'Wet': 3
}

# Molecular weights
MW = {
    'Pb': 207.2,
    'Cu': 63.55,
    'Zn': 65.38,
    'EDTA': 292.24,
    'Citrate': 189.1
}

def mg_to_mol(mg_per_L, element):
    """Convert mg/L to mol/L"""
    return (mg_per_L / 1000) / MW[element]

def create_phreeqc_input(scenario_id, ph, metal_level_name, chelator, dose_mg, 
                         texture_name, texture_val, moisture_name, pe):
    """Create a PHREEQC input file for one scenario with multiple metals"""
    
    # Get metal concentrations for this level
    metals = metal_levels[metal_level_name]
    pb_mol = mg_to_mol(metals['Pb'], 'Pb')
    cu_mol = mg_to_mol(metals['Cu'], 'Cu')
    zn_mol = mg_to_mol(metals['Zn'], 'Zn')
    
    # Chelator setup
    if chelator == 'None':
        chelator_line = ""
        chelator_mol = 0
    elif chelator == 'EDTA':
        chelator_mol = mg_to_mol(dose_mg, 'EDTA')
        chelator_line = f"    Edta      {chelator_mol:.4e}  # {dose_mg} mg/L"
    elif chelator == 'Citrate':
        chelator_mol = mg_to_mol(dose_mg, 'Citrate')
        chelator_line = f"    Citrate   {chelator_mol:.4e}  # {dose_mg} mg/L"
    
    input_text = f"""TITLE Scenario {scenario_id}: pH={ph}, Metals={metal_level_name}, {chelator}={dose_mg}mg/L, {texture_name}, {moisture_name}
SOLUTION 1  Multi-metal contaminated soil
    temp      25
    pH        {ph}
    pe        {pe}
    units     mol/L
    Pb        {pb_mol:.4e}  # {metals['Pb']} mg/L
    Cu        {cu_mol:.4e}  # {metals['Cu']} mg/L
    Zn        {zn_mol:.4e}  # {metals['Zn']} mg/L
    Ca        1.25e-3
    Mg        8.23e-4
    Na        4.35e-3
    Cl        4.23e-3
    C(4)      1.0e-3
{chelator_line}
    
SURFACE 1  Soil surface sites
    Hfo_wOH   {texture_val}  600  0.09
    -equil 1
    
END
"""
    
    filename = f"../phreeqc_inputs/full_scenario_{scenario_id:04d}.phr"
    with open(filename, 'w') as f:
        f.write(input_text)
    
    return filename, {
        'scenario_id': scenario_id,
        'ph': ph,
        'metal_level': metal_level_name,
        'pb_mg_L': metals['Pb'],
        'cu_mg_L': metals['Cu'],
        'zn_mg_L': metals['Zn'],
        'chelator': chelator,
        'dose_mg_L': dose_mg,
        'texture': texture_name,
        'texture_sites': texture_val,
        'moisture': moisture_name,
        'pe': pe
    }

def run_phreeqc(input_file, output_file):
    """Run PHREEQC on an input file"""
    database = "/usr/local/share/phreeqc_databases/minteq.v4.dat"
    cmd = ['phreeqc', input_file, output_file, database]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error running {input_file}: {e}")
        return False

def parse_phreeqc_output(output_file):
    """Extract results for all metals from PHREEQC output"""
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        results = {}
        
        # Extract for each metal
        for metal in ['Pb', 'Cu', 'Zn']:
            # Find total and free metal
            pattern = rf'{metal}\s+(\d\.\d+e[+-]\d+).*?{metal}\+2\s+(\d\.\d+e[+-]\d+)'
            metal_match = re.search(pattern, content, re.DOTALL)
            
            if metal_match:
                total = float(metal_match.group(1))
                free = float(metal_match.group(2))
                percent_free = (free / total) * 100 if total > 0 else 0
                
                results[f'{metal.lower()}_total_mol'] = total
                results[f'{metal.lower()}_free_mol'] = free
                results[f'{metal.lower()}_percent_free'] = percent_free
            
            # Extract sorbed metal
            sorbed_match = re.search(rf'Hfo_wO{metal}\+\s+(\d\.\d+e[+-]\d+)', content)
            if sorbed_match:
                results[f'{metal.lower()}_sorbed_mol'] = float(sorbed_match.group(1))
            else:
                results[f'{metal.lower()}_sorbed_mol'] = 0
        
        return results if results else None
        
    except Exception as e:
        print(f"Error parsing {output_file}: {e}")
        return None

# Generate all scenarios
print("=" * 70)
print("GENERATING FULL-SCALE TRAINING DATA")
print("=" * 70)
print("\nParameter ranges:")
print(f"  pH values: {ph_values}")
print(f"  Metal levels: {list(metal_levels.keys())}")
print(f"  Chelators: {list(chelators.keys())}")
print(f"  Textures: {list(textures.keys())}")
print(f"  Moisture: {list(moisture_levels.keys())}")

scenarios = []
scenario_id = 1

for ph, metal_level, texture_name, moisture_name in itertools.product(
    ph_values, metal_levels.keys(), textures.keys(), moisture_levels.keys()
):
    texture_val = textures[texture_name]
    pe = moisture_levels[moisture_name]
    
    # For each combination of environmental conditions, test all chelator scenarios
    for chelator, doses in chelators.items():
        for dose in doses:
            _, params = create_phreeqc_input(
                scenario_id, ph, metal_level, chelator, dose,
                texture_name, texture_val, moisture_name, pe
            )
            scenarios.append(params)
            scenario_id += 1

print(f"\n✓ Created {len(scenarios)} scenario input files")

# Run PHREEQC on all scenarios
print(f"\n{'Running PHREEQC simulations':-^70}")
results = []
start_time = time.time()

for i, params in enumerate(scenarios, 1):
    input_file = f"../phreeqc_inputs/full_scenario_{params['scenario_id']:04d}.phr"
    output_file = f"../phreeqc_outputs/full_scenario_{params['scenario_id']:04d}.txt"
    
    if run_phreeqc(input_file, output_file):
        parsed = parse_phreeqc_output(output_file)
        if parsed:
            result = {**params, **parsed}
            results.append(result)
            
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(scenarios) - i) / rate
                print(f"  Progress: {i}/{len(scenarios)} ({i/len(scenarios)*100:.1f}%) - "
                      f"Est. {remaining/60:.1f} min remaining")
    else:
        print(f"  ✗ Failed: scenario {params['scenario_id']}")

# Save to CSV
print(f"\n{'Saving results':-^70}")
df = pd.DataFrame(results)
output_csv = '../data/full_training_data.csv'
df.to_csv(output_csv, index=False)

elapsed_total = time.time() - start_time

print(f"\n{'='*70}")
print(f"✓ COMPLETE!")
print(f"✓ Generated {len(results)}/{len(scenarios)} training examples")
print(f"✓ Total time: {elapsed_total/60:.1f} minutes")
print(f"✓ Saved to: {output_csv}")
print(f"{'='*70}")

# Quick preview
print(f"\nPreview of data:")
print(df.head(10)[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 
                   'pb_percent_free', 'cu_percent_free', 'zn_percent_free']])

print(f"\nSummary statistics (% Free Metal):")
print(df[['pb_percent_free', 'cu_percent_free', 'zn_percent_free']].describe())
