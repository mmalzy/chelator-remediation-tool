#!/usr/bin/env python3
"""
Generate training data with DTPA (manually defined)
DTPA is superior for Zn chelation
"""

import os
import subprocess
import itertools
import pandas as pd
import re
import time

# Define parameter ranges
ph_values = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]
metal_levels = {
    'Low': {'Pb': 25, 'Cu': 20, 'Zn': 30},
    'Medium': {'Pb': 100, 'Cu': 80, 'Zn': 120},
    'High': {'Pb': 300, 'Cu': 250, 'Zn': 400}
}
chelators = {
    'None': [0],
    'EDTA': [50, 200],        # Fewer doses, focus on low/high
    'Citrate': [50, 200],
    'DTPA': [50, 200]         # NEW! Should excel at Zn
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
    'Citrate': 189.1,
    'DTPA': 393.35  # Diethylenetriaminepentaacetic acid
}

def mg_to_mol(mg_per_L, element):
    return (mg_per_L / 1000) / MW[element]

def create_phreeqc_input(scenario_id, ph, metal_level_name, chelator, dose_mg,
                         texture_name, texture_val, moisture_name, pe):
    
    metals = metal_levels[metal_level_name]
    pb_mol = mg_to_mol(metals['Pb'], 'Pb')
    cu_mol = mg_to_mol(metals['Cu'], 'Cu')
    zn_mol = mg_to_mol(metals['Zn'], 'Zn')
    
    # Chelator setup
    chelator_definitions = ""
    chelator_line = ""
    
    if chelator == 'None':
        pass
    elif chelator == 'EDTA':
        chelator_mol = mg_to_mol(dose_mg, 'EDTA')
        chelator_line = f"    Edta      {chelator_mol:.4e}"
    elif chelator == 'Citrate':
        chelator_mol = mg_to_mol(dose_mg, 'Citrate')
        chelator_line = f"    Citrate   {chelator_mol:.4e}"
    elif chelator == 'DTPA':
        chelator_mol = mg_to_mol(dose_mg, 'DTPA')
        # Define DTPA species manually
        chelator_definitions = """
# DTPA (Diethylenetriaminepentaacetic acid) definitions
SOLUTION_MASTER_SPECIES
Dtpa    Dtpa-5    0    393.35    393.35

SOLUTION_SPECIES
# Base species
Dtpa-5 = Dtpa-5
    log_k   0.0

# Protonation
H+ + Dtpa-5 = HDtpa-4
    log_k   10.5
2H+ + Dtpa-5 = H2Dtpa-3
    log_k   18.8
3H+ + Dtpa-5 = H3Dtpa-2
    log_k   24.2
4H+ + Dtpa-5 = H4Dtpa-
    log_k   28.2
5H+ + Dtpa-5 = H5Dtpa
    log_k   30.5

# Lead complexes
Pb+2 + Dtpa-5 = PbDtpa-3
    log_k   18.8
Pb+2 + Dtpa-5 + H+ = PbHDtpa-2
    log_k   21.0

# Copper complexes
Cu+2 + Dtpa-5 = CuDtpa-3
    log_k   21.4
Cu+2 + Dtpa-5 + H+ = CuHDtpa-2
    log_k   24.0

# Zinc complexes (STRONG!)
Zn+2 + Dtpa-5 = ZnDtpa-3
    log_k   18.6
Zn+2 + Dtpa-5 + H+ = ZnHDtpa-2
    log_k   21.2

# Calcium competition
Ca+2 + Dtpa-5 = CaDtpa-3
    log_k   10.7

# Magnesium competition
Mg+2 + Dtpa-5 = MgDtpa-3
    log_k   9.3
"""
        chelator_line = f"    Dtpa      {chelator_mol:.4e}  # {dose_mg} mg/L DTPA"
    
    input_text = f"""TITLE Scenario {scenario_id}: pH={ph}, Metals={metal_level_name}, {chelator}={dose_mg}mg/L, {texture_name}, {moisture_name}
{chelator_definitions}
SOLUTION 1  Multi-metal contaminated soil
    temp      25
    pH        {ph}
    pe        {pe}
    units     mol/L
    Pb        {pb_mol:.4e}
    Cu        {cu_mol:.4e}
    Zn        {zn_mol:.4e}
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
    
    filename = f"../phreeqc_inputs/dtpa_scenario_{scenario_id:04d}.phr"
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
    database = "/usr/local/share/phreeqc_databases/minteq.v4.dat"
    cmd = ['phreeqc', input_file, output_file, database]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return True
    except Exception as e:
        return False

def parse_phreeqc_output(output_file):
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        results = {}
        
        for metal in ['Pb', 'Cu', 'Zn']:
            pattern = rf'{metal}\s+(\d\.\d+e[+-]\d+).*?{metal}\+2\s+(\d\.\d+e[+-]\d+)'
            metal_match = re.search(pattern, content, re.DOTALL)
            
            if metal_match:
                total = float(metal_match.group(1))
                free = float(metal_match.group(2))
                percent_free = (free / total) * 100 if total > 0 else 0
                
                results[f'{metal.lower()}_total_mol'] = total
                results[f'{metal.lower()}_free_mol'] = free
                results[f'{metal.lower()}_percent_free'] = percent_free
            
            sorbed_match = re.search(rf'Hfo_wO{metal}\+\s+(\d\.\d+e[+-]\d+)', content)
            if sorbed_match:
                results[f'{metal.lower()}_sorbed_mol'] = float(sorbed_match.group(1))
            else:
                results[f'{metal.lower()}_sorbed_mol'] = 0
        
        return results if results else None
        
    except Exception as e:
        return None

# Generate scenarios
print("=" * 70)
print("GENERATING TRAINING DATA WITH DTPA")
print("=" * 70)

scenarios = []
scenario_id = 1

for ph, metal_level, texture_name, moisture_name in itertools.product(
    ph_values, metal_levels.keys(), textures.keys(), moisture_levels.keys()
):
    texture_val = textures[texture_name]
    pe = moisture_levels[moisture_name]
    
    for chelator, doses in chelators.items():
        for dose in doses:
            _, params = create_phreeqc_input(
                scenario_id, ph, metal_level, chelator, dose,
                texture_name, texture_val, moisture_name, pe
            )
            scenarios.append(params)
            scenario_id += 1

print(f"✓ Created {len(scenarios)} scenarios")

# Run simulations
print(f"\nRunning PHREEQC...")
results = []
start_time = time.time()

for i, params in enumerate(scenarios, 1):
    input_file = f"../phreeqc_inputs/dtpa_scenario_{params['scenario_id']:04d}.phr"
    output_file = f"../phreeqc_outputs/dtpa_scenario_{params['scenario_id']:04d}.txt"
    
    if run_phreeqc(input_file, output_file):
        parsed = parse_phreeqc_output(output_file)
        if parsed:
            results.append({**params, **parsed})
            
            if i % 50 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                remaining = (len(scenarios) - i) / rate
                print(f"  {i}/{len(scenarios)} ({i/len(scenarios)*100:.1f}%) - "
                      f"~{remaining/60:.1f} min left")

# Save
df = pd.DataFrame(results)
df.to_csv('../data/dtpa_training_data.csv', index=False)

print(f"\n{'='*70}")
print(f"✓ Complete! {len(results)} examples in {(time.time()-start_time)/60:.1f} min")
print(f"✓ Saved to: ../data/dtpa_training_data.csv")

print(f"\nZn chelation comparison (mean % free at 200 mg/L):")
high_dose = df[df['dose_mg_L'] == 200].groupby('chelator')['zn_percent_free'].mean()
print(high_dose.sort_values())

print(f"\nBest Zn scenario:")
best_zn = df.nsmallest(1, 'zn_percent_free')[['chelator', 'dose_mg_L', 'ph', 'zn_percent_free']]
print(best_zn.to_string(index=False))
