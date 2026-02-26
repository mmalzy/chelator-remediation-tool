#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TRAINING DATA - RHODE ISLAND COASTAL REMEDIATION
12,150 scenarios optimized for realistic RI conditions including high salinity
Metals: Pb, Cu, Zn, Cd
Chelators: EDTA, NTA, Citrate, Humic, Fulvic
"""

import os
import subprocess
import itertools
import pandas as pd
import re
import time
from datetime import datetime

print("=" * 80)
print("RHODE ISLAND COASTAL REMEDIATION - COMPREHENSIVE TRAINING DATA")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nOptimized for Rhode Island conditions:")
print("  ✓ Coastal salinity (high ionic strength)")
print("  ✓ Road salt impacts")
print("  ✓ Multiple heavy metals")
print("  ✓ 5 chelating agents")
print("=" * 80)

# Parameter definitions
params = {
    'ph': [5.5, 6.0, 6.5, 7.0, 7.5],  # Skip extreme pH values
    
    'metal_levels': {
        'Low': {'Pb': 25, 'Cu': 20, 'Zn': 30, 'Cd': 2},
        'Medium': {'Pb': 100, 'Cu': 80, 'Zn': 120, 'Cd': 8},
        'High': {'Pb': 300, 'Cu': 250, 'Zn': 400, 'Cd': 25}
    },
    
    'textures': {
        'Sand': {'hfo': 0.1, 'doc': 10},   # Low OM
        'Loam': {'hfo': 0.5, 'doc': 25},   # Medium OM
        'Clay': {'hfo': 1.5, 'doc': 40}    # High OM
    },
    
    'ca_mg_levels': {
        'Low': {'Ca': 20, 'Mg': 10},      # Soft water
        'High': {'Ca': 100, 'Mg': 50}     # Hard water
    },
    
    'ionic_strength': {
        'Low': {'Na': 100, 'Cl': 150},       # Inland freshwater
        'Medium': {'Na': 500, 'Cl': 700},    # Typical soil
        'High': {'Na': 2000, 'Cl': 3000}     # Coastal saline/road salt
    },
    
    'chelators': {
        'EDTA': [50, 150, 300],
        'NTA': [50, 150, 300],
        'Citrate': [50, 150, 300],
        'Humic': [50, 150, 300],
        'Fulvic': [50, 150, 300]
    },
    
    'moisture': {
        'Mesic': 8,   # Moderate moisture
        'Wet': 3      # Saturated/reducing
    }
}

# Molecular weights (g/mol)
MW = {
    'Pb': 207.2, 'Cu': 63.55, 'Zn': 65.38, 'Cd': 112.41,
    'Ca': 40.08, 'Mg': 24.31, 'Na': 22.99, 'Cl': 35.45,
    'EDTA': 292.24, 'NTA': 191.14, 'Citrate': 189.1, 
    'Humic': 1000, 'Fulvic': 800  # Approximate MW for organic acids
}

def mg_to_mol(mg_per_L, element):
    """Convert mg/L to mol/L"""
    return (mg_per_L / 1000) / MW[element]

def create_phreeqc_input(scenario_id, ph, metal_level, texture, ca_mg_level, 
                         ionic_level, chelator, dose, moisture):
    """Generate PHREEQC input file"""
    
    # Get parameter values
    metals = params['metal_levels'][metal_level]
    texture_params = params['textures'][texture]
    ca_mg = params['ca_mg_levels'][ca_mg_level]
    ionic = params['ionic_strength'][ionic_level]
    pe = params['moisture'][moisture]
    
    # Convert to mol/L
    pb_mol = mg_to_mol(metals['Pb'], 'Pb')
    cu_mol = mg_to_mol(metals['Cu'], 'Cu')
    zn_mol = mg_to_mol(metals['Zn'], 'Zn')
    cd_mol = mg_to_mol(metals['Cd'], 'Cd')
    ca_mol = mg_to_mol(ca_mg['Ca'], 'Ca')
    mg_mol = mg_to_mol(ca_mg['Mg'], 'Mg')
    na_mol = mg_to_mol(ionic['Na'], 'Na')
    cl_mol = mg_to_mol(ionic['Cl'], 'Cl')
    doc_mol = mg_to_mol(texture_params['doc'], 'Pb')  # Use as C(4) proxy
    
    # Chelator line
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
        # Humic acid - add as extra DOC
        chelator_mol = mg_to_mol(dose, 'Humic')
        doc_mol += chelator_mol
        chelator_line = f"    # Humic acid: {dose} mg/L (as additional DOC)"
    elif chelator == 'Fulvic':
        # Fulvic acid - add as extra DOC (smaller, more mobile than humic)
        chelator_mol = mg_to_mol(dose, 'Fulvic')
        doc_mol += chelator_mol * 0.8  # Fulvic has different binding capacity
        chelator_line = f"    # Fulvic acid: {dose} mg/L (as additional DOC)"
    
    input_text = f"""TITLE Scenario {scenario_id}: pH={ph}, {metal_level}, {chelator}={dose}mg/L, {texture}, {moisture}, Ionic={ionic_level}
SOLUTION 1  RI contaminated soil pore water
    temp      25
    pH        {ph}
    pe        {pe}
    units     mol/L
    # Heavy metals
    Pb        {pb_mol:.4e}  # {metals['Pb']} mg/L
    Cu        {cu_mol:.4e}  # {metals['Cu']} mg/L
    Zn        {zn_mol:.4e}  # {metals['Zn']} mg/L
    Cd        {cd_mol:.4e}  # {metals['Cd']} mg/L
    # Competing cations
    Ca        {ca_mol:.4e}  # {ca_mg['Ca']} mg/L
    Mg        {mg_mol:.4e}  # {ca_mg['Mg']} mg/L
    # Ionic strength (salinity)
    Na        {na_mol:.4e}  # {ionic['Na']} mg/L
    Cl        {cl_mol:.4e}  # {ionic['Cl']} mg/L
    # Dissolved organic carbon
    C(4)      {doc_mol:.4e}  # {texture_params['doc']} mg/L base DOC
{chelator_line}
    
SURFACE 1  Soil mineral surfaces (Fe/Al oxides)
    Hfo_wOH   {texture_params['hfo']}  600  0.09
    -equil 1
    
END
"""
    
    filename = f"../phreeqc_inputs/RI_final_{scenario_id:05d}.phr"
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
    """Run PHREEQC simulation"""
    input_file = f"../phreeqc_inputs/RI_final_{scenario_id:05d}.phr"
    output_file = f"../phreeqc_outputs/RI_final_{scenario_id:05d}.txt"
    database = "/usr/local/share/phreeqc_databases/minteq.v4.dat"
    
    cmd = ['phreeqc', input_file, output_file, database]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_file
    except Exception:
        return None

def parse_output(output_file):
    """Extract % free metal for each heavy metal"""
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
        
        results = {}
        
        for metal in ['Pb', 'Cu', 'Zn', 'Cd']:
            # Find total and free metal
            pattern = rf'{metal}\s+(\d\.\d+e[+-]\d+).*?{metal}\+2\s+(\d\.\d+e[+-]\d+)'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                total = float(match.group(1))
                free = float(match.group(2))
                percent_free = (free / total) * 100 if total > 0 else 0
                
                results[f'{metal.lower()}_total_mol'] = total
                results[f'{metal.lower()}_free_mol'] = free
                results[f'{metal.lower()}_percent_free'] = percent_free
            
            # Sorbed to surface
            sorbed_match = re.search(rf'Hfo_wO{metal}\+\s+(\d\.\d+e[+-]\d+)', content)
            results[f'{metal.lower()}_sorbed_mol'] = float(sorbed_match.group(1)) if sorbed_match else 0
        
        return results
    except Exception:
        return None

# Generate all scenarios
print("\nGenerating scenarios...")
scenarios = []
scenario_id = 1

for (ph, metal_level, texture, ca_mg_level, ionic_level, moisture) in itertools.product(
    params['ph'],
    params['metal_levels'].keys(),
    params['textures'].keys(),
    params['ca_mg_levels'].keys(),
    params['ionic_strength'].keys(),
    params['moisture'].keys()
):
    for chelator, dose_list in params['chelators'].items():
        for dose in dose_list:
            scenario = create_phreeqc_input(
                scenario_id, ph, metal_level, texture, ca_mg_level,
                ionic_level, chelator, dose, moisture
            )
            scenarios.append(scenario)
            scenario_id += 1

total_scenarios = len(scenarios)
print(f"✓ Created {total_scenarios:,} input files")
print(f"\nParameter breakdown:")
print(f"  pH levels: {len(params['ph'])}")
print(f"  Metal contamination levels: {len(params['metal_levels'])}")
print(f"  Soil textures: {len(params['textures'])}")
print(f"  Ca/Mg levels: {len(params['ca_mg_levels'])}")
print(f"  Ionic strength levels: {len(params['ionic_strength'])} (includes HIGH salinity)")
print(f"  Moisture conditions: {len(params['moisture'])}")
print(f"  Chelators: {len(params['chelators'])}")
print(f"  Doses per chelator: {len(params['chelators']['EDTA'])}")
print(f"  Total chelator scenarios: {len(params['chelators']) * len(params['chelators']['EDTA'])}")

# Run simulations
print(f"\n{'RUNNING PHREEQC SIMULATIONS':-^80}")
print(f"Estimated time: {total_scenarios * 0.5 / 60:.0f}-{total_scenarios * 1.2 / 60:.0f} minutes")
print(f"               ({total_scenarios * 0.5 / 3600:.1f}-{total_scenarios * 1.2 / 3600:.1f} hours)")
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
    
    # Progress updates every 100 scenarios
    if i % 100 == 0 or i == total_scenarios:
        elapsed = time.time() - start_time
        rate = i / elapsed
        remaining = (total_scenarios - i) / rate if rate > 0 else 0
        percent = i / total_scenarios * 100
        
        print(f"Progress: {i:6d}/{total_scenarios} ({percent:5.1f}%) | "
              f"Success: {len(results):6d} | Failed: {failed:4d} | "
              f"Time left: ~{remaining/60:5.1f} min ({remaining/3600:4.2f} hrs)")

# Save results
print("\n" + "=" * 80)
print("Saving results to CSV...")

df = pd.DataFrame(results)
output_csv = '../data/RI_final_training_data.csv'
df.to_csv(output_csv, index=False)

elapsed_total = time.time() - start_time

print(f"\n{'COMPLETE!':-^80}")
print(f"✓ Generated {len(results):,} training examples from {total_scenarios:,} scenarios")
print(f"✓ Success rate: {len(results)/total_scenarios*100:.1f}%")
print(f"✓ Total time: {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
print(f"✓ File size: {os.path.getsize(output_csv)/1024/1024:.1f} MB")
print(f"✓ Saved to: {output_csv}")
print(f"✓ Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Comprehensive summary statistics
print(f"\n{'DATA SUMMARY':-^80}")
print(f"Dataset dimensions: {len(df):,} rows × {len(df.columns)} columns")
print(f"\nInput features: {len([c for c in df.columns if not c.endswith('_mol') and not c.endswith('_free')])} columns")
print(f"Output targets: {len([c for c in df.columns if '_percent_free' in c])} metals")

print(f"\n{'Metal % Free Statistics':-^80}")
for metal in ['pb', 'cu', 'zn', 'cd']:
    col = f'{metal}_percent_free'
    stats = df[col].describe()
    print(f"{metal.upper():3s}: mean={stats['mean']:6.2f}%  std={stats['std']:6.2f}%  "
          f"min={stats['min']:6.2f}%  max={stats['max']:6.2f}%")

print(f"\n{'Effect of Ionic Strength (Salinity)':-^80}")
ionic_effect = df.groupby('ionic_level')[['pb_percent_free', 'cu_percent_free', 
                                           'zn_percent_free', 'cd_percent_free']].mean()
print(ionic_effect.round(1))

print(f"\n{'Best Chelator Performance (mean % free across all conditions)':-^80}")
chelator_perf = df.groupby('chelator')[['pb_percent_free', 'cu_percent_free', 
                                         'zn_percent_free', 'cd_percent_free']].mean()
print(chelator_perf.round(1).sort_values('pb_percent_free'))

print(f"\n{'Best Overall Scenario (lowest total % free)':-^80}")
df['total_free'] = (df['pb_percent_free'] + df['cu_percent_free'] + 
                    df['zn_percent_free'] + df['cd_percent_free'])
best = df.nsmallest(1, 'total_free').iloc[0]
print(f"  Scenario ID: {best['scenario_id']}")
print(f"  pH: {best['ph']}")
print(f"  Chelator: {best['chelator']} @ {best['dose_mg_L']} mg/L")
print(f"  Texture: {best['texture']}")
print(f"  Ionic level: {best['ionic_level']}")
print(f"  Results:")
print(f"    Pb: {best['pb_percent_free']:.2f}% free")
print(f"    Cu: {best['cu_percent_free']:.2f}% free")
print(f"    Zn: {best['zn_percent_free']:.2f}% free")
print(f"    Cd: {best['cd_percent_free']:.2f}% free")
print(f"    Total: {best['total_free']:.2f}% free")

print("\n" + "=" * 80)
print("READY FOR MACHINE LEARNING!")
print("Dataset optimized for Rhode Island coastal remediation applications")
print("=" * 80)
