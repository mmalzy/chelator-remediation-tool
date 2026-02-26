#!/usr/bin/env python3
"""
generate_baseline_no_chelator.py
================================
Generates PHREEQC input files for NO-CHELATOR baseline scenarios,
runs them through PHREEQC, and parses outputs into a CSV.

These baselines use the EXACT same parameter combinations as the main
training data (ph, metal_level, texture, moisture, ionic_level, ca_mg_level)
but with NO chelating agent added. This allows the ML model to learn
the improvement each chelator provides over untreated conditions.

Parameter Space (matches existing training data):
- pH: 5.5, 6.5, 7.5 (3 levels)
- Metal level: Low, Medium, High (3 levels)
- Texture: Sand, Loam, Clay (3 levels)
- Moisture: Dry, Mesic, Wet (3 levels)
- Ionic level: Low, Medium, High (3 levels)
- Ca/Mg level: Low, High (2 levels)

Total: 3 × 3 × 3 × 3 × 3 × 2 = 486 baseline scenarios

Usage:
    python3 generate_baseline_no_chelator.py

Output:
    - PHREEQC input files in phreeqc_inputs/baseline_XXXXX.phr
    - PHREEQC output files in phreeqc_outputs/baseline_XXXXX.txt
    - CSV at data/baseline_no_chelator.csv
    
Author: Mallory Malz (with AI co-creator)
Project: Chelator ML Remediation - Rhode Island Coastal Soils
"""

import os
import subprocess
import csv
import re
import sys
from itertools import product

# ============================================================
# CONFIGURATION - Adjust paths if needed
# ============================================================
BASE_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
PHREEQC_EXE = "/usr/local/bin/phreeqc"
PHREEQC_DB = "/usr/local/share/phreeqc_databases/minteq.v4.dat"

INPUT_DIR = os.path.join(BASE_DIR, "phreeqc_inputs")
OUTPUT_DIR = os.path.join(BASE_DIR, "phreeqc_outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ============================================================
# PARAMETER DEFINITIONS (matches existing training data exactly)
# ============================================================

# pH levels
PH_VALUES = [5.5, 6.5, 7.5]

# Metal contamination levels (mg/L)
METAL_LEVELS = {
    "Low":    {"Pb": 25,  "Cu": 20,  "Zn": 30,  "Cd": 2},
    "Medium": {"Pb": 100, "Cu": 80,  "Zn": 120, "Cd": 8},
    "High":   {"Pb": 300, "Cu": 250, "Zn": 400, "Cd": 25},
}

# Soil texture → HFO sites (mol) and DOC (mg/L)
TEXTURE_MAP = {
    "Sand": {"hfo": 0.1, "doc": 10},
    "Loam": {"hfo": 0.5, "doc": 25},
    "Clay": {"hfo": 1.5, "doc": 40},
}

# Moisture → pe (redox proxy)
MOISTURE_MAP = {
    "Dry":   12,
    "Mesic": 8,
    "Wet":   3,
}

# Ionic strength levels (Na/Cl in mg/L)
IONIC_MAP = {
    "Low":    {"Na": 100,  "Cl": 150},
    "Medium": {"Na": 500,  "Cl": 700},
    "High":   {"Na": 2000, "Cl": 3000},
}

# Ca/Mg competition levels (mg/L)
CA_MG_MAP = {
    "Low":  {"Ca": 20,  "Mg": 10},
    "High": {"Ca": 100, "Mg": 50},
}

# Molecular weights for unit conversion (mg/L → mol/L)
MW = {
    "Pb": 207.2,
    "Cu": 63.546,
    "Zn": 65.38,
    "Cd": 112.411,
    "Ca": 40.078,
    "Mg": 24.305,
    "Na": 22.990,
    "Cl": 35.453,
    "C":  12.011,   # For DOC as C(4)
}


def mg_to_mol(mg_per_L, element):
    """Convert mg/L to mol/L using molecular weight."""
    return (mg_per_L / 1000.0) / MW[element]


def generate_phreeqc_input(params):
    """
    Generate a PHREEQC input file string for a no-chelator baseline scenario.
    
    params: dict with keys:
        ph, metal_level, pb_mg, cu_mg, zn_mg, cd_mg, doc_mg,
        ca_mg, mg_mg, na_mg, cl_mg, texture, hfo, moisture, pe
    """
    # Convert all concentrations to mol/L
    pb_mol = mg_to_mol(params["pb_mg"], "Pb")
    cu_mol = mg_to_mol(params["cu_mg"], "Cu")
    zn_mol = mg_to_mol(params["zn_mg"], "Zn")
    cd_mol = mg_to_mol(params["cd_mg"], "Cd")
    ca_mol = mg_to_mol(params["ca_mg"], "Ca")
    mg_mol = mg_to_mol(params["mg_mg"], "Mg")
    na_mol = mg_to_mol(params["na_mg"], "Na")
    cl_mol = mg_to_mol(params["cl_mg"], "Cl")
    doc_mol = mg_to_mol(params["doc_mg"], "C")  # DOC as C(4)

    title = (f"Baseline No Chelator | pH={params['ph']} "
             f"Metal={params['metal_level']} Texture={params['texture']} "
             f"Moisture={params['moisture']} Ionic={params['ionic_level']} "
             f"CaMg={params['ca_mg_level']}")

    lines = [
        f"TITLE {title}",
        "SOLUTION 1",
        "    temp      25",
        f"    pH        {params['ph']}",
        f"    pe        {params['pe']}",
        "    units     mol/L",
        f"    Pb        {pb_mol:.6e}",
        f"    Cu        {cu_mol:.6e}",
        f"    Zn        {zn_mol:.6e}",
        f"    Cd        {cd_mol:.6e}",
        f"    Ca        {ca_mol:.6e}",
        f"    Mg        {mg_mol:.6e}",
        f"    Na        {na_mol:.6e}",
        f"    Cl        {cl_mol:.6e}",
        f"    C(4)      {doc_mol:.6e}",
        "",
        "SURFACE 1",
        f"    Hfo_wOH   {params['hfo']}  600  0.09",
        "    -equil 1",
        "",
        "SELECTED_OUTPUT",
        "    -reset       false",
        "    -totals      Pb Cu Zn Cd",
        "    -molalities  Pb+2 Cu+2 Zn+2 Cd+2",
        "                 Hfo_wOPb+ Hfo_wOCu+ Hfo_wOZn+ Hfo_wOCd+",
        "",
        "END",
    ]
    return "\n".join(lines)


def parse_phreeqc_output(output_file, params):
    """
    Parse a PHREEQC output file to extract species molalities.
    Returns dict with percent_free and sorbed_mol for each metal.
    
    Uses the same parsing approach as the main training data generators:
    searches for species molalities in the output text.
    """
    try:
        with open(output_file, 'r', encoding='latin-1') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  WARNING: Output file not found: {output_file}")
        return None

    results = {}
    
    # For each metal, find total dissolved and free ion molality
    metals = {
        "pb": {"free_species": "Pb+2", "total_key": "Pb", "sorbed": "Hfo_wOPb+"},
        "cu": {"free_species": "Cu+2", "total_key": "Cu", "sorbed": "Hfo_wOCu+"},
        "zn": {"free_species": "Zn+2", "total_key": "Zn", "sorbed": "Hfo_wOZn+"},
        "cd": {"free_species": "Cd+2", "total_key": "Cd", "sorbed": "Hfo_wOCd+"},
    }

    for metal_key, species_info in metals.items():
        free_mol = None
        total_mol = None
        sorbed_mol = 0.0

        # --- Parse from SELECTED_OUTPUT or species distribution ---
        # Method 1: Look in "Distribution of species" section
        # Find free ion molality
        free_pattern = rf"^\s+{re.escape(species_info['free_species'])}\s+([\d.eE+-]+)\s+([\d.eE+-]+)"
        free_match = re.search(free_pattern, content, re.MULTILINE)
        if free_match:
            free_mol = float(free_match.group(1))  # molality column

        # Find total dissolved from "Solution composition"
        total_pattern = rf"^\s+{species_info['total_key']}\s+([\d.eE+-]+)"
        total_match = re.search(total_pattern, content, re.MULTILINE)
        if total_match:
            total_mol = float(total_match.group(1))

        # Find sorbed amount
        sorbed_pattern = rf"^\s+{re.escape(species_info['sorbed'])}\s+([\d.eE+-]+)"
        sorbed_match = re.search(sorbed_pattern, content, re.MULTILINE)
        if sorbed_match:
            sorbed_mol = float(sorbed_match.group(1))

        # Calculate percent free
        if free_mol is not None and total_mol is not None and total_mol > 0:
            pct_free = (free_mol / total_mol) * 100.0
            # Clamp to 0-100 range (floating point edge cases)
            pct_free = max(0.0, min(100.0, pct_free))
        else:
            pct_free = None
            print(f"  WARNING: Could not parse {metal_key} from {output_file}")

        results[f"{metal_key}_percent_free"] = pct_free
        results[f"{metal_key}_sorbed_mol"] = sorbed_mol

    return results


def run_phreeqc(input_file, output_file):
    """Run PHREEQC simulation. Returns True if successful."""
    try:
        result = subprocess.run(
            [PHREEQC_EXE, input_file, output_file, PHREEQC_DB],
            capture_output=True, text=True, timeout=60
        )
        # PHREEQC returns 0 on success
        if result.returncode != 0:
            print(f"  PHREEQC error (return code {result.returncode})")
            if result.stderr:
                print(f"  stderr: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  PHREEQC timeout on {input_file}")
        return False
    except Exception as e:
        print(f"  Error running PHREEQC: {e}")
        return False


def main():
    """Generate all baseline no-chelator scenarios."""
    
    # Ensure directories exist
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Build parameter combinations
    combinations = list(product(
        PH_VALUES,                      # 3
        METAL_LEVELS.keys(),            # 3
        TEXTURE_MAP.keys(),             # 3
        MOISTURE_MAP.keys(),            # 3
        IONIC_MAP.keys(),               # 3
        CA_MG_MAP.keys(),               # 2
    ))
    
    total = len(combinations)
    print(f"=" * 60)
    print(f"BASELINE NO-CHELATOR SCENARIO GENERATOR")
    print(f"=" * 60)
    print(f"Total scenarios to generate: {total}")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Data dir:   {DATA_DIR}")
    print(f"Database:   {PHREEQC_DB}")
    print(f"=" * 60)
    
    # Verify PHREEQC is accessible
    if not os.path.exists(PHREEQC_EXE):
        print(f"ERROR: PHREEQC not found at {PHREEQC_EXE}")
        sys.exit(1)
    if not os.path.exists(PHREEQC_DB):
        print(f"ERROR: Database not found at {PHREEQC_DB}")
        sys.exit(1)
    
    # CSV output
    csv_file = os.path.join(DATA_DIR, "baseline_no_chelator.csv")
    
    # Define CSV columns to match existing training data structure
    fieldnames = [
        "scenario_id", "ph", "metal_level",
        "pb_mg_L", "cu_mg_L", "zn_mg_L", "cd_mg_L",
        "doc_mg_L", "ca_mg_L", "mg_mg_L", "na_mg_L", "cl_mg_L",
        "chelator", "dose_mg_L",
        "texture", "hfo_sites", "moisture", "pe",
        "ca_mg_level", "ionic_level",
        # Targets
        "pb_percent_free", "cu_percent_free", "zn_percent_free", "cd_percent_free",
        "pb_sorbed_mol", "cu_sorbed_mol", "zn_sorbed_mol", "cd_sorbed_mol",
    ]
    
    success_count = 0
    fail_count = 0
    
    with open(csv_file, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, combo in enumerate(combinations):
            ph, metal_lvl, texture, moisture, ionic_lvl, ca_mg_lvl = combo
            scenario_num = i + 1
            scenario_id = f"baseline_{scenario_num:05d}"
            
            # Progress indicator
            if scenario_num % 50 == 0 or scenario_num == 1:
                print(f"  Processing {scenario_num}/{total} "
                      f"({100*scenario_num/total:.0f}%)...")
            
            # Build parameter dict
            metals = METAL_LEVELS[metal_lvl]
            tex = TEXTURE_MAP[texture]
            ionic = IONIC_MAP[ionic_lvl]
            ca_mg = CA_MG_MAP[ca_mg_lvl]
            
            params = {
                "ph": ph,
                "metal_level": metal_lvl,
                "pb_mg": metals["Pb"],
                "cu_mg": metals["Cu"],
                "zn_mg": metals["Zn"],
                "cd_mg": metals["Cd"],
                "doc_mg": tex["doc"],
                "ca_mg": ca_mg["Ca"],
                "mg_mg": ca_mg["Mg"],
                "na_mg": ionic["Na"],
                "cl_mg": ionic["Cl"],
                "texture": texture,
                "hfo": tex["hfo"],
                "moisture": moisture,
                "pe": MOISTURE_MAP[moisture],
                "ionic_level": ionic_lvl,
                "ca_mg_level": ca_mg_lvl,
            }
            
            # Generate PHREEQC input
            input_text = generate_phreeqc_input(params)
            input_file = os.path.join(INPUT_DIR, f"{scenario_id}.phr")
            output_file = os.path.join(OUTPUT_DIR, f"{scenario_id}.txt")
            
            with open(input_file, 'w') as f:
                f.write(input_text)
            
            # Run PHREEQC
            if not run_phreeqc(input_file, output_file):
                fail_count += 1
                continue
            
            # Parse results
            results = parse_phreeqc_output(output_file, params)
            if results is None:
                fail_count += 1
                continue
            
            # Check for any None values in results
            if any(v is None for v in results.values()):
                fail_count += 1
                continue
            
            # Write CSV row
            row = {
                "scenario_id": scenario_id,
                "ph": ph,
                "metal_level": metal_lvl,
                "pb_mg_L": metals["Pb"],
                "cu_mg_L": metals["Cu"],
                "zn_mg_L": metals["Zn"],
                "cd_mg_L": metals["Cd"],
                "doc_mg_L": tex["doc"],
                "ca_mg_L": ca_mg["Ca"],
                "mg_mg_L": ca_mg["Mg"],
                "na_mg_L": ionic["Na"],
                "cl_mg_L": ionic["Cl"],
                "chelator": "None",
                "dose_mg_L": 0,
                "texture": texture,
                "hfo_sites": tex["hfo"],
                "moisture": moisture,
                "pe": MOISTURE_MAP[moisture],
                "ca_mg_level": ca_mg_lvl,
                "ionic_level": ionic_lvl,
            }
            row.update(results)
            writer.writerow(row)
            success_count += 1
    
    # Summary
    print(f"\n{'=' * 60}")
    print(f"COMPLETE!")
    print(f"  Successful: {success_count}/{total}")
    print(f"  Failed:     {fail_count}/{total}")
    print(f"  CSV saved:  {csv_file}")
    print(f"{'=' * 60}")
    
    if success_count == total:
        print(f"\nAll {total} baseline scenarios generated successfully!")
        print(f"\nNEXT STEP: Merge this with your existing training data:")
        print(f"  Run: python3 merge_baseline_into_training.py")
    else:
        print(f"\nWARNING: {fail_count} scenarios failed. Check output above for errors.")


if __name__ == "__main__":
    main()
