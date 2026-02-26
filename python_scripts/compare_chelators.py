#!/usr/bin/env python3
"""
Quick script to extract free Pb from PHREEQC output files
"""

import re

def extract_free_pb(filename):
    """Extract Pb+2 and total Pb from PHREEQC output"""
    with open(filename, 'r', encoding='latin-1') as f:  # Changed encoding
        content = f.read()
    
    # Find the Pb section in the distribution of species
    pb_section = re.search(r'Pb\s+(\d\.\d+e[+-]\d+).*?Pb\+2\s+(\d\.\d+e[+-]\d+)', content, re.DOTALL)
    
    if pb_section:
        total_pb = float(pb_section.group(1))
        free_pb = float(pb_section.group(2))
        percent_free = (free_pb / total_pb) * 100
        
        return {
            'total_pb': total_pb,
            'free_pb': free_pb,
            'percent_free': percent_free
        }
    return None

# Compare the three tests
tests = {
    'No chelator (test3)': '../phreeqc_outputs/test3_output.txt',
    'Citrate (test4)': '../phreeqc_outputs/test4_output.txt',
    'EDTA (test5)': '../phreeqc_outputs/test5_output.txt'
}

print("=" * 60)
print("COMPARISON: Free Pb+2 with Different Chelators")
print("=" * 60)

for name, filepath in tests.items():
    try:
        result = extract_free_pb(filepath)
        if result:
            print(f"\n{name}:")
            print(f"  Total Pb:     {result['total_pb']:.3e} mol/L")
            print(f"  Free Pb+2:    {result['free_pb']:.3e} mol/L")
            print(f"  % Free:       {result['percent_free']:.1f}%")
        else:
            print(f"\n{name}: Could not extract data")
    except Exception as e:
        print(f"\n{name}: Error - {e}")

print("\n" + "=" * 60)
