#!/usr/bin/env python3
"""
Analyze pilot training data to see patterns
"""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('../data/pilot_training_data.csv')

print("=" * 70)
print("PILOT DATA ANALYSIS: 120 Scenarios")
print("=" * 70)

# Overall statistics
print("\n1. OVERALL STATISTICS")
print("-" * 70)
print(df[['percent_free', 'free_pb_mol', 'sorbed_pb_mol']].describe())

# Effect of chelator type
print("\n2. EFFECT OF CHELATOR TYPE")
print("-" * 70)
chelator_effect = df.groupby('chelator')['percent_free'].agg(['mean', 'std', 'min', 'max'])
print(chelator_effect)

# Effect of dose (for EDTA and Citrate only)
print("\n3. EFFECT OF CHELATOR DOSE")
print("-" * 70)
for chelator in ['EDTA', 'Citrate']:
    print(f"\n{chelator}:")
    dose_effect = df[df['chelator'] == chelator].groupby('dose_mg_L')['percent_free'].agg(['mean', 'std', 'count'])
    print(dose_effect)

# Effect of pH
print("\n4. EFFECT OF pH")
print("-" * 70)
ph_effect = df.groupby('ph')['percent_free'].agg(['mean', 'std', 'min', 'max'])
print(ph_effect)

# Effect of soil texture
print("\n5. EFFECT OF SOIL TEXTURE")
print("-" * 70)
texture_effect = df.groupby('texture')['percent_free'].agg(['mean', 'std', 'min', 'max'])
print(texture_effect)

# Effect of moisture
print("\n6. EFFECT OF MOISTURE")
print("-" * 70)
moisture_effect = df.groupby('moisture')['percent_free'].agg(['mean', 'std', 'min', 'max'])
print(moisture_effect)

# Best and worst scenarios
print("\n7. BEST SCENARIOS (Lowest % Free Pb)")
print("-" * 70)
best = df.nsmallest(5, 'percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'percent_free']]
print(best.to_string(index=False))

print("\n8. WORST SCENARIOS (Highest % Free Pb)")
print("-" * 70)
worst = df.nlargest(5, 'percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'percent_free']]
print(worst.to_string(index=False))

# Key insights
print("\n" + "=" * 70)
print("KEY INSIGHTS:")
print("=" * 70)

# Best chelator at high dose
edta_100 = df[(df['chelator'] == 'EDTA') & (df['dose_mg_L'] == 100)]['percent_free'].mean()
citrate_100 = df[(df['chelator'] == 'Citrate') & (df['dose_mg_L'] == 100)]['percent_free'].mean()
none = df[df['chelator'] == 'None']['percent_free'].mean()

print(f"✓ No chelator: {none:.1f}% free Pb (baseline)")
print(f"✓ EDTA at 100 mg/L: {edta_100:.4f}% free Pb (reduction: {none - edta_100:.1f}%)")
print(f"✓ Citrate at 100 mg/L: {citrate_100:.1f}% free Pb (reduction: {none - citrate_100:.1f}%)")
print(f"\n✓ EDTA is {citrate_100/edta_100:.0f}x more effective than Citrate at high dose!")

# pH effect
print(f"\n✓ pH 5.5: {df[df['ph'] == 5.5]['percent_free'].mean():.1f}% free")
print(f"✓ pH 6.5: {df[df['ph'] == 6.5]['percent_free'].mean():.1f}% free")
print(f"✓ pH 7.5: {df[df['ph'] == 7.5]['percent_free'].mean():.1f}% free")

print("\n" + "=" * 70)
