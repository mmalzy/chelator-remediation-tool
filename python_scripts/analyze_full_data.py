#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('../data/full_training_data.csv')

print("=" * 80)
print("BEST SCENARIOS FOR EACH METAL")
print("=" * 80)

# Best for Pb
print("\nBest 5 for Pb (lowest % free):")
best_pb = df.nsmallest(5, 'pb_percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'moisture', 'pb_percent_free']]
print(best_pb.to_string(index=False))

# Best for Cu
print("\nBest 5 for Cu (lowest % free):")
best_cu = df.nsmallest(5, 'cu_percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'moisture', 'cu_percent_free']]
print(best_cu.to_string(index=False))

# Best for Zn
print("\nBest 5 for Zn (lowest % free):")
best_zn = df.nsmallest(5, 'zn_percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'moisture', 'zn_percent_free']]
print(best_zn.to_string(index=False))

# Best overall (sum of all three)
df['total_percent_free'] = df['pb_percent_free'] + df['cu_percent_free'] + df['zn_percent_free']
print("\nBest 5 overall (lowest combined % free for all metals):")
best_all = df.nsmallest(5, 'total_percent_free')[['scenario_id', 'ph', 'chelator', 'dose_mg_L', 'texture', 'pb_percent_free', 'cu_percent_free', 'zn_percent_free', 'total_percent_free']]
print(best_all.to_string(index=False))

# Effect of chelator on Zn specifically
print("\n" + "=" * 80)
print("ZN CHELATION EFFECTIVENESS")
print("=" * 80)
zn_by_chelator = df.groupby(['chelator', 'dose_mg_L'])['zn_percent_free'].agg(['mean', 'min', 'max'])
print(zn_by_chelator)

# High dose effectiveness
print("\n" + "=" * 80)
print("AT HIGHEST DOSE (150 mg/L) - ALL METALS")
print("=" * 80)
high_dose = df[df['dose_mg_L'] == 150].groupby('chelator')[['pb_percent_free', 'cu_percent_free', 'zn_percent_free']].mean()
print(high_dose)
