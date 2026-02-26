#!/usr/bin/env python3
"""Quick diagnostic to check what's in the literature CSV and how it maps."""
import pandas as pd
import os

CSV = "/Users/mallorymalz/Documents/chelator_ml_project/data/literature_benchmark_data.csv"

print("=== CHECKING LITERATURE CSV ===\n")

if not os.path.exists(CSV):
    print(f"FILE NOT FOUND: {CSV}")
    print("\nLooking for any CSV with 'literature' or 'benchmark' in name...")
    data_dir = "/Users/mallorymalz/Documents/chelator_ml_project/data/"
    for f in os.listdir(data_dir):
        if 'lit' in f.lower() or 'bench' in f.lower():
            print(f"  Found: {f}")
    exit()

df = pd.read_csv(CSV)
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 3 rows:")
print(df.head(3).to_string())

print(f"\n--- Key column values ---")
print(f"chelator_used unique: {df['chelator_used'].unique().tolist()}")
print(f"metal unique: {df['metal'].unique().tolist()}")
print(f"chelator_dose dtype: {df['chelator_dose'].dtype}")
print(f"chelator_dose values: {df['chelator_dose'].unique().tolist()[:10]}")
print(f"dose_unit unique: {df['dose_unit'].unique().tolist()}")
print(f"ph dtype: {df['ph'].dtype}")
print(f"ph values: {df['ph'].unique().tolist()}")
print(f"pb_mg_kg dtype: {df['pb_mg_kg'].dtype}")

# Test the mapping
def map_chelator(name):
    n = str(name).upper().strip()
    if 'EDTA' in n: return 'EDTA'
    if 'NTA' in n: return 'NTA'
    if 'CITRI' in n or 'CITRATE' in n: return 'Citrate'
    if 'HUMIC' in n: return 'Humic'
    if 'FULVIC' in n: return 'Fulvic'
    if n in ['NONE', 'CONTROL', '']: return 'nan'
    return None

print(f"\n--- Chelator mapping test ---")
for chel in df['chelator_used'].unique():
    mapped = map_chelator(chel)
    print(f"  '{chel}' -> '{mapped}'")

print(f"\n--- Metal column test ---")
for m in df['metal'].unique():
    valid = str(m).lower() in ['pb', 'cu', 'zn', 'cd']
    print(f"  '{m}' -> valid={valid}")

# Check training data matching
TRAINING = "/Users/mallorymalz/Documents/chelator_ml_project/data/complete_training_data_with_baseline.csv"
train = pd.read_csv(TRAINING)
print(f"\n--- Training data unique values ---")
print(f"chelator: {train['chelator'].dropna().unique().tolist()}")
print(f"metal_level: {train['metal_level'].unique().tolist()}")
print(f"texture: {train['texture'].unique().tolist()}")
print(f"ionic_level: {train['ionic_level'].unique().tolist()}")
print(f"moisture: {train['moisture'].unique().tolist()}")
print(f"dose_mg_L: {sorted(train['dose_mg_L'].unique().tolist())}")
print(f"ph: {sorted(train['ph'].unique().tolist())}")
