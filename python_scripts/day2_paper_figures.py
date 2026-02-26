#!/usr/bin/env python3
"""
DAY 2: Generate publication-quality figures for methodology paper.
Consistent color schemes throughout. Saves PNG + PDF versions.

Figures:
  2 — pH vs. % Free Metal by Chelator (faceted)
  3 — Chelator Comparison Heatmap
  4 — Dose-Response Curves
  5 — Feature Importance (4 panels)
  6 — Tier 1 Validation Summary
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import json
import os

# === PATHS ===
BASE = "/Users/mallorymalz/Documents/chelator_ml_project"
DATA = os.path.join(BASE, "data", "complete_training_data_with_baseline.csv")
MODELS = os.path.join(BASE, "models")
TIER1 = os.path.join(BASE, "data", "tier1_validation_report_v2.csv")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# === LOAD DATA ===
print("Loading data...")
df = pd.read_csv(DATA)
print(f"  {len(df)} rows loaded")

# === GLOBAL STYLE ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# === CONSISTENT COLOR PALETTES ===
# Metal colors — used everywhere metals appear
METAL_COLORS = {
    'Pb': '#2166AC',   # steel blue
    'Cu': '#B2182B',   # deep red
    'Zn': '#4DAF4A',   # green
    'Cd': '#FF7F00',   # orange
}

# Chelator colors — used everywhere chelators appear
CHELATOR_COLORS = {
    'No Treatment': '#999999',  # gray
    'EDTA':    '#2166AC',       # blue
    'NTA':     '#B2182B',       # red
    'Citrate': '#4DAF4A',       # green
    'Humic':   '#FF7F00',       # orange
    'Fulvic':  '#984EA3',       # purple
}

# Chelator line styles for line plots
CHELATOR_LINESTYLES = {
    'EDTA': '-',
    'NTA': '--',
    'Citrate': '-.',
    'Humic': ':',
    'Fulvic': ':',
}

CHELATOR_MARKERS = {
    'EDTA': 'o',
    'NTA': 's',
    'Citrate': '^',
    'Humic': 'D',
    'Fulvic': 'v',
}

METALS = ['Pb', 'Cu', 'Zn', 'Cd']
TARGET_COLS = ['pb_percent_free', 'cu_percent_free', 'zn_percent_free', 'cd_percent_free']
CHELATORS_ORDERED = ['EDTA', 'NTA', 'Citrate', 'Humic', 'Fulvic']

def save_fig(fig, name):
    """Save figure as both PNG and PDF."""
    png_path = os.path.join(FIGDIR, f"{name}.png")
    pdf_path = os.path.join(FIGDIR, f"{name}.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {name}.png and {name}.pdf")
    plt.close(fig)


# =====================================================================
# FIGURE 2: pH vs. % Free Metal, faceted by chelator
# =====================================================================
print("\nGenerating Figure 2: pH vs. Free Metal by Chelator...")

chelated = df[df['dose_mg_L'] > 0].copy()

fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True)
axes = axes.flatten()

for i, chel in enumerate(CHELATORS_ORDERED):
    ax = axes[i]
    subset = chelated[chelated['chelator'] == chel]

    for metal, col in zip(METALS, TARGET_COLS):
        means = subset.groupby('ph')[col].mean()
        sems = subset.groupby('ph')[col].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    color=METAL_COLORS[metal], marker='o', markersize=5,
                    linewidth=2, capsize=3, label=metal)

    ax.set_title(chel)
    ax.set_xlabel('pH')
    ax.set_xlim(5.2, 7.8)
    ax.set_ylim(-2, 105)
    ax.set_xticks([5.5, 6.0, 6.5, 7.0, 7.5])
    if i % 3 == 0:
        ax.set_ylabel('Free Metal Fraction (%)')
    if i == 0:
        ax.legend(loc='upper right', framealpha=0.9)

# Use last panel for no-treatment baseline
ax = axes[5]
baseline = df[df['dose_mg_L'] == 0]
for metal, col in zip(METALS, TARGET_COLS):
    means = baseline.groupby('ph')[col].mean()
    sems = baseline.groupby('ph')[col].sem()
    ax.errorbar(means.index, means.values, yerr=sems.values,
                color=METAL_COLORS[metal], marker='o', markersize=5,
                linewidth=2, capsize=3, label=metal)
ax.set_title('No Treatment')
ax.set_xlabel('pH')
ax.set_xlim(5.2, 7.8)
ax.set_xticks([5.5, 6.0, 6.5, 7.0, 7.5])

fig.suptitle('Effect of pH on Free Metal Fraction by Chelator Type', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, "fig2_ph_vs_free_metal_by_chelator")


# =====================================================================
# FIGURE 3: Chelator Comparison Heatmap
# =====================================================================
print("\nGenerating Figure 3: Chelator Comparison Heatmap...")

# Mean % free at pH 7.0 for clearest comparison
ph7 = df[df['ph'] == 7.0]
chelators_with_baseline = ['No Treatment'] + CHELATORS_ORDERED

heatmap_data = []
for chel in chelators_with_baseline:
    if chel == 'No Treatment':
        subset = ph7[ph7['dose_mg_L'] == 0]
    else:
        subset = ph7[(ph7['chelator'] == chel) & (ph7['dose_mg_L'] > 0)]

    row = {}
    for metal, col in zip(METALS, TARGET_COLS):
        row[metal] = subset[col].mean() if len(subset) > 0 else np.nan
    heatmap_data.append(row)

hm_df = pd.DataFrame(heatmap_data, index=chelators_with_baseline)

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(hm_df.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

# Labels
ax.set_xticks(range(len(METALS)))
ax.set_xticklabels(METALS, fontsize=12)
ax.set_yticks(range(len(chelators_with_baseline)))
ax.set_yticklabels(chelators_with_baseline, fontsize=11)

# Annotate cells with values
for i in range(len(chelators_with_baseline)):
    for j in range(len(METALS)):
        val = hm_df.values[i, j]
        if not np.isnan(val):
            text_color = 'white' if val > 65 or val < 15 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='Free Metal Fraction (%)')
ax.set_title('Mean Free Metal Fraction by Chelator at pH 7.0', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Metal', fontsize=12)
ax.set_ylabel('Chelator', fontsize=12)

fig.tight_layout()
save_fig(fig, "fig3_chelator_heatmap")


# =====================================================================
# FIGURE 4: Dose-Response Curves
# =====================================================================
print("\nGenerating Figure 4: Dose-Response Curves...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

doses = [50, 150, 300]

for idx, (metal, col) in enumerate(zip(METALS, TARGET_COLS)):
    ax = axes[idx]

    for chel in CHELATORS_ORDERED:
        means = []
        sems = []
        for dose in doses:
            subset = chelated[(chelated['chelator'] == chel) & (chelated['dose_mg_L'] == dose)]
            means.append(subset[col].mean())
            sems.append(subset[col].sem())

        ax.errorbar(doses, means, yerr=sems,
                    color=CHELATOR_COLORS[chel],
                    linestyle=CHELATOR_LINESTYLES[chel],
                    marker=CHELATOR_MARKERS[chel],
                    markersize=7, linewidth=2, capsize=4,
                    label=chel)

    # Add no-treatment baseline as horizontal dashed line
    baseline_mean = df[df['dose_mg_L'] == 0][col].mean()
    ax.axhline(y=baseline_mean, color=CHELATOR_COLORS['No Treatment'],
               linestyle='--', linewidth=1.5, alpha=0.7, label='No Treatment')

    ax.set_title(metal, fontsize=13, fontweight='bold')
    ax.set_xlabel('Chelator Dose (mg/L)')
    ax.set_ylabel('Free Metal Fraction (%)')
    ax.set_xticks(doses)
    ax.set_xlim(20, 330)
    ax.set_ylim(-2, 105)

    if idx == 0:
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

fig.suptitle('Dose-Response Relationships by Metal and Chelator', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, "fig4_dose_response")


# =====================================================================
# FIGURE 5: Feature Importance (4 panels)
# =====================================================================
print("\nGenerating Figure 5: Feature Importance...")

report_path = os.path.join(MODELS, "training_report.json")
with open(report_path, 'r') as f:
    report = json.load(f)

# Clean feature names for display
FEATURE_DISPLAY = {
    'ph': 'pH',
    'chelator_encoded': 'Chelator Type',
    'dose_mg_L': 'Chelator Dose',
    'pe': 'pe (Redox)',
    'cd_mg_L': 'Cd Conc.',
    'cl_mg_L': 'Cl Conc.',
    'na_mg_L': 'Na Conc.',
    'hfo_sites': 'Surface Sites (Hfo)',
    'doc_mg_L': 'DOC',
    'pb_mg_L': 'Pb Conc.',
    'cu_mg_L': 'Cu Conc.',
    'zn_mg_L': 'Zn Conc.',
    'ca_mg_L': 'Ca Conc.',
    'mg_mg_L': 'Mg Conc.',
}

# Try to get feature importances from model files
import warnings
warnings.filterwarnings('ignore')

try:
    import joblib
    has_joblib = True
except ImportError:
    has_joblib = False

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

feature_cols = report.get('feature_columns', [])

for idx, (metal, col) in enumerate(zip(METALS, TARGET_COLS)):
    ax = axes[idx]

    # Try loading actual model for feature importances
    importances = None
    if has_joblib:
        model_path = os.path.join(MODELS, f"{col}_model.joblib")
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                importances = model.feature_importances_
            except Exception as e:
                print(f"    Could not load {col} model: {e}")

    if importances is not None and len(importances) == len(feature_cols):
        # Sort by importance
        feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        top_n = 10  # Show top 10
        feat_imp = feat_imp[:top_n]

        names = [FEATURE_DISPLAY.get(f, f) for f, _ in feat_imp]
        values = [v for _, v in feat_imp]

        colors = [METAL_COLORS[metal]] * len(names)
        # Highlight top 3
        for j in range(min(3, len(colors))):
            colors[j] = METAL_COLORS[metal]

        bars = ax.barh(range(len(names)), values, color=METAL_COLORS[metal], alpha=0.85, edgecolor='white')

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance')
        ax.set_title(metal, fontsize=13, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0.01:
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{val:.3f}', va='center', fontsize=9)
    else:
        # Fallback: show top 3 from report
        top3 = report['results'].get(col, {}).get('top_3_features', [])
        clean_top3 = [FEATURE_DISPLAY.get(f, f) for f in top3]
        ax.text(0.5, 0.5, f"Top features:\n" + "\n".join(f"{i+1}. {f}" for i, f in enumerate(clean_top3)),
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(metal, fontsize=13, fontweight='bold')

fig.suptitle('Feature Importance by Metal (Gradient Boosting)', fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()
save_fig(fig, "fig5_feature_importance")


# =====================================================================
# FIGURE 6: Tier 1 Validation Summary
# =====================================================================
print("\nGenerating Figure 6: Tier 1 Validation Summary...")

tier1 = pd.read_csv(TIER1)

fig, ax = plt.subplots(figsize=(10, 5))

# Short rule names for display
rule_short = [
    'pH Decreases\nFree Metal',
    'Chelator Reduces\nFree vs. Baseline',
    'Higher Dose\nDecreases Free',
    'More Surface Sites\nDecrease Free',
    'EDTA Outperforms\nNTA for Pb/Cu',
    'Zn Harder to\nChelate than Cu',
    'Chelator Better\nat Higher pH',
    'High Ionic Strength\nReduces Free Pb/Cu',
]

pass_rates = tier1['pass_rate'].values
colors = ['#2166AC' if r >= 99.9 else '#FF7F00' for r in pass_rates]

bars = ax.bar(range(len(pass_rates)), pass_rates, color=colors, edgecolor='white', width=0.7)

# Add value labels
for bar, val in zip(bars, pass_rates):
    y_pos = bar.get_height() - 3 if bar.get_height() > 10 else bar.get_height() + 1
    text_color = 'white' if bar.get_height() > 50 else 'black'
    ax.text(bar.get_x() + bar.get_width()/2, y_pos,
            f'{val:.1f}%', ha='center', va='top', fontsize=10,
            fontweight='bold', color=text_color)

ax.set_xticks(range(len(rule_short)))
ax.set_xticklabels(rule_short, fontsize=9, ha='center')
ax.set_ylabel('Pass Rate (%)')
ax.set_ylim(0, 108)
ax.set_title('Internal Consistency Validation: Chemical Logic Rules (Tier 1)', fontsize=14, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2166AC', label='100% Pass'),
                   Patch(facecolor='#FF7F00', label='Edge-Case Violations')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Add horizontal line at 100%
ax.axhline(y=100, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

fig.tight_layout()
save_fig(fig, "fig6_tier1_validation")


# =====================================================================
# DONE
# =====================================================================
print("\n" + "=" * 70)
print("  DAY 2 COMPLETE: All Figures Generated")
print("=" * 70)
print(f"\n  Saved to: {FIGDIR}/")
for f in sorted(os.listdir(FIGDIR)):
    if f.startswith('fig'):
        print(f"    {f}")
print("\n  Both PNG (for drafting) and PDF (for submission) versions saved.")
print("  Figure 1 (pipeline schematic) and Figure 7 (screenshot) are manual tasks.")
