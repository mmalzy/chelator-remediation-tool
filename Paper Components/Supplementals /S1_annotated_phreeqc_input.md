# Supplementary Material S1: Annotated PHREEQC Input File

## Example Scenario: Medium Contamination, EDTA 150 mg/L, pH 6.5, Loam, Mesic, Medium Ionic Strength

The following is a representative PHREEQC input file from the 12,636-scenario training dataset. This scenario models a moderately contaminated loam soil under mesic (field capacity) moisture conditions with medium ionic strength, treated with EDTA at 150 mg/L. Annotations explain each block and parameter choice.

```
# =====================================================================
# PHREEQC Input File — Chelator ML Remediation Project
# Scenario: Medium contamination, EDTA 150 mg/L, pH 6.5, Loam, Mesic,
#           Medium ionic strength, High Ca/Mg competition
# Database: minteq.v4.dat
# Run command: phreeqc <this_file>.phr output.txt /usr/local/share/phreeqc_databases/minteq.v4.dat
# =====================================================================

TITLE Medium metals, EDTA 150 mg/L, pH 6.5, Loam, Mesic, Medium ionic

# ----- SOLUTION BLOCK -----
# Defines the aqueous solution representing soil pore water.
# All concentrations are specified in mol/L.
# PHREEQC calculates thermodynamic equilibrium speciation for all
# species in the minteq.v4.dat database at the specified pH, pe, and
# temperature.

SOLUTION 1
    temp      25              # Temperature in degrees Celsius (standard lab conditions)
    pH        6.5             # Soil pore water pH (one of five levels: 5.5, 6.0, 6.5, 7.0, 7.5)
    pe        8               # Electron activity: Mesic conditions (Dry=12, Mesic=8, Wet=3)
                              #   pe is a proxy for soil moisture/redox state.
                              #   Lower pe = more reducing = wetter conditions.
    units     mol/L           # All concentrations below in mol/L

    # --- Target metals (Medium contamination level) ---
    # Converted from mg/L to mol/L using: mol/L = (mg/L / 1000) / atomic_weight
    Pb        4.826e-04       # 100 mg/L Pb ÷ 1000 ÷ 207.2 g/mol = 4.826e-04 mol/L
    Cu        1.259e-03       # 80 mg/L Cu ÷ 1000 ÷ 63.546 g/mol = 1.259e-03 mol/L
    Zn        1.835e-03       # 120 mg/L Zn ÷ 1000 ÷ 65.38 g/mol = 1.835e-03 mol/L
    Cd        7.120e-05       # 8 mg/L Cd ÷ 1000 ÷ 112.41 g/mol = 7.120e-05 mol/L

    # --- Competing cations (High Ca/Mg level) ---
    # Ca and Mg compete with target metals for chelator binding sites.
    # Two levels used: Low (Ca=20, Mg=10 mg/L) and High (Ca=100, Mg=50 mg/L).
    Ca        2.495e-03       # 100 mg/L Ca ÷ 1000 ÷ 40.078 g/mol = 2.495e-03 mol/L
    Mg        2.058e-03       # 50 mg/L Mg ÷ 1000 ÷ 24.305 g/mol = 2.058e-03 mol/L

    # --- Ionic strength ions (Medium level) ---
    # Na and Cl control ionic strength, which affects activity coefficients
    # and enables chloride complexation (PbCl+, CuCl+, CdCl+, CdCl2).
    # Three levels: Low (Na=100, Cl=150), Medium (Na=500, Cl=700), High (Na=2000, Cl=3000 mg/L).
    # High level represents Rhode Island coastal/road salt conditions.
    Na        2.174e-02       # 500 mg/L Na ÷ 1000 ÷ 23.0 g/mol = 2.174e-02 mol/L
    Cl        1.974e-02       # 700 mg/L Cl ÷ 1000 ÷ 35.453 g/mol = 1.974e-02 mol/L

    # --- Dissolved organic carbon ---
    # DOC is tied to soil texture: Sand=10, Loam=25, Clay=40 mg/L.
    # Entered as C(4) (dissolved inorganic + organic carbon species).
    # DOC provides additional solution-phase complexation capacity for metals.
    C(4)      2.083e-03       # 25 mg/L DOC ÷ 1000 ÷ 12.011 g/mol = 2.083e-03 mol/L

    # --- Chelating agent: EDTA at 150 mg/L ---
    # EDTA (ethylenediaminetetraacetic acid), MW = 292.24 g/mol.
    # minteq.v4.dat contains stability constants for Pb-EDTA, Cu-EDTA,
    # Zn-EDTA, Cd-EDTA, Ca-EDTA, Mg-EDTA, and Fe-EDTA complexes.
    # EDTA speciation is pH-dependent: H6EDTA2+ to EDTA4- across pH range.
    Edta      5.133e-04       # 150 mg/L EDTA ÷ 1000 ÷ 292.24 g/mol = 5.133e-04 mol/L

    # Note: For NTA, the keyword is "Nta" (MW = 191.14 g/mol).
    # For Citrate, the keyword is "Citrate" (MW = 189.1 g/mol).
    # For Humic/Fulvic acids, additional C(4) is added as a DOC proxy
    #   because minteq.v4.dat does not include explicit humic binding models.
    #   Humic: extra C(4) = dose / 12.011 (full DOC equivalent)
    #   Fulvic: extra C(4) = dose * 0.8 / 12.011 (reduced binding capacity)

# ----- SURFACE BLOCK -----
# Models metal sorption to iron/aluminum oxide surfaces in soil.
# Uses the generalized two-layer surface complexation model
# (Dzombak and Morel, 1990) implemented in PHREEQC.

SURFACE 1
    Hfo_wOH   0.5   600   0.09
    # Hfo_wOH = hydrous ferric oxide weak binding sites
    #   0.5 = moles of surface sites (Loam texture)
    #         Sand = 0.1 mol, Loam = 0.5 mol, Clay = 1.5 mol
    #         More sites = more sorption capacity = lower free metal
    #   600 = specific surface area (m²/g), standard for Hfo
    #   0.09 = mass of Hfo (g), standard parameterization
    -equil 1  # Equilibrate surface with Solution 1

END
```

## Parameter Conversion Reference

All metal and reagent concentrations are converted from field-relevant mg/L units to mol/L for PHREEQC input using:

**mol/L = (mg/L ÷ 1000) ÷ molecular weight (g/mol)**

## Output Parsing

PHREEQC output files (latin-1 encoding) report the molality of every dissolved species. The target variable — percent free metal — is calculated as:

**% free = (free ion molality ÷ total dissolved metal molality) × 100**

For example, for lead: % free Pb = molality of Pb²⁺ ÷ (molality of Pb²⁺ + PbOH⁺ + PbCO₃⁰ + PbCl⁺ + PbCl₂⁰ + Pb(EDTA)²⁻ + Pb(Citrate)⁻ + ...) × 100

Species sorbed to the Hfo surface (Hfo_wOPb⁺, Hfo_wOCu⁺, etc.) are reported separately and recorded as secondary target variables (pb_sorbed_mol, cu_sorbed_mol, etc.).

---
