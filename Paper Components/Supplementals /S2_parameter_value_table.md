# Supplementary Material S2: Complete Parameter Value Table

## Table S2.1: Metal Concentrations

| Parameter | Level | mg/L | Molecular Weight (g/mol) | mol/L | PHREEQC Keyword |
|-----------|-------|------|--------------------------|-------|-----------------|
| Pb | Low | 25 | 207.2 | 1.207e-04 | Pb |
| Pb | Medium | 100 | 207.2 | 4.826e-04 | Pb |
| Pb | High | 300 | 207.2 | 1.448e-03 | Pb |
| Cu | Low | 20 | 63.546 | 3.147e-04 | Cu |
| Cu | Medium | 80 | 63.546 | 1.259e-03 | Cu |
| Cu | High | 250 | 63.546 | 3.934e-03 | Cu |
| Zn | Low | 30 | 65.38 | 4.590e-04 | Zn |
| Zn | Medium | 120 | 65.38 | 1.835e-03 | Zn |
| Zn | High | 400 | 65.38 | 6.118e-03 | Zn |
| Cd | Low | 2 | 112.41 | 1.780e-05 | Cd |
| Cd | Medium | 8 | 112.41 | 7.120e-05 | Cd |
| Cd | High | 25 | 112.41 | 2.224e-04 | Cd |

## Table S2.2: Chelator Concentrations

| Chelator | Dose Level | mg/L | Molecular Weight (g/mol) | mol/L | PHREEQC Keyword |
|----------|-----------|------|--------------------------|-------|-----------------|
| EDTA | Low | 50 | 292.24 | 1.711e-04 | Edta |
| EDTA | Medium | 150 | 292.24 | 5.133e-04 | Edta |
| EDTA | High | 300 | 292.24 | 1.027e-03 | Edta |
| NTA | Low | 50 | 191.14 | 2.616e-04 | Nta |
| NTA | Medium | 150 | 191.14 | 7.847e-04 | Nta |
| NTA | High | 300 | 191.14 | 1.569e-03 | Nta |
| Citrate | Low | 50 | 189.10 | 2.644e-04 | Citrate |
| Citrate | Medium | 150 | 189.10 | 7.932e-04 | Citrate |
| Citrate | High | 300 | 189.10 | 1.586e-03 | Citrate |
| Humic acid | Low | 50 | — | 4.163e-03* | C(4) addition |
| Humic acid | Medium | 150 | — | 1.249e-02* | C(4) addition |
| Humic acid | High | 300 | — | 2.498e-02* | C(4) addition |
| Fulvic acid | Low | 50 | — | 3.330e-03* | C(4) addition |
| Fulvic acid | Medium | 150 | — | 9.992e-03* | C(4) addition |
| Fulvic acid | High | 300 | — | 1.998e-02* | C(4) addition |

*Humic and fulvic acids are modeled as additional dissolved organic carbon (C(4) species) because the minteq.v4.dat database does not include explicit humic substance binding models (e.g., NICA-Donnan or WHAM Model VI). Humic acid: dose (mg/L) ÷ 1000 ÷ 12.011 g/mol. Fulvic acid: dose × 0.8 ÷ 1000 ÷ 12.011 g/mol (factor of 0.8 reflects smaller molecular weight and reduced binding capacity relative to humic acid). This is an acknowledged simplification discussed in Section 3.5.

## Table S2.3: Competing Cations

| Parameter | Level | mg/L | Molecular Weight (g/mol) | mol/L | PHREEQC Keyword |
|-----------|-------|------|--------------------------|-------|-----------------|
| Ca | Low | 20 | 40.078 | 4.990e-04 | Ca |
| Ca | High | 100 | 40.078 | 2.495e-03 | Ca |
| Mg | Low | 10 | 24.305 | 4.114e-04 | Mg |
| Mg | High | 50 | 24.305 | 2.058e-03 | Mg |

## Table S2.4: Ionic Strength Ions

| Parameter | Level | mg/L | Molecular Weight (g/mol) | mol/L | PHREEQC Keyword |
|-----------|-------|------|--------------------------|-------|-----------------|
| Na | Low | 100 | 22.990 | 4.350e-03 | Na |
| Na | Medium | 500 | 22.990 | 2.174e-02 | Na |
| Na | High | 2000 | 22.990 | 8.699e-02 | Na |
| Cl | Low | 150 | 35.453 | 4.231e-03 | Cl |
| Cl | Medium | 700 | 35.453 | 1.974e-02 | Cl |
| Cl | High | 3000 | 35.453 | 8.462e-02 | Cl |

Note: The High ionic strength level (Na = 2000, Cl = 3000 mg/L) represents Rhode Island coastal conditions influenced by tidal saline intrusion and/or winter road salt application.

## Table S2.5: Soil Properties and Environmental Conditions

| Parameter | Level | Value | Proxy for | PHREEQC Implementation |
|-----------|-------|-------|-----------|----------------------|
| pH | — | 5.5, 6.0, 6.5, 7.0, 7.5 | Soil acidity | pH keyword in SOLUTION block |
| Texture | Sand | Hfo_wOH = 0.1 mol; DOC = 10 mg/L | Low clay, low OM, low sorption | SURFACE block + C(4) in SOLUTION |
| Texture | Loam | Hfo_wOH = 0.5 mol; DOC = 25 mg/L | Moderate clay, moderate OM | SURFACE block + C(4) in SOLUTION |
| Texture | Clay | Hfo_wOH = 1.5 mol; DOC = 40 mg/L | High clay, high OM, high sorption | SURFACE block + C(4) in SOLUTION |
| Moisture | Dry | pe = 12 | Oxidizing, well-drained | pe keyword in SOLUTION block |
| Moisture | Mesic | pe = 8 | Field capacity, moderate redox | pe keyword in SOLUTION block |
| Moisture | Wet | pe = 3 | Reducing, saturated/near-saturated | pe keyword in SOLUTION block |

## Table S2.6: Dissolved Organic Carbon by Texture

| Texture | DOC (mg/L) | DOC (mol/L as C) | Rationale |
|---------|-----------|-------------------|-----------|
| Sand | 10 | 8.326e-04 | Low organic matter in sandy soils |
| Loam | 25 | 2.083e-03 | Moderate organic matter |
| Clay | 40 | 3.330e-03 | High organic matter in clay soils |

DOC is tied to texture class because organic matter content correlates strongly with clay content and specific surface area in real soils (Brady and Weil, 2017).

## Table S2.7: Scenario Count Summary

| Factor | Levels | Count |
|--------|--------|-------|
| pH | 5 | 5.5, 6.0, 6.5, 7.0, 7.5 |
| Metal contamination level | 3 | Low, Medium, High |
| Chelator type | 5 | EDTA, NTA, Citrate, Humic, Fulvic |
| Chelator dose | 3 | 50, 150, 300 mg/L |
| Soil texture | 3 | Sand, Loam, Clay |
| Moisture/pe | 3 | Dry (pe=12), Mesic (pe=8), Wet (pe=3) |
| Ionic strength | 3 | Low, Medium, High |
| Ca/Mg competition | 2 | Low, High |

**Chelator scenarios:** 5 pH × 3 metals × 5 chelators × 3 doses × 3 textures × 3 moisture × 3 ionic × 2 Ca/Mg = 12,150

**No-chelator baselines:** 5 pH × 3 metals × 3 textures × 3 moisture × 3 ionic × 2 Ca/Mg ÷ (accounting for unique environmental combinations) = 486

**Total:** 12,150 + 486 = **12,636 scenarios**

---
