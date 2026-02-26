# Key Findings to Highlight in Paper
## Reference Notes from Day 1 Data Analysis
## February 25, 2026

---

## SURPRISING FINDING 1: Chelators Can Increase Free Pb (Average Effect)

Averaged across all chelators and all conditions, chelators slightly INCREASE free Pb:
- No Treatment: 43.1% free Pb
- With Chelator: 44.2% free Pb (net increase of 1.1 percentage points)

Why this matters: The average is dragged down by Humic/Fulvic (which barely help) and
low-dose scenarios at low pH where competitive desorption occurs. EDTA and Citrate at
adequate doses dramatically reduce free Pb, but the overall average masks this. This is
an important practical finding — choosing the wrong chelator or underdosing can make
things worse, not better. Emphasize in Discussion that chelator selection is not
"any chelator helps" but requires matching the right agent to site conditions.

Use in paper: Section 3.1.3 (Chelator Effectiveness Patterns) and Section 3.4
(Practical Application discussion)

---

## SURPRISING FINDING 2: pH Effect on Cu Is Dramatic (6x Reduction)

Cu free fraction drops from 49.1% at pH 5.5 to 8.0% at pH 7.5 — a sixfold reduction
from pH alone, without any chelator. This is the strongest pH response of any metal.

Why this matters: For Cu-contaminated sites, liming (raising pH) may be more
cost-effective than chelation. At pH 7.5, Cu is already 92% complexed or sorbed even
without treatment. Chelators provide marginal additional benefit at high pH but are
critical at low pH where Cu is highly mobile.

Use in paper: Section 3.1.2 (pH as Dominant Control) — highlight Cu as the most
pH-sensitive metal. Also Section 3.5 (Practical Application) — practitioners should
consider pH amendment before or alongside chelation for Cu.

---

## SURPRISING FINDING 3: Ionic Strength Effect on Cd Is Massive

Cd free fraction drops from 76.1% at low ionic strength to 19.0% at high ionic strength.
That is a 57 percentage point reduction — larger than any chelator effect for any metal.

Why this matters: For Rhode Island coastal sites with high salinity, the natural ionic
environment is already doing significant Cd immobilization through chloride complexation
(CdCl+, CdCl2). This is a site-specific advantage that inland contaminated sites do not
have. However, chloride-complexed Cd is still mobile (just not "free") so the
bioavailability interpretation requires nuance.

Use in paper: Section 3.1.4 (Rhode Island-Specific Findings) — this is your strongest
RI-specific result. Also Discussion — the model captures geochemical effects that
practitioners would not intuit without running speciation calculations.

---

## SURPRISING FINDING 4: Texture Effect Is Surprisingly Small

Mean percent free metal by texture:
- Clay: Pb 43.5%, Cu 31.0%, Zn 84.3%, Cd 47.5%
- Loam: Pb 44.2%, Cu 31.2%, Zn 84.5%, Cd 47.5%
- Sand: Pb 44.9%, Cu 31.5%, Zn 84.6%, Cd 47.5%

Differences are only 1-1.5 percentage points between Clay and Sand.

Why this matters: In pore water speciation, pH and ionic strength effects dominate
over surface sorption within the Hfo ranges modeled (0.1 to 1.5 mol). This does NOT
mean texture is unimportant for total soil remediation — it strongly affects
hydraulic conductivity, chelator delivery, and physical access to contaminants. But
for the specific question of "what fraction of dissolved metal is free vs complexed,"
solution chemistry (pH, chelators, ionic strength) matters more than surface chemistry
in our model's parameter space.

Use in paper: Section 3.1.5 (Texture and Surface Complexation) — present honestly,
discuss why solution-phase effects dominate in pore water modeling.

---

## CHELATOR RANKINGS BY METAL (for quick reference while writing)

Lead (Pb) — mean % free:
1. Citrate: 32.6% (best)
2. EDTA: 37.4%
3. NTA: 40.2%
4. Humic/Fulvic: 55.5% (weakest)

Copper (Cu) — mean % free:
1. NTA: 18.6% (best)
2. Citrate: 22.6%
3. EDTA: 24.4%
4. Humic/Fulvic: 44.5% (weakest)

Zinc (Zn) — mean % free:
1. NTA: 73.6% (best, but still very high)
2. Citrate: 74.4%
3. EDTA: 81.6%
4. Humic/Fulvic: 95.5% (essentially no effect)

Cadmium (Cd) — mean % free:
1. NTA: 43.6% (best)
2. EDTA: 43.7% (essentially tied)
3. Citrate: 48.3%
4. Humic/Fulvic: 50.6%

Key takeaway: NTA is surprisingly competitive — best for Cu and Zn, tied for Cd.
EDTA is not always the best choice despite being the industry standard. Citrate
is best for Pb specifically. This metal-specific ranking is one of the most
practically useful outputs of the model.

---

## pH EFFECT BY METAL (for Section 3.1.2)

| pH  | Pb (%) | Cu (%) | Zn (%) | Cd (%) |
|-----|--------|--------|--------|--------|
| 5.5 | 54.4   | 49.1   | 86.0   | 47.9   |
| 6.0 | 52.6   | 45.1   | 85.1   | 47.6   |
| 6.5 | 49.8   | 35.1   | 85.4   | 47.7   |
| 7.0 | 40.8   | 19.0   | 83.9   | 47.2   |
| 7.5 | 23.7   | 8.0    | 81.8   | 46.9   |

Key observations:
- Pb drops 30.7 pp from pH 5.5 to 7.5 (strong response)
- Cu drops 41.1 pp from pH 5.5 to 7.5 (strongest response of all metals)
- Zn drops only 4.2 pp (weak pH response — Zn stays highly free regardless)
- Cd drops only 1.0 pp (essentially no pH response in this range)

This differential pH sensitivity is important — it means pH amendment helps
dramatically for Pb and Cu but barely affects Zn and Cd.

---

## MODEL PERFORMANCE HIGHLIGHTS (for Section 3.2)

- All four models: Gradient Boosting outperformed Random Forest
- Best performing: Cd (R² = 1.0000, RMSE = 0.15%)
- Most challenging: Cu (CV R² = 0.9481, highest CV std = 0.0765)
- Cu complexity likely due to strong pH-chelator-organic matter interactions
- Top features vary by metal:
  - Pb: pH, chelator, dose (classic chelation response)
  - Cu: pH, pe, chelator (redox matters more for Cu)
  - Zn: chelator, dose, Cd concentration (cross-metal competition)
  - Cd: Cl, Na, chelator (ionic strength dominates — the RI finding)
- The feature importance results confirm the model learned real geochemistry

---
