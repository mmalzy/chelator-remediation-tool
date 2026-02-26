# Session Notes — February 25, 2026
## Chelator ML Remediation Project: Paper Writing Phase

---

## WHAT WE ACCOMPLISHED TODAY

### 1. Day 1: Summary Statistics & Core Tables (complete)
- Generated all summary statistics for the training dataset (12,636 rows)
- Created publication-ready tables saved to data/paper_tables/:
  - table1_parameter_space.csv — Simulation design (Section 2.1.2)
  - table3_summary_statistics.csv — Free metal fraction stats (Section 3.1.1)
  - table4_model_performance.csv — GB vs RF metrics (Section 3.2)
  - table5_tier1_validation.csv — Chemical logic results (Section 3.3.1)
- Tables have professional formatting: proper capitalization, no underscores, clean column names
- All tables already inserted into the working document

### 2. Day 2: Publication-Quality Figures (complete)
- Generated 5 data figures saved to figures/ as both PNG and PDF:
  - fig2_ph_vs_free_metal_by_chelator — 6-panel faceted plot (5 chelators + No Treatment)
  - fig3_chelator_heatmap — Mean % free by chelator x metal at pH 7.0
  - fig4_dose_response — 4-panel dose-response curves with baseline
  - fig5_feature_importance — 4-panel feature importance from trained models
  - fig6_tier1_validation — Bar chart of 8 chemical logic rules
- Created fig1_pipeline_schematic.svg — Three-panel pipeline diagram (PHREEQC → ML → Streamlit)
  - Fixed text overlapping arrows (moved labels above arrows)
  - Strengthened box outlines to 3px with deeper background fills for Word compatibility
- Consistent color scheme across all figures:
  - Metals: Pb = steel blue (#2166AC), Cu = deep red (#B2182B), Zn = green (#4DAF4A), Cd = orange (#FF7F00)
  - Chelators: EDTA = blue, NTA = red, Citrate = green, Humic = orange, Fulvic = purple, No Treatment = gray
- Figure 7 (Streamlit screenshot) still needed — take manually by running the app

### 3. Day 3: Feature Importance & Validation Figures (complete)
- Already covered in Day 2 figure generation (fig5, fig6)
- No additional work needed

### 4. Day 4: Literature Benchmark & References (complete)
- Regenerated literature_benchmark_data.csv with Kim et al. (2003) replacing Alaboudi et al. (2019)
  - CRITICAL: Alaboudi et al. (2019) Pol J Environ Stud citation could NOT be verified — paper does not exist
  - Replaced with: Kim, C., Lee, Y., & Ong, S. K. (2003). Factors affecting EDTA extraction of lead from lead-contaminated soils. Chemosphere, 51(9), 845-853. DOI: 10.1016/S0045-6535(03)00155-3
  - Kim data points: KIM_1 (equimolar EDTA:Pb, ~50% extraction) and KIM_2 (5:1 ratio, ~90% extraction) at pH 5 on Superfund site soils
  - NOTE: Kim extraction percentages are estimates from abstract/secondary sources. Verify against actual paper before submission.
- Generated updated benchmark tables in data/paper_tables/:
  - table5b_literature_benchmark.csv — 24-row study-by-study comparison (supplementary)
  - table5c_ranking_agreement.csv — 8-row ranking agreement summary (main paper Section 3.3.2)
  - Result: 7/8 full agreement, 1 partial (Citrate vs EDTA for Pb)
- Compiled full reference list (~30+ references across all categories):
  - Geochemical Modeling: 3 refs (Parkhurst & Appelo, Allison et al., Dzombak & Morel)
  - Chelator Chemistry & Remediation: 8 refs (Lestan, Tandy, Hasegawa, Naghipour, Meers, Labanowski, Peters, Dermont)
  - Tier 2 Benchmark Studies: 6 refs (Naghipour, Tandy, Hasegawa, Labanowski, Meers, Kim)
  - ML for Soil Remediation: 5 refs (Barkhordari, Chen, Janga, Qiu, Zhang)
  - ML + Geochemical Modeling: 3 refs (Chang, Molina, Prasianakis)
  - Road Salt & Metal Speciation: 5 refs (Amrhein, Backstrom, Merrikhpour, Woosley, Zirino)
  - RI/NE Context: 6+ refs (Pouyat, Burman, Santschi, Sharma, Thompson, Thornton + RIDEM Tiverton page + EPA Superfund list)
- Key findings document saved to notes/key_findings_for_paper.md

### 5. Tier 2 Literature Benchmark Script Fixes
- Fixed multiple TypeError bugs in tier2_benchmark_final.py where pandas read CSV values as strings instead of numbers
- Added float() conversion with error handling to ALL mapping functions: map_ph, map_dose_mg_L, map_metal_level, map_ionic, om_to_doc
- Fixed column-shift bug in original literature CSV caused by commas in text fields — regenerated CSV using Python/pandas with proper quoting
- Working script: tier2_benchmark_final.py (in python_scripts/)

---

## KEY FINDINGS TO REMEMBER (saved in notes/key_findings_for_paper.md)

1. **Chelators can increase free Pb on average** — dragged down by Humic/Fulvic and low-dose/low-pH scenarios. Important practical finding: wrong chelator or underdosing makes things worse.
2. **Cu pH response is dramatic** — 49.1% free at pH 5.5 to 8.0% at pH 7.5 (6x reduction from pH alone). Liming may be more effective than chelation for Cu.
3. **Ionic strength effect on Cd is massive** — 76.1% free at low ionic to 19.0% at high ionic (57 pp reduction). Bigger than any chelator effect. RI coastal advantage.
4. **Texture effect is surprisingly small** — only 1-1.5 pp difference between Clay and Sand. Solution chemistry dominates over surface sorption in pore water.
5. **NTA is surprisingly competitive** — best for Cu and Zn, tied with EDTA for Cd. EDTA not always the best choice.
6. **Feature importance confirms real geochemistry** — Pb driven by pH/chelator/dose, Cu by pH/pe/chelator, Zn by chelator/dose/Cd concentration, Cd by Cl/Na/chelator (ionic strength dominates).

---

## WHAT'S NEXT: DAY 5 — WRITING SECTION 2.1

Start writing the Methods section. Day 5 covers Section 2.1 (PHREEQC Configuration & Parameter Design):
- 2.1.1: PHREEQC version, minteq.v4.dat database choice, justification vs alternatives (~200 words)
- Table 1 caption and finalize parameter space table
- pH as master variable paragraph (~200 words)
- Texture-DOC-Hfo coupling paragraph (~200 words)

Day 6 continues with:
- Ionic strength paragraph (RI-specific rationale)
- pe-moisture proxy paragraph
- Humic/fulvic modeling paragraph
- Section 2.1.3: Simulation execution

Full daily action plan is in notes/paper_daily_action_plan.md

---

## FILE LOCATIONS (current state)

### Paper Tables (in data/paper_tables/):
- table1_parameter_space.csv
- table3_summary_statistics.csv
- table4_model_performance.csv
- table5_tier1_validation.csv
- table5b_literature_benchmark.csv (updated — Kim replaces Alaboudi)
- table5c_ranking_agreement.csv (updated)
- tier1_validation_summary.csv

### Figures (in figures/):
- fig1_pipeline_schematic.svg
- fig2_ph_vs_free_metal_by_chelator.png/.pdf
- fig3_chelator_heatmap.png/.pdf
- fig4_dose_response.png/.pdf
- fig5_feature_importance.png/.pdf
- fig6_tier1_validation.png/.pdf
- Figure 7 (Streamlit screenshot) — NOT YET CREATED, do manually

### Reference Documents (in notes/):
- paper_daily_action_plan.md — Full 21-day checklist
- key_findings_for_paper.md — Surprising findings to highlight
- methodology_paper_outline.md — Detailed section-by-section outline
- chelator_project_roadmap.md — 10 goals with full context

### Data (in data/):
- complete_training_data_with_baseline.csv — MASTER dataset (12,636 rows)
- literature_benchmark_data.csv — 24 data points from 6 verified studies (updated)
- tier1_validation_report_v2.csv — Detailed Tier 1 results
- tier2_benchmark_results.csv — Tier 2 comparison results

### Scripts Used Today (in python_scripts/):
- day1_paper_tables.py / day1_paper_tables_v2.py
- day1_professional_tables.py
- day1_literature_tables.py
- day2_paper_figures.py
- update_lit_benchmark.py
- generate_lit_csv.py
- check_csv.py
- tier2_benchmark_final.py

---

## HOW TO RESUME

1. Open a new conversation with Claude
2. Share these project files: PROJECT_README.md, this session notes file, methodology_paper_outline.md, key_findings_for_paper.md, paper_daily_action_plan.md, chelator_project_roadmap.md
3. Say: "I'm working on the methodology paper for my chelator ML project. Days 1-4 are complete (tables, figures, references). I'm ready to start Day 5 — writing Section 2.1 (PHREEQC Configuration and Parameter Design). See the session notes for where we left off."
4. Claude will have full context to help write the Methods section

---

## KNOWN ISSUES / REMINDERS

1. Kim et al. (2003) extraction percentages (50% and 90%) are estimates — verify against actual paper before submission
2. Figure 7 (Streamlit screenshot) still needs to be created manually
3. The label encoder stores no-chelator as "nan" (not "None") — relevant if app code is touched
4. Figure 3 heatmap values differ from Day 1 summary table because heatmap filters to pH 7.0 only — this is correct
5. Hasegawa citation: verify this is the correct paper for the soil extraction data points used in the benchmark
6. AI disclosure statement needed in final paper acknowledgments

---
