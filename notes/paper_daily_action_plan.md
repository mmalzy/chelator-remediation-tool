# Methodology Paper: Daily Action Plan & Checklist
## Target: Journal of Hazardous Materials Submission
## Start Date: February 25, 2026

---

## PHASE 1: GENERATE FIGURES, TABLES, AND MISSING DATA (Days 1-4)
Everything else is easier to write once you can see your results visually.

### Day 1 — Summary Statistics & Core Tables
- [ ] Run script to compute full summary statistics for Table 3
      (mean, std dev, min, max, median for each metal's % free)
- [ ] Pull exact hyperparameters from training_report.json for Table 4
      (n_estimators, max_depth, learning_rate, min_samples_split)
- [ ] Pull Random Forest R² values from training_report.json for Table 4 comparison column
- [ ] Compile Tier 1 validation results into clean Table format
      (rule name, description, pass rate, number tested, example violations)
- [ ] Save all table data as a single reference file: paper_tables.csv or paper_tables.md

### Day 2 — Core Figures (pH, Chelator Comparison)
- [ ] Generate Figure 2: pH vs. % free metal for all 4 metals, faceted by chelator type
      (box plots or line plots with error bands, 5 pH levels on x-axis)
- [ ] Generate Figure 3: Chelator comparison heatmap
      (rows = chelators including No Treatment, columns = metals, cells = mean % free at pH 7.0)
- [ ] Generate Figure 4: Dose-response curves
      (3 doses on x-axis, % free on y-axis, one panel per metal, lines for each chelator)
- [ ] Save all figures as both PNG (for drafting) and PDF or SVG (for submission)

### Day 3 — Feature Importance & Validation Figures
- [ ] Generate Figure 5: Feature importance bar charts (4 panels, one per metal)
      (already have PNGs in models/ folder — check if publication quality)
- [ ] Generate Figure 6: Tier 1 validation summary visual
      (bar chart showing pass rate for each of 8 rules, color-coded green/yellow)
- [ ] Generate Figure 7: Screenshot of Streamlit interface with example case
      (run the app, input a realistic RI scenario, take clean screenshot)
- [ ] Create Figure 1: Pipeline schematic (can be done in PowerPoint, Google Slides, or Figma)
      (three panels: PHREEQC simulations → ML training → Streamlit interface)

### Day 4 — Literature Benchmark Table & Cleanup
- [ ] Format Tier 2 results into clean Table 5 for the paper
      (study, metal, chelator, observed ranking, predicted ranking, agreement Y/N)
- [ ] Compile full reference list — start with the ~15 references in the outline
- [ ] Do targeted searches for missing references:
      - [ ] 3-4 papers: ML applied to soil remediation prediction
      - [ ] 2-3 papers: ML combined with geochemical modeling
      - [ ] 2-3 papers: RI or New England soil contamination
      - [ ] 1-2 papers: road salt impacts on metal speciation
- [ ] Record each reference with full citation info (authors, year, title, journal, volume, pages, DOI)

---

## PHASE 2: WRITE METHODS SECTION (Days 5-8)
Write Methods first — it's the easiest because you're describing what you did.

### Day 5 — Section 2.1: PHREEQC Configuration & Parameter Design
- [ ] Write 2.1.1: PHREEQC version, database choice, justification vs alternatives (~200 words)
- [ ] Write Table 1 caption and finalize the parameter space table
- [ ] Write the pH paragraph: why it's the master variable, protonation effects on chelators,
      hydroxide/carbonate speciation shifts (~200 words)
- [ ] Write the texture-DOC-Hfo coupling paragraph: why tied together, real-soil correlation,
      generalized two-layer model, specific values chosen (~200 words)

### Day 6 — Section 2.1 continued: Proxies and Simulation Execution
- [ ] Write the ionic strength paragraph: RI coastal rationale, three levels,
      counterintuitive chloride complexation finding (~200 words)
- [ ] Write the pe-moisture proxy paragraph: oxygen diffusion link, three conditions,
      acknowledge simplification (~150 words)
- [ ] Write the humic/fulvic modeling paragraph: what we did, why, limitations,
      NICA-Donnan as future improvement (~200 words)
- [ ] Write 2.1.3: Simulation execution — scenario count, batch pipeline,
      output parsing, target variable definition (~200 words)

### Day 7 — Sections 2.2 and 2.3: ML Pipeline & Validation
- [ ] Write 2.2.1: Feature engineering — 20 features, encoding, deliberate collinearity (~200 words)
- [ ] Write 2.2.2: Model selection — RF vs GB comparison, why GB won,
      hyperparameters, train/test split, cross-validation (~250 words)
- [ ] Write 2.2.3: Evaluation metrics — R², RMSE, CV R² (~100 words)
- [ ] Write 2.3.1: Tier 1 chemical logic — 8 rules, principle, tolerance thresholds (~250 words)
- [ ] Write 2.3.2: Tier 2 literature — methodology, measurement mismatch acknowledgment,
      mapping procedure, what we compared (~250 words)

### Day 8 — Section 2.4: Interface & Methods Review
- [ ] Write 2.4: Streamlit interface description — inputs, outputs, conversion logic (~200 words)
- [ ] Read through entire Methods section start to finish
- [ ] Check: does every parameter choice have a stated rationale?
- [ ] Check: are all numbers accurate (verify against actual data files)?
- [ ] Check: consistent units throughout (mg/L, mol/L, etc.)?

---

## PHASE 3: WRITE RESULTS AND DISCUSSION (Days 9-13)

### Day 9 — Section 3.1: PHREEQC Results Overview
- [ ] Write 3.1.1: Dataset overview with Table 3, variation in free metal (~300 words)
- [ ] Write 3.1.2: pH as dominant control, reference Figure 2, mechanistic explanation (~400 words)

### Day 10 — Sections 3.1.3-3.1.5: Chelator, Ionic, Texture Effects
- [ ] Write 3.1.3: Chelator effectiveness patterns by metal, dose-response,
      reference Figures 3 and 4 (~400 words)
- [ ] Write 3.1.4: Ionic strength effects — counterintuitive finding,
      RI implications, chloride complexation nuance (~300 words)
- [ ] Write 3.1.5: Texture and surface complexation, competing DOC effect (~200 words)

### Day 11 — Section 3.2: ML Performance
- [ ] Write 3.2: Model performance with Table 4, explain why R² so high,
      discuss Cu complexity (~300 words)
- [ ] Write 3.2.1: Feature importance analysis, reference Figure 5,
      connect to known geochemistry (~300 words)

### Day 12 — Section 3.3: Validation Results
- [ ] Write 3.3.1: Tier 1 results — 4 perfect passes, 4 edge-case "failures" with
      scientific explanations for each, reference Table and Figure 6 (~500 words)
- [ ] Write 3.3.2: Tier 2 results — what agreed (rankings, difficulty order, pH effects),
      what didn't (Zn magnitude), measurement domain mismatch discussion (~400 words)

### Day 13 — Sections 3.4-3.5: Interface & Limitations
- [ ] Write 3.4: Interface walkthrough with example RI case,
      reference Figure 7 screenshot (~300 words)
- [ ] Write 3.5: Limitations — equilibrium assumption, simplified OM, closed system,
      parameter range constraints, bench validation needed (~500 words)
      Frame each limitation with "what we plan to do about it"

---

## PHASE 4: WRITE INTRO, ABSTRACT, CONCLUSIONS (Days 14-16)
Write these last — easier once you know exactly what's in the paper.

### Day 14 — Introduction
- [ ] Write 1.1: Problem scope — contamination globally, in US, in RI specifically (~300 words)
- [ ] Write 1.2: Chelation chemistry fundamentals, current practice limitations (~400 words)
- [ ] Write 1.2 continued: Existing modeling and ML approaches, cite prior work (~300 words)
- [ ] Write 1.3: The gap, our novel approach, objectives statement (~300 words)

### Day 15 — Abstract & Conclusions
- [ ] Write Conclusions: 3 paragraphs summarizing accomplishment, key findings,
      significance and future work (~400 words)
- [ ] Write Abstract: 3 paragraphs — problem, what we did, key findings (~250 words)
- [ ] Write Highlights: 5 bullet points, each under 85 characters
- [ ] Draft Graphical Abstract concept (sketch or describe for later design)

### Day 16 — Supplementary Materials
- [ ] Prepare S1: Example PHREEQC input file with line-by-line annotations
- [ ] Prepare S2: Complete parameter value table with all unit conversions
- [ ] Prepare S3: Full Tier 1 validation details (all 8 rules, all violations)
- [ ] Prepare S4: Full Tier 2 literature comparison table
- [ ] Prepare S5: GitHub repository (or plan for it) — training data, model code, app code

---

## PHASE 5: POLISH AND SUBMIT (Days 17-21)

### Day 17 — Full Read-Through & Internal Review
- [ ] Print or export the full manuscript and read it straight through
- [ ] Mark any sections that feel weak, unclear, or repetitive
- [ ] Check all figure/table references in text match actual figure/table numbers
- [ ] Verify all numbers in text match the actual data

### Day 18 — Revisions
- [ ] Address all issues marked during read-through
- [ ] Ensure consistent terminology throughout (% free metal, not switching terms)
- [ ] Check that every figure and table is referenced in the text
- [ ] Verify reference list is complete — every in-text citation has a full reference

### Day 19 — Final Figures & Formatting
- [ ] Finalize all figures at publication resolution (300+ DPI, PDF or TIFF)
- [ ] Create the Graphical Abstract (required by J Hazard Mater)
- [ ] Format manuscript per J Hazard Mater author guidelines
      (check their Guide for Authors for specific requirements: font, spacing, heading style)
- [ ] Add AI disclosure statement to Acknowledgments
- [ ] Add Data Availability statement (link to GitHub repo or "available on request")

### Day 20 — Cover Letter & Final Check
- [ ] Write cover letter to editor (~300 words):
      - Why this paper fits J Hazard Mater
      - The 3 novel contributions
      - Statement that it's not under review elsewhere
- [ ] Final proofread of entire manuscript
- [ ] Check all author information, affiliations, contact details
- [ ] Confirm supplementary files are formatted correctly

### Day 21 — Submit
- [ ] Create account on Elsevier Editorial Manager (if not already)
- [ ] Upload manuscript, figures, supplementary materials, cover letter
- [ ] Select article type, keywords, suggest reviewers (optional but helpful)
- [ ] Submit and celebrate

---

## WHAT TO BRING TO EACH CLAUDE SESSION

When you sit down to work on a section, share:
1. This checklist (so I know where you are)
2. The methodology_paper_outline.md (in project files, I already have it)
3. What you want to work on today: "I'm on Day 6, writing the ionic strength paragraph"
4. Any draft text you've written that you want me to help refine

I can help you:
- Draft paragraphs based on your data and the outline
- Generate figures and tables from your training data
- Find and format specific references
- Review and tighten your writing
- Check scientific accuracy of claims

---

## QUICK REFERENCE: WORD COUNT TARGETS

| Section | Target Words |
|---------|-------------|
| Abstract | 250 |
| Introduction | 1,200 |
| Methods | 2,500 |
| Results & Discussion | 3,000 |
| Conclusions | 400 |
| **Total main text** | **~7,350** |
| Figure captions | ~500 |
| Table captions | ~300 |
| **Grand total** | **~8,000** |

This is well within J Hazard Mater's guidelines.

---

## MOTIVATION TRACKER

| Phase | Days | Status |
|-------|------|--------|
| Phase 1: Figures & Tables | 1-4 | [ ] |
| Phase 2: Methods | 5-8 | [ ] |
| Phase 3: Results & Discussion | 9-13 | [ ] |
| Phase 4: Intro, Abstract, Conclusions | 14-16 | [ ] |
| Phase 5: Polish & Submit | 17-21 | [ ] |
