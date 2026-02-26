# Geochemical Simulation-Trained Machine Learning for Chelator-Assisted Heavy Metal Remediation: A Decision-Support Framework

## Manuscript Outline for Journal of Hazardous Materials

### Authors
Mallory Malz [affiliation]

### Target Journal
Journal of Hazardous Materials (Impact Factor ~13.6)
Article type: Research Article
Word limit: ~8,000 words (excluding references, tables, figures)
Format: Introduction, Materials and Methods, Results and Discussion, Conclusions

---

## GRAPHICAL ABSTRACT (required by J Hazard Mater)

Visual summary showing the three-stage pipeline:
[Left panel] PHREEQC geochemical simulations (12,636 scenarios) with icons for pH, metals, chelators, soil types
[Center panel] Machine learning training (Gradient Boosting) with feature importance plot
[Right panel] Streamlit decision-support interface showing chelator recommendation output
Arrow across bottom: "Site parameters in → Chelator recommendation out"

---

## HIGHLIGHTS (required, 3-5 bullet points, max 85 characters each)

- 12,636 PHREEQC simulations generate ML training data for chelator selection
- Gradient Boosting predicts free metal fraction with R-squared above 0.99
- pH is the dominant control; chelator type and dose are key treatment variables
- Model validated against chemical logic rules and published extraction data
- Open-source Streamlit interface translates geochemistry to field decisions

---

## ABSTRACT (~250 words)

[Paragraph 1 — Problem]
Heavy metal contamination in urban and coastal soils poses persistent risks to human health and ecosystem function. Chelator-assisted remediation is widely used but chelator selection remains largely empirical, with practitioners choosing agents based on experience rather than site-specific geochemical analysis. No existing tool provides rapid, multi-metal, multi-chelator comparison across the range of soil conditions encountered in practice.

[Paragraph 2 — What we did]
We present a novel framework that couples thermodynamic geochemical simulation with machine learning to create a decision-support tool for chelator selection. Using PHREEQC with the minteq.v4.dat thermodynamic database, we generated 12,636 unique simulation scenarios spanning four priority metals (Pb, Cu, Zn, Cd), five chelating agents (EDTA, NTA, citrate, humic acid, fulvic acid), three dose levels, five pH values (5.5-7.5), three soil textures with corresponding surface complexation and organic carbon parameters, three moisture/redox conditions, three ionic strength levels reflecting coastal and road-salt-impacted conditions, and no-chelator baselines. Gradient Boosting regression models trained on simulation outputs predict percent free dissolved metal with R-squared values of 0.979-1.000 and RMSE of 0.15-0.83 percentage points across all four metals in cross-validation.

[Paragraph 3 — Key findings and significance]
Internal validation confirms the models obey fundamental geochemical rules, and comparison with published experimental data shows agreement on chelator rankings and metal difficulty trends. The framework is deployed as an open-source Streamlit interface where practitioners input site-specific soil parameters and receive chelator recommendations with predicted effectiveness. This approach — using simulation to systematically explore parameter space that would require thousands of laboratory experiments — represents a scalable methodology for translating geochemical knowledge into practical remediation guidance, with direct applicability to the contaminated coastal soils of Rhode Island and similar legacy-contaminated environments.

---

## 1. INTRODUCTION (~1,200 words)

### 1.1 The Problem: Heavy Metal Contamination in Soils

Opening paragraph establishing scope and significance. Heavy metal contamination from legacy industrial activity, leaded paint and gasoline, agricultural amendments, and atmospheric deposition affects millions of hectares of soil globally. In the United States alone, thousands of Superfund and brownfield sites contain elevated concentrations of Pb, Cu, Zn, and Cd that exceed risk-based screening levels. Coastal urban environments face compounded contamination from maritime and industrial activity layered over residential lead sources.

Rhode Island context paragraph. Rhode Island's coastal urban-industrial history has produced widespread soil contamination. The combination of legacy paint and plumbing lead, historic industrial activity, and ongoing road salt application creates a distinctive geochemical environment characterized by elevated metal concentrations in soils with high ionic strength pore water. This context motivates the inclusion of coastal salinity as a key parameter in the modeling framework.

### 1.2 Chelator-Assisted Remediation: Current State

Paragraph on chelation chemistry fundamentals. Chelating agents form thermodynamically stable complexes with dissolved metal ions, reducing the free (bioavailable) metal fraction in soil pore water and enhancing metal mobility for extraction. The effectiveness of chelation depends on the chelator's stability constants with target metals, competition from major cations (Ca2+, Mg2+, Fe3+), solution pH (which controls both chelator protonation state and metal hydroxide/carbonate speciation), and soil properties including organic matter content and mineral surface area.

Paragraph on current chelator selection practice and its limitations. In practice, chelator selection is often based on general recommendations (EDTA for Pb, citrate as a biodegradable alternative) or site-specific trial-and-error. The multi-dimensional nature of the problem — multiple metals, multiple chelators, site-variable pH, organic matter, texture, salinity — makes intuitive optimization difficult. A practitioner facing a site contaminated with Pb and Zn at pH 6.2 in a sandy coastal soil has no systematic way to determine whether EDTA, NTA, or citrate at what dose will minimize the free fraction of both metals simultaneously.

Paragraph on existing modeling approaches. PHREEQC and similar geochemical models can predict metal speciation under specific conditions, but running individual simulations for each site is time-consuming and requires specialized expertise. Machine learning has been applied to environmental remediation prediction, but most studies train on sparse experimental datasets that do not systematically cover the parameter space relevant to chelator selection. Review key prior work: ML for soil remediation (cite 3-4 papers), geochemical modeling for metal speciation (cite 3-4 papers), and the gap between them.

### 1.3 The Gap and Our Approach

Paragraph articulating the novel contribution. We address this gap by using geochemical simulation itself as the data generation engine for machine learning. Rather than relying on limited experimental datasets, we systematically simulate 12,636 unique combinations of metal concentrations, chelator types and doses, pH, soil texture, organic carbon, moisture/redox conditions, and ionic strength using PHREEQC with thermodynamically consistent parameters. The resulting dataset encodes the full complexity of aqueous metal speciation including chelator competition, surface complexation, and ion-activity effects across the parameter space relevant to field remediation. ML models trained on this dataset can then provide near-instantaneous predictions for any new combination of site parameters.

Closing paragraph stating objectives. The objectives of this study are to: (1) design and execute a systematic PHREEQC simulation framework spanning realistic soil remediation conditions, (2) train and validate ML models that faithfully reproduce thermodynamic speciation predictions, (3) verify model predictions against both internal chemical logic and published experimental data, and (4) deploy the trained models in a practitioner-accessible decision-support interface.

---

## 2. MATERIALS AND METHODS (~2,500 words)

### 2.1 Geochemical Modeling Framework

#### 2.1.1 PHREEQC Configuration

Software version (PHREEQC 3.5.0, USGS). Thermodynamic database selection: minteq.v4.dat chosen because it includes stability constants for EDTA, NTA, and citrate complexation with all target metals, as well as surface complexation parameters for iron hydroxide (Hfo) surfaces. Justify database choice vs. alternatives (phreeqc.dat lacks organic chelators, pitzer.dat better for high ionic strength but lacks surface complexation, llnl.dat has different parameterization).

#### 2.1.2 Parameter Space Design

**Table 1: Simulation Parameter Space**

| Parameter | Proxy for | Values | Rationale |
|-----------|----------|--------|-----------|
| pH | Soil acidity | 5.5, 6.0, 6.5, 7.0, 7.5 | Spans acidic to neutral range common in contaminated soils |
| Metal concentrations (Pb, Cu, Zn, Cd) | Contamination severity | Low, Medium, High (specific mg/L values) | Represents EPA screening level exceedances from 1x to 10x+ |
| Chelator type | Treatment agent | EDTA, NTA, Citrate, Humic acid, Fulvic acid | Industry standard plus biodegradable alternatives |
| Chelator dose | Treatment intensity | 50, 150, 300 mg/L | Sub-stoichiometric to excess relative to metal concentrations |
| Soil texture (Sand, Loam, Clay) | Surface area and sorption capacity | Hfo_wOH = 0.1, 0.5, 1.5 mol | Based on typical iron oxide content by texture class |
| DOC | Organic matter content | 10, 25, 40 mg/L | Tied to texture: sand=10, loam=25, clay=40 |
| pe | Redox/moisture condition | 12, 8, 3 | Dry (oxidizing), Mesic (moderate), Wet (reducing) |
| Na/Cl | Ionic strength | Low (100/150), Medium (500/700), High (2000/3000 mg/L) | Non-saline to coastal/road-salt-impacted |
| Ca/Mg | Competing cations | Low (20/10), High (100/50 mg/L) | Represent competition for chelator binding |
| No chelator | Baseline | dose = 0 | Required for calculating chelator effectiveness |

Discuss the proxy relationships in detail — this is where your thoughtfulness shows:

pH as master variable paragraph. Explain why pH controls both metal speciation (hydroxide, carbonate, chloride complex formation) and chelator protonation (EDTA transitions from H6EDTA2+ to EDTA4- across the pH range, with each protonation step reducing metal-binding capacity). Reference stability constant data.

Texture-DOC-Hfo coupling paragraph. Explain the decision to tie dissolved organic carbon to texture class rather than treating it as independent. In real soils, organic matter content correlates strongly with clay content and specific surface area. The Hfo_wOH surface sites represent generic iron/aluminum oxide surfaces available for metal sorption, parameterized using the generalized two-layer model in PHREEQC. Justify the specific values chosen.

Ionic strength as a RI-specific parameter paragraph. Explain why three ionic strength levels were included specifically to capture Rhode Island coastal conditions. High Na/Cl represents tidal influence and road salt runoff. Discuss the counterintuitive finding that high ionic strength can reduce free metal through chloride complexation (PbCl+, CuCl+) while simultaneously reducing surface sorption through competition.

pe as moisture proxy paragraph. Explain the conceptual link between soil moisture, oxygen diffusion, and redox potential. Dry soils (pe=12) are oxidizing with metals primarily as divalent cations. Wet soils (pe=3) are reducing and may promote sulfide precipitation. Mesic conditions (pe=8) represent typical field capacity. Acknowledge that pe is a simplified proxy for the complex redox zonation in real soils.

Humic/fulvic acid modeling paragraph. Acknowledge this is the most significant simplification. Minteq.v4.dat does not include explicit humic substance binding models (unlike WHAM or NICA-Donnan). Humic and fulvic acids were modeled as additional dissolved organic carbon (C(4) species), which captures the increase in solution-phase complexation capacity but does not represent the specific multi-dentate binding behavior of humic substances. Discuss implications for model accuracy.

#### 2.1.3 Simulation Execution

Total scenario count: 12,636 (12,150 original factorial design + 486 no-chelator baselines). PHREEQC input file structure (brief description with example in supplementary). Batch execution via Python-scripted pipeline. Output parsing with latin-1 encoding. Target variable extraction: percent free metal calculated as (free ion molality / total dissolved metal molality) x 100 for each of Pb2+, Cu2+, Zn2+, Cd2+. Secondary outputs: sorbed metal concentrations from surface complexation.

### 2.2 Machine Learning Pipeline

#### 2.2.1 Feature Engineering

Input features (20 total): numerical (pH, metal concentrations, DOC, Ca, Mg, Na, Cl, Hfo sites, pe, dose) and categorical (chelator type, texture, moisture condition, metal level, ionic level, Ca/Mg level). Categorical encoding using label encoding for tree-based models. Note the deliberate collinearity between paired features (texture/Hfo/DOC, moisture/pe, ionic_level/Na/Cl) — these represent the same physical property from different perspectives, and tree-based models handle this naturally through feature selection at each split.

#### 2.2.2 Model Selection and Training

Compared Random Forest and Gradient Boosting regression for each of the four metal targets independently. Gradient Boosting (scikit-learn GradientBoostingRegressor) selected based on superior cross-validated performance for all four metals. Hyperparameters (report key ones: n_estimators, max_depth, learning_rate, min_samples_split). Train/test split (80/20). 5-fold cross-validation on training set.

#### 2.2.3 Model Evaluation

Metrics: R-squared, RMSE, and cross-validated R-squared for each metal. Feature importance analysis using impurity-based importance from Gradient Boosting.

### 2.3 Validation Framework

#### 2.3.1 Tier 1: Internal Chemical Logic

Describe the eight chemical logic tests (Table 2 or supplementary). Explain the principle: if the ML model has learned the underlying geochemistry correctly, its predictions must obey fundamental thermodynamic and surface chemistry rules. This is a necessary (not sufficient) condition for model validity.

#### 2.3.2 Tier 2: Literature Benchmarking

Describe the literature comparison methodology. Acknowledge the fundamental measurement difference (our model predicts percent free ion in pore water at equilibrium; literature reports percent metal extracted from soil in batch experiments). Explain the mapping procedure for converting published conditions to model inputs. Focus comparison on directional agreement, relative chelator rankings, and metal difficulty ordering rather than absolute numerical match. Report number of studies, metals, chelators, and conditions covered.

### 2.4 Decision-Support Interface

Brief description of the Streamlit deployment. User inputs: pH, metal concentrations, organic matter percent (converted to DOC internally), soil texture (sand/silt/clay percentages with USDA classification), moisture condition, coastal/saline flag. Model outputs: predicted percent free for each metal with each chelator, recommended chelator and dose, effectiveness rating, warning flags for difficult conditions.

---

## 3. RESULTS AND DISCUSSION (~3,000 words)

### 3.1 PHREEQC Simulation Results

#### 3.1.1 Overview of the Training Dataset

Summary statistics of the 12,636-scenario dataset. Distribution of percent free metal across all conditions for each metal. Key finding: enormous variation in free metal fraction depending on conditions (e.g., Pb ranges from near 0% to over 90% free depending on pH, chelator, and soil properties).

**Table 3: Summary Statistics of Simulated Free Metal Fractions**

| Metal | Mean % Free | Std Dev | Min | Max | Median |
|-------|------------|---------|-----|-----|--------|
| Pb | 43.67 | | | | |
| Cu | 25.09 | | | | |
| Zn | 83.65 | | | | |
| Cd | 47.70 | | | | |

#### 3.1.2 pH as the Dominant Control

Present the pH-free metal relationship across all conditions. pH explains more variance in free metal than any other single variable. Discuss the mechanistic basis: increasing pH shifts metal speciation from free ions to hydroxide and carbonate complexes, increases chelator deprotonation (enhancing binding capacity), and increases surface charge on iron oxides (enhancing sorption). Include figure showing pH vs. free metal for each metal, stratified by chelator.

#### 3.1.3 Chelator Effectiveness Patterns

**Chelator ranking by metal:**
- Pb: Citrate > EDTA > NTA > Humic/Fulvic (discuss why citrate performs well for Pb in the model)
- Cu: EDTA > NTA > Citrate > Humic/Fulvic (Cu has very strong EDTA stability constant)
- Zn: All chelators show limited effectiveness (discuss Zn's weak complexation constants)
- Cd: EDTA > NTA > Citrate (moderate response)

Discuss dose-response relationships. Diminishing returns at high doses for Pb and Cu (already highly complexed); essentially no dose response for Zn (stability constants too low).

#### 3.1.4 Rhode Island-Specific Findings: Ionic Strength Effects

Present the counterintuitive finding that high ionic strength (coastal/road salt) reduces free Pb and Cu through chloride complexation. Discuss implications for RI practitioners: chloride-complexed metals (PbCl+, CuCl+) are not "free" by our definition but are still mobile and potentially bioavailable through different exposure pathways. This nuance is important for interpreting model outputs at coastal sites.

#### 3.1.5 Soil Texture and Surface Complexation Effects

Clay > Loam > Sand for metal immobilization through surface sorption. The tied DOC-texture relationship means clay soils also have more organic complexation in solution, creating a competing effect. Discuss the net outcome and conditions where sorption vs. solution complexation dominates.

### 3.2 Machine Learning Model Performance

**Table 4: Model Performance Metrics**

| Metal | R² (test) | CV R² (5-fold) | RMSE (%) | RF R² for comparison |
|-------|-----------|----------------|----------|---------------------|
| Pb | 0.9990 | 0.9788 | 0.83 | [value] |
| Cu | 0.9997 | 0.9481 | 0.59 | [value] |
| Zn | 0.9998 | 0.9972 | 0.33 | [value] |
| Cd | 1.0000 | 0.9999 | 0.15 | [value] |

Discuss why performance is so high: the training data comes from deterministic thermodynamic calculations with no measurement noise. The ML model is essentially learning a complex but deterministic function mapping inputs to outputs. The slightly lower cross-validated R² for Cu (0.9481) reflects more complex speciation behavior with stronger pH-chelator interactions.

#### 3.2.1 Feature Importance Analysis

**Figure: Feature importance for each metal model**

pH is the most important feature for all four metals. Chelator type and dose are the second and third most important treatment-related features. Texture/Hfo_sites important for metals with strong surface complexation (Pb, Cu). Ionic strength features (Na, Cl) important specifically for Pb (chloride complexation). Discuss how feature importance confirms known geochemical principles — the model has learned real chemistry, not artifacts.

### 3.3 Validation Results

#### 3.3.1 Tier 1: Chemical Logic Consistency

Report results of the eight-rule validation. Four rules pass at 100% (pH effect, dose response, texture effect, chelator-pH interaction). Four rules show minor violations at specific edge conditions with scientifically explainable causes. Discuss each edge case:
- Low-dose chelator at low pH can increase free metal through competitive desorption (known phenomenon)
- NTA outperforms EDTA for Pb at pH 5.5 due to differential protonation effects (consistent with literature)
- Zn can be easier to chelate than Cu specifically with humic/fulvic at low pH (Irving-Williams series effect with organic matter)
- Ionic strength effect reverses for specific no-chelator baseline conditions at high pH (minor, 1.7% of tests)

#### 3.3.2 Tier 2: Literature Comparison

Present comparison results with published experimental data (24 data points from 6 studies). Acknowledge the measurement-domain mismatch. Report: model correctly predicts chelator rankings for Pb (EDTA > NTA), metal difficulty order (Zn hardest, Cu easiest), and pH effects on chelator performance. Discuss the Zn discrepancy (model predicts high free Zn while literature shows moderate extraction) and potential explanations including kinetic effects in batch experiments vs. equilibrium predictions.

### 3.4 Practical Application: The Decision-Support Interface

Brief description with screenshot figure. Walk through an example case: a Rhode Island coastal site with pH 6.2, moderately contaminated (Pb 150 mg/kg, Cu 80 mg/kg, Zn 300 mg/kg), sandy loam, mesic conditions. Show the model's recommendation and explain the reasoning.

### 3.5 Limitations and Future Directions

**Equilibrium assumption.** PHREEQC calculates thermodynamic equilibrium, but real soil systems are kinetically controlled. Chelator degradation (especially citrate), slow mineral dissolution, and aging effects on metal extractability are not captured. Future integration with ORCHESTRA (for kinetic reactions) or HP1 (for reactive transport) would address this.

**Simplified organic matter modeling.** Humic and fulvic acids modeled as DOC additions rather than using specialized binding models (NICA-Donnan, WHAM). This underestimates the complexity of metal-organic matter interactions, particularly for Cu which has strong humic binding. ORCHESTRA integration would provide the NICA-Donnan model.

**Closed system assumption.** PHREEQC simulations model a closed batch system. Real soils are open systems with gas exchange (CO2, O2), water flow, plant uptake, and microbial activity. The multi-horizon mode planned for future versions would partially address vertical heterogeneity.

**Parameter range constraints.** The model is trained on specific parameter ranges (pH 5.5-7.5, three texture classes, etc.) and predictions outside these ranges are extrapolations. The interface includes boundary checks and warnings.

**Bench-scale validation needed.** Tier 3 validation using actual contaminated RI soil samples with laboratory chelator extraction experiments would provide the strongest evidence for model reliability. This experimental program is planned as a follow-up study.

---

## 4. CONCLUSIONS (~400 words)

[Paragraph 1] Summarize what was accomplished: a novel framework coupling geochemical simulation with machine learning for chelator selection in heavy metal remediation. The approach uses PHREEQC simulations as a systematic data generation engine, producing 12,636 training scenarios that encode thermodynamic metal speciation across realistic soil conditions.

[Paragraph 2] Summarize key findings: pH is the dominant control on chelator effectiveness, chelator rankings are metal-specific (EDTA best for Pb/Cu, limited effectiveness for Zn across all chelators), high ionic strength in coastal soils creates both opportunities (chloride complexation) and complications (reduced surface sorption), and the trained models faithfully reproduce these geochemical relationships with R-squared above 0.97 in cross-validation.

[Paragraph 3] Significance and forward look: this simulation-to-ML pipeline is transferable to other remediation contexts beyond chelation — any geochemical modeling scenario where practitioners need rapid predictions across variable site conditions. The deployed decision-support interface makes complex geochemistry accessible to field practitioners, potentially improving remediation outcomes by replacing empirical chelator selection with thermodynamically-grounded recommendations. Future work will incorporate reactive transport modeling, expand to additional metals (Hg, Cr, Ni) and chelators (GLDA, EDDHA), and validate against bench-scale experiments with contaminated Rhode Island soils.

---

## FIGURES LIST (plan for 5-7 figures)

1. **Schematic of the simulation-to-ML pipeline** (graphical abstract expanded)
2. **pH vs. percent free metal** for all four metals, faceted by chelator type (key result)
3. **Chelator comparison heatmap** showing mean percent free metal by chelator x metal at pH 7.0
4. **Dose-response curves** for each chelator-metal combination
5. **Feature importance bar charts** for each of the four metal models
6. **Tier 1 validation summary** showing pass rates for eight chemical logic rules
7. **Streamlit interface screenshot** showing example recommendation output

---

## TABLES LIST (plan for 4-5 tables)

1. Parameter space design (Section 2.1.2)
2. Summary statistics of training dataset (Section 3.1.1)
3. ML model performance metrics (Section 3.2)
4. Tier 1 validation results (Section 3.3.1)
5. Literature benchmark comparison summary (Section 3.3.2)

---

## SUPPLEMENTARY MATERIALS

- S1: Example PHREEQC input file with annotations
- S2: Complete parameter value table (all mg/L conversions)
- S3: Detailed Tier 1 validation results for all eight rules
- S4: Detailed literature benchmark comparison table
- S5: Link to GitHub repository with training data, model code, and Streamlit application

---

## KEY REFERENCES TO INCLUDE (starter list, expand to ~40-50)

### Geochemical Modeling
- Parkhurst & Appelo (2013) — PHREEQC v3 documentation (USGS)
- Allison et al. (1991) — MINTEQA2 thermodynamic database
- Dzombak & Morel (1990) — Surface complexation modeling

### Chelator Chemistry and Soil Remediation
- Lestan et al. (2008) — Review: chelating agents in metal-contaminated soil remediation. Environ Pollut.
- Tandy et al. (2004) — Biodegradable chelating agents comparison. Environ Sci Technol.
- Hasegawa et al. (2019) — Chelator-assisted washing for Pb, Cu, Zn. Appl Geochem.
- Naghipour et al. (2016) — EDTA and NTA extraction from sandy-loam soils.
- Meers et al. (2005) — EDTA and citric acid for enhanced phytoextraction.
- Labanowski et al. (2008) — EDTA vs citrate kinetic extractions. Environ Pollut.
- Peters (1999) — Chelant extraction of heavy metals: comprehensive review.
- Dermont et al. (2008) — Soil washing review: physical/chemical technologies.

### ML for Environmental Applications
- [Find 3-4 recent papers on ML for soil remediation prediction]
- [Find 2-3 papers on ML combined with geochemical modeling]

### Rhode Island / Coastal Contamination Context
- [Find 2-3 papers on RI or New England soil contamination]
- [Find 1-2 papers on road salt impacts on metal speciation]

---

## WRITING TIMELINE (suggested)

- Week 1: Fill in Introduction and Methods with full text
- Week 2: Generate all figures and tables from existing data/scripts
- Week 3: Write Results and Discussion
- Week 4: Write Abstract, Conclusions, compile references
- Week 5: Internal review, polish, prepare supplementary materials
- Week 6: Format for journal submission, write cover letter, submit

---

## NOTES FOR WRITING

### Voice and framing
- Write in third person passive for Methods ("Simulations were conducted using...")
- Use first person plural sparingly for interpretive statements ("We interpret this as...")
- Frame the tool as a complement to, not replacement for, professional judgment
- Be upfront about limitations — reviewers respect honesty more than overselling
- Emphasize the methodology (simulation-to-ML pipeline) as the transferable contribution, not just the specific chelator results

### Reviewer anticipation
Likely reviewer concerns and how to address them:
1. "Equilibrium assumption is unrealistic" → Acknowledge in limitations, cite that thermodynamic predictions provide upper-bound effectiveness estimates, discuss planned kinetic extensions
2. "No experimental validation" → Present Tier 1 and Tier 2 results, describe planned Tier 3 bench validation, frame current paper as methodology establishment
3. "Humic/fulvic modeling is too simple" → Acknowledge explicitly, discuss NICA-Donnan as future improvement, note that the framework is designed to incorporate improved sub-models
4. "How does this improve on just running PHREEQC?" → Speed (instant vs. hours), accessibility (no PHREEQC expertise needed), systematic comparison (all chelators simultaneously), and the ML model enables optimization/screening across the full parameter space
5. "Parameter ranges too narrow" → Justify ranges based on RI conditions and common contamination levels, note that the pipeline is easily extensible to broader ranges

### What makes this paper novel (emphasize these)
1. First systematic use of geochemical simulation as a training data generator for ML in remediation
2. Multi-metal (4), multi-chelator (5), multi-condition comparison — most studies examine one metal with one chelator
3. Explicit inclusion of coastal/saline conditions rarely addressed in chelation studies
4. The proxy system (pe for moisture, Hfo for texture, DOC tied to texture) makes environmental complexity tractable for ML while maintaining geochemical realism
5. End-to-end pipeline from simulation to deployed practitioner tool
