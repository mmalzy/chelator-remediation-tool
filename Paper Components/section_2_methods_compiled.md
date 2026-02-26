# Section 2: Materials and Methods

---

## 2.1 Geochemical Modeling Framework

### 2.1.1 PHREEQC Configuration

Aqueous geochemical speciation was simulated using PHREEQC version 3.5.0 (Parkhurst and Appelo, 2013), a widely validated thermodynamic modeling code developed by the U.S. Geological Survey. PHREEQC calculates the equilibrium distribution of dissolved species, surface-complexed species, and mineral saturation states for user-defined solution compositions. Among the available thermodynamic databases, minteq.v4.dat was selected for all simulations because it includes stability constants for metal complexes with the chelating agents central to this study — specifically EDTA, NTA, and citrate — as well as parameters for the generalized two-layer surface complexation model for hydrous ferric oxide (Hfo) surfaces (Dzombak and Morel, 1990). Alternative databases were considered but rejected for specific reasons: phreeqc.dat lacks organic chelator species entirely; pitzer.dat provides superior activity coefficient calculations at high ionic strength but does not include surface complexation parameters; and llnl.dat uses a different parameterization framework that is less well-validated for the chelator–metal systems of interest (Allison et al., 1991). The minteq.v4.dat database represents a practical compromise, offering the broadest coverage of chelator–metal thermodynamic data with internally consistent surface complexation parameters within a single database.

### 2.1.2 Parameter Space Design

The simulation parameter space was designed to span the range of environmental conditions encountered at contaminated sites in Rhode Island and similar coastal urban-industrial environments (Table 1). Twenty input parameters were varied in a structured factorial design, producing 12,150 unique chelator-treatment scenarios plus 486 no-chelator baseline scenarios (12,636 total simulations). Parameters were selected to represent the key geochemical controls on metal speciation in soil pore water: solution pH, metal contamination severity, chelator identity and dose, soil texture and associated surface properties, soil moisture and redox conditions, ionic strength, and competing cation concentrations. Several parameters were coupled to reflect real soil correlations rather than treated as independent variables, as described below.

[TABLE 1 PLACEMENT — Table 1: Simulation parameter space. Ten input parameters, their environmental proxy meaning, discrete values, and scientific rationale.]

**pH as the master variable.** Solution pH was varied across five levels (5.5, 6.0, 6.5, 7.0, and 7.5), spanning the range commonly observed in contaminated urban and agricultural soils. pH exerts dominant control over metal speciation through multiple simultaneous mechanisms. First, increasing pH shifts metal partitioning from free aquo ions (e.g., Pb²⁺, Cu²⁺) toward hydroxide and carbonate complexes (e.g., PbOH⁺, PbCO₃⁰, Cu(OH)₂⁰), which directly reduces the free ion fraction. Second, pH controls the protonation state of chelating agents: EDTA transitions from the weakly metal-binding H₆EDTA²⁺ at low pH to the strongly binding EDTA⁴⁻ at circumneutral pH, with each successive deprotonation step increasing the effective metal-binding capacity (Lestan et al., 2008). Third, the surface charge of iron and aluminum oxide minerals becomes increasingly negative with rising pH, enhancing electrostatic attraction for cationic metal species and increasing sorption capacity (Dzombak and Morel, 1990). These three pH-dependent processes act in concert, making pH the single most influential variable governing the effectiveness of chelator-assisted remediation.

**Soil texture, dissolved organic carbon, and surface site coupling.** Rather than treating soil texture, dissolved organic carbon (DOC), and iron oxide surface site density as independent variables, these three parameters were coupled to reflect their well-established covariation in natural soils. Clay-rich soils contain more organic matter (and thus higher pore water DOC) and more iron/aluminum oxide mineral surfaces than sandy soils (Brady and Weil, 2017). Three texture classes were defined — Sand, Loam, and Clay — with each class assigned a corresponding DOC concentration (10, 25, and 40 mg/L, respectively) and iron oxide surface site density expressed as moles of Hfo_wOH (0.1, 0.5, and 1.5 mol, respectively). Surface complexation was modeled using the generalized two-layer model with default Hfo parameters from Dzombak and Morel (1990): a specific surface area of 600 m² g⁻¹ and a site density corresponding to the weak binding sites (Hfo_wOH). The Hfo_wOH values were parameterized to represent the total reactive surface area available for metal sorption in each texture class, encompassing contributions from goethite, ferrihydrite, and amorphous iron oxyhydroxides that are common in weathered temperate soils (Dzombak and Morel, 1990; Cornell and Schwertmann, 2003). This coupling approach introduces deliberate collinearity between texture, DOC, and Hfo_sites in the training data — a design choice that reflects physical reality and that tree-based machine learning models handle naturally through feature selection at individual decision splits, without the instability that collinearity introduces in linear models.

**Ionic strength as a Rhode Island–specific parameter.** Three levels of ionic strength were included in the simulation design, defined by paired sodium and chloride concentrations: low (Na = 100, Cl = 150 mg/L), medium (Na = 500, Cl = 700 mg/L), and high (Na = 2000, Cl = 3000 mg/L). The high ionic strength level represents conditions found in coastal soils subject to tidal influence and in roadside soils receiving deicing salt applications, both of which are widespread in Rhode Island. Road salt has been shown to mobilize heavy metals from soil through competitive displacement of sorbed cations and formation of soluble chloride complexes (Amrhein et al., 1992). In the PHREEQC simulations, elevated chloride concentrations lead to the formation of metal–chloride species (e.g., PbCl⁺, PbCl₂⁰, CdCl⁺, CdCl₂⁰), which reduce the free metal ion fraction but produce mobile complexes that remain in solution. This effect is particularly pronounced for Cd, which forms strong chloride complexes at concentrations relevant to coastal and road-salt-impacted environments. Including three ionic strength levels allows the model to capture both the direct effects of salinity on speciation and the interaction between ionic strength and chelator performance, which is relevant to practitioners working in the distinctive geochemical setting of coastal New England.

**Electron activity as a moisture and redox proxy.** The redox state of soil pore water was represented by the electron activity parameter pe, set at three levels: 12 (dry, oxidizing), 8 (mesic, moderately reducing), and 3 (wet, strongly reducing). These values correspond to soils where oxygen diffusion is uninhibited (well-drained sandy soils at field capacity), partially restricted (loamy soils near saturation), and severely limited (waterlogged or seasonally flooded conditions), respectively. Under reducing conditions (low pe), reductive dissolution of iron and manganese oxyhydroxides can release sorbed metals, while sulfide mineral precipitation may immobilize certain metals, particularly Cu and Pb (Dzombak and Morel, 1990). The pe parameter in PHREEQC sets the initial equilibrium redox state and influences the relative abundance of oxidized and reduced species for redox-active elements. This is an acknowledged simplification: real soils contain redox microgradients at the aggregate scale, and pe does not capture these spatial heterogeneities. Nevertheless, the three-level parameterization captures the first-order effect of moisture-driven redox variation on metal speciation.

**Humic and fulvic acid modeling.** Humic and fulvic acids were included as chelator options to represent the metal-binding capacity of natural dissolved organic matter. The minteq.v4.dat database does not contain explicit humic substance binding models comparable to the NICA-Donnan model (Kinniburgh et al., 1996) or Model VI/VII of the Windermere Humic Aqueous Model (WHAM) (Tipping, 1998). In the absence of these specialized frameworks, humic and fulvic acid treatments were modeled as additions of dissolved organic carbon to the solution, implemented as increased C(4) concentration in the PHREEQC SOLUTION block. This approach captures the general increase in solution-phase complexation capacity associated with organic matter but does not represent the heterogeneous, multi-dentate binding behavior characteristic of real humic substances. Fulvic acid additions were scaled to 80% of the equivalent humic acid dose to reflect the generally lower molecular weight and reduced binding capacity of fulvic acids relative to humic acids. This modeling simplification is the most significant limitation of the current simulation framework and likely underestimates the true metal-binding capacity of natural organic matter, particularly for Cu, which forms exceptionally strong complexes with humic carboxyl and phenolic groups. Integration of the NICA-Donnan model through coupling with the ORCHESTRA modeling framework is planned as a future improvement to address this limitation.

### 2.1.3 Simulation Execution

The 12,636 PHREEQC input files were generated programmatically using Python scripts that iterated over the factorial parameter combinations and wrote correctly formatted input files with element concentrations converted from mg/L to mol/L using standard atomic and molecular weights (Pb: 207.2, Cu: 63.55, Zn: 65.38, Cd: 112.41 g/mol; EDTA: 292.24, NTA: 191.14, citrate: 189.1 g/mol). Each simulation was executed in batch mode with the command-line syntax: phreeqc [input] [output] minteq.v4.dat. Output files were parsed using Python with latin-1 character encoding to extract the molality of each dissolved metal species. The percent free metal fraction was calculated for each target metal as (free ion molality / total dissolved metal molality) × 100, where the free ion species are Pb²⁺, Cu²⁺, Zn²⁺, and Cd²⁺. Secondary output variables included the moles of each metal sorbed to Hfo surface sites, extracted from the SURFACE block of the PHREEQC output. All 12,636 simulations converged successfully with no numerical failures. The resulting dataset was assembled into a single CSV file (12,636 rows × 36 columns) comprising 20 input features and 8 target variables (4 percent free metal values and 4 sorbed metal values).

---

## 2.2 Machine Learning Pipeline

### 2.2.1 Feature Engineering

The PHREEQC simulation outputs were organized into a tabular dataset with 12,636 rows (one per simulation scenario) and 14 input features mapped to four continuous target variables (percent free Pb²⁺, Cu²⁺, Zn²⁺, and Cd²⁺). Thirteen features were numeric: pH, individual metal concentrations (Pb, Cu, Zn, and Cd in mg/L), dissolved organic carbon (mg/L), competing cation concentrations (Ca and Mg in mg/L), ionic strength proxies (Na and Cl in mg/L), chelator dose (mg/L), iron oxide surface site density (Hfo_wOH in mol), and pe. One feature — chelator type — was categorical, with six levels (EDTA, NTA, Citrate, Humic, Fulvic, and no chelator) encoded as integer labels using scikit-learn's LabelEncoder.

The feature set contains deliberate collinearity between certain pairs: soil texture class determines both iron oxide surface site density and dissolved organic carbon concentration (Sand maps to Hfo = 0.1 mol and DOC = 10 mg/L; Loam to 0.5 and 25; Clay to 1.5 and 40), moisture condition determines pe (Dry = 12, Mesic = 8, Wet = 3), and ionic strength level determines Na and Cl concentrations. The categorical labels (texture, moisture, ionic level) were excluded from the feature set because tree-based models can learn equivalent decision boundaries from the underlying numeric values, and including both would create perfectly collinear feature pairs without adding information. The numeric representations were retained because they preserve the quantitative relationships (e.g., the threefold difference in Hfo between Loam and Clay) that categorical labels would obscure.

### 2.2.2 Model Selection and Training

Two ensemble regression algorithms were compared for each target metal independently: Random Forest and Gradient Boosting, both implemented in scikit-learn version 1.3 (Pedregosa et al., 2011). Random Forest hyperparameters were set at 200 trees, maximum depth of 20, minimum samples to split a node of 5, minimum samples per leaf of 2, and the square root of the number of features considered at each split. Gradient Boosting hyperparameters were 200 boosting stages, maximum depth of 6, learning rate of 0.1, minimum samples to split of 5, minimum samples per leaf of 2, and stochastic subsampling of 80% of the training data per stage to reduce overfitting. All models used a fixed random seed of 42 for reproducibility. No hyperparameter tuning (grid search or Bayesian optimization) was performed; the selected values represent established defaults for moderate-sized tabular regression problems and were chosen to prioritize reproducibility and interpretability over marginal performance gains.

The dataset was split into 80% training (10,109 rows) and 20% test (2,527 rows) using stratification-free random sampling. Both algorithms were trained on the training set, and generalization performance was assessed through 5-fold cross-validation on the full dataset using R² as the scoring metric. The winning algorithm for each metal was selected based on the higher mean cross-validated R², which provides a more robust estimate of generalization than test-set R² alone. Gradient Boosting was selected for all four metals.

### 2.2.3 Model Evaluation

Model performance was assessed using four metrics. R² (coefficient of determination) on the held-out test set measures the proportion of variance explained by the model, where values approaching 1.0 indicate near-perfect prediction. Root mean squared error (RMSE) quantifies the average prediction error in the same units as the target variable (percentage points of free metal), penalizing large errors more heavily than small ones. Mean absolute error (MAE) provides a complementary measure of average prediction error without the squaring penalty. Five-fold cross-validated R² (CV R²) and its standard deviation across folds assess generalization stability: a CV R² close to the test-set R² indicates that the model is not overfitting to the particular train/test partition, while high variance across folds would suggest sensitivity to the data split.

Feature importance was extracted using the impurity-based method native to Gradient Boosting ensembles, in which the importance of each feature is calculated as the total reduction in the loss function (mean squared error) attributable to splits on that feature, normalized across all trees in the ensemble. While impurity-based importance can overweight high-cardinality features in some contexts, this concern is minimal here because all numeric features operate on comparable scales and the single categorical feature has only six levels. The feature importance rankings serve as a scientific consistency check: if the ML models have correctly learned the underlying geochemical relationships encoded in the PHREEQC simulations, pH should emerge as the dominant feature (consistent with its role as the master variable in aqueous metal speciation), followed by chelator-related features (type and dose) and soil properties (surface sites, ionic strength proxies) in an order consistent with known thermodynamic controls.

---

## 2.3 Validation Framework

Model validation followed a two-tier framework designed to assess both internal consistency and external plausibility.

### 2.3.1 Tier 1: Internal Chemical Logic

The first validation tier tests whether model predictions obey fundamental geochemical principles that must hold regardless of the specific conditions being modeled. Eight chemical logic rules were formulated based on established thermodynamic relationships: (1) increasing pH should decrease the percent free metal fraction, because hydroxide and carbonate complexation increases with pH; (2) the addition of any chelator should reduce the free metal fraction relative to the no-chelator baseline; (3) increasing chelator dose should monotonically decrease the free metal fraction; (4) increasing surface site density (Clay > Loam > Sand) should decrease the free fraction through enhanced sorption; (5) EDTA should outperform NTA for Pb and Cu, reflecting the higher thermodynamic stability constants of EDTA complexes with these metals; (6) Zn should be harder to chelate than Cu across all conditions, consistent with the lower stability constants of Zn-chelator complexes relative to Cu; (7) chelator effectiveness should increase at higher pH, because hydrogen ion competition for chelator binding sites decreases; and (8) high ionic strength should reduce free Pb and Cu through chloride complex formation, a prediction specific to the coastal Rhode Island conditions represented in the training data.

Each rule was tested across all relevant condition pairs in the training dataset using tolerance thresholds ranging from 0.5 to 3.0 percentage points (depending on the rule) to accommodate minor numerical noise from the thermodynamic solver. Rule violations — cases where the data contradicts the expected direction — were recorded with their frequency and the specific conditions under which they occurred. A rule was considered to pass if the violation rate fell below 5% of tested pairs, with the expectation that minor violations at boundary conditions (e.g., very low chelator doses at low pH) may reflect real geochemical edge cases rather than model errors.

### 2.3.2 Tier 2: Literature Benchmarking

The second validation tier compares model predictions against published experimental results from peer-reviewed chelator-assisted soil remediation studies. Published experimental conditions were mapped to the model's input feature space using defined conversion rules: soil texture descriptions were classified as Sand, Loam, or Clay; organic matter content (percent by mass) was converted to dissolved organic carbon using the van Bemmelen factor (DOC ≈ OM% × 5.8, assuming organic matter is approximately 58% carbon; Brady and Weil, 2017) and assigned to the nearest training level (10, 25, or 40 mg/L); chelator doses reported in various units (mmol/kg, g/L, mM) were converted to mg/L and mapped to the closest training dose (50, 150, or 300 mg/L); and ionic strength was estimated from reported electrical conductivity or sodium concentrations.

A fundamental measurement-domain mismatch exists between the model's predictions and most published extraction data. The model predicts the percent of dissolved metal present as free (uncomplexed) ions at thermodynamic equilibrium in pore water, while published studies typically report the percent of total soil metal extracted into solution during batch washing experiments. These quantities are related — chelators that reduce the free ion fraction in solution also tend to extract more metal from soil — but they are not numerically equivalent, because extraction efficiency also depends on kinetic factors (contact time, desorption rates from aged contamination), physical factors (liquid-to-solid ratio, agitation), and processes not captured in equilibrium speciation models (mineral dissolution, colloid mobilization). The literature comparison therefore focuses on three forms of agreement that are robust to this measurement mismatch: directional agreement (does the chelator reduce metal availability relative to the control?), relative ranking (does the model correctly predict which chelator performs best for a given metal and soil?), and metal difficulty ordering (does the model correctly rank which metals are most and least responsive to chelation?). Exact numerical correspondence between predicted percent free and observed percent extracted is not expected and is not used as a validation criterion.

---

## 2.4 Decision-Support Interface

The trained Gradient Boosting models were deployed in an interactive web application built with Streamlit, an open-source Python framework for data-driven applications. The interface is designed for use by environmental practitioners, remediation engineers, and regulatory staff who may not have expertise in geochemical modeling or machine learning, and therefore translates all model inputs and outputs into terms familiar to field practitioners.

Users provide site characterization data through the following inputs: soil pH (numeric slider, range 5.0–8.5); total concentrations of Pb, Cu, Zn, and Cd in mg/kg; soil organic matter as a percentage by mass, which the application converts internally to dissolved organic carbon using the van Bemmelen factor (DOC ≈ OM% × 5.8); soil texture specified as sand, silt, and clay percentages, which the application classifies into USDA texture classes and maps to the corresponding surface site density and DOC levels used in the training data; moisture condition (Dry, Mesic, or Wet); and a coastal or saline conditions flag that sets the ionic strength level. Where user-provided values fall between the discrete levels used in the training data (e.g., a pH of 6.3 or a DOC of 18 mg/L), the model accepts the continuous numeric input directly, as the Gradient Boosting models can interpolate between training points.

For each set of site parameters, the application runs predictions across all five chelator types (EDTA, NTA, Citrate, Humic acid, Fulvic acid) at all three dose levels (50, 150, and 300 mg/L), as well as a no-treatment baseline. The outputs include the predicted percent free dissolved metal for each of the four target metals under each treatment scenario, a recommended chelator and dose combination that minimizes the free metal fraction across all metals of concern, an effectiveness rating (Excellent, Good, Moderate, or Poor) based on the predicted reduction relative to baseline, and a full comparison table allowing practitioners to evaluate tradeoffs across chelators and doses. The interface also generates context-specific warning flags: a warning when Zn concentrations are elevated (because all chelators show limited effectiveness for Zn), a warning when pH is below 6.0 (because chelator performance decreases substantially at low pH), and a warning when high ionic strength is detected (because chloride-complexed metals, while not "free," remain mobile in solution).

The application and all associated code, trained models, and training data are intended for release as open-source supplementary materials accompanying this publication.

---

# REFERENCE LIST (Verified Citations — Running)

## Section 2.1 Citations

Allison, J.D., Brown, D.S., Novo-Gradac, K.J., 1991. MINTEQA2/PRODEFA2, A Geochemical Assessment Model for Environmental Systems: Version 3.0 User's Manual. EPA/600/3-91/021, U.S. Environmental Protection Agency, Athens, GA.

Amrhein, C., Strong, J.E., Mosher, P.A., 1992. Effect of deicing salts on metal and organic matter mobilization in roadside soils. Environmental Science & Technology 26(4), 703–709. https://doi.org/10.1021/es00028a006

Brady, N.C., Weil, R.R., 2017. The Nature and Properties of Soils, 15th ed. Pearson, Upper Saddle River, NJ.

Cornell, R.M., Schwertmann, U., 2003. The Iron Oxides: Structure, Properties, Reactions, Occurrences and Uses, 2nd ed. Wiley-VCH, Weinheim.

Dzombak, D.A., Morel, F.M.M., 1990. Surface Complexation Modeling: Hydrous Ferric Oxide. John Wiley & Sons, New York.

Kinniburgh, D.G., Milne, C.J., Benedetti, M.F., Pinheiro, J.P., Filius, J., Koopal, L.K., Van Riemsdijk, W.H., 1996. Metal ion binding by humic acid: Application of the NICA-Donnan model. Environmental Science & Technology 30(5), 1687–1698. https://doi.org/10.1021/es950695h

Lestan, D., Luo, C., Li, X., 2008. The use of chelating agents in the remediation of metal-contaminated soils: A review. Environmental Pollution 153(1), 3–13. https://doi.org/10.1016/j.envpol.2007.11.015

Parkhurst, D.L., Appelo, C.A.J., 2013. Description of Input and Examples for PHREEQC Version 3 — A Computer Program for Speciation, Batch-Reaction, One-Dimensional Transport, and Inverse Geochemical Calculations. U.S. Geological Survey Techniques and Methods, Book 6, Chapter A43, 497 p. https://doi.org/10.3133/tm6A43

Tipping, E., 1998. Humic Ion-Binding Model VI: An improved description of the interactions of protons and metal ions with humic substances. Aquatic Geochemistry 4(1), 3–47. https://doi.org/10.1023/A:1009627214459

## Section 2.2 Citations

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E., 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2825–2830.

## Verified Citations for Introduction / Discussion (Not Yet Placed In-Text)

Barkhordari, M.S., et al., 2024. XGBoost for electrokinetic remediation efficiency prediction. Journal of Environmental Chemical Engineering 12(6), 114330.

Chang, S., et al., 2023. PHREEQC-informed Random Forest for U(VI) adsorption onto minerals. Applied Geochemistry 155, 105731.

Chen, S., et al., 2024. ML for thermal desorption of PAH-contaminated soils. Science of the Total Environment 927, 172173.

Janga, B., et al., 2023. Review of AI/ML/DL in contaminated site remediation. Chemosphere 345, 140476.

Molina, O., et al., 2025. PHREEQC-informed ML for scaling indices in Permian Basin produced waters. URTEC conference paper D021S034R003. https://doi.org/10.15530/urtec-2025-4265270

Prasianakis, N.I., et al., 2025. Geochemistry and ML benchmarking for reactive transport. Environmental Earth Sciences 84(5), 121.

Qiu, Y., et al., 2025. ML prediction of heavy metal extraction by leaching agents. Journal of Environmental Chemical Engineering 14(1), 120716.

Zhang, Y., et al., 2024. ML-assisted screening of soil remediation strategies. Processes 12(6), 1157.

---

# FORMATTING NOTES

- J Hazard Mater uses numbered references [1], [2] in order of appearance. Author-year format used during drafting for readability; conversion in single pass at final assembly.
- Table 1 content defined but not typeset here (will be formatted in final .docx).
- Chemical formulas use Unicode superscripts/subscripts throughout (Pb²⁺, H₆EDTA²⁺, etc.).
- Target word count for full Section 2: ~2,500 words. Current estimate: ~2,100 words (Sections 2.1–2.3), leaving ~400 words for Section 2.4.

---

# STATUS TRACKER

| Section | Status | Word Estimate |
|---------|--------|---------------|
| 2.1.1 PHREEQC Configuration | COMPLETE | ~180 |
| 2.1.2 Parameter Space Design | COMPLETE | ~850 |
| 2.1.3 Simulation Execution | COMPLETE | ~200 |
| 2.2.1 Feature Engineering | COMPLETE | ~230 |
| 2.2.2 Model Selection and Training | COMPLETE | ~230 |
| 2.2.3 Model Evaluation | COMPLETE | ~220 |
| 2.3.1 Tier 1: Chemical Logic | COMPLETE | ~250 |
| 2.3.2 Tier 2: Literature Benchmarking | COMPLETE | ~280 |
| 2.4 Decision-Support Interface | COMPLETE | ~350 |
| **SECTION 2 TOTAL** | **COMPLETE** | **~2,790** |
