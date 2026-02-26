# Section 4: Conclusions

This study presents a novel framework that couples systematic thermodynamic geochemical simulation with machine learning to address the practical challenge of chelator selection in heavy metal remediation. Using PHREEQC with the minteq.v4.dat database, 12,636 unique simulation scenarios were generated spanning the full factorial combination of metal concentrations, chelator types and doses, soil properties, and environmental conditions relevant to coastal remediation in Rhode Island, including no-chelator baselines. This systematic approach — using simulation as the data generation engine rather than relying on sparse experimental datasets — produces a training set that encodes the full thermodynamic complexity of metal speciation across the parameter space relevant to field remediation.

Gradient Boosting regression models trained on the simulation outputs predict percent free dissolved metal with cross-validated R² values of 0.979–1.000 and root mean square errors of 0.15–0.83 percentage points across all four metals. Feature importance analysis confirms that the models have learned geochemically consistent relationships: pH emerges as the dominant control variable for all metals, with chelator type and dose as the primary treatment variables, while chloride and sodium concentrations — reflecting the coastal ionic strength conditions specific to Rhode Island — dominate cadmium speciation predictions. The models correctly capture metal-specific chelator rankings, the counterintuitive reduction of free metal at high ionic strength through chloride complexation, and the resistance of zinc to chelation across all agents and conditions. Internal chemical logic validation and comparison with published experimental data confirm that predictions are consistent with known thermodynamic and surface chemistry principles.

The framework is deployed as an open-source Streamlit decision-support interface where practitioners input site-specific soil parameters and receive chelator recommendations with predicted effectiveness ratings for each metal. By replacing hours of manual PHREEQC simulation setup — or costly laboratory screening experiments — with near-instantaneous model queries, this tool makes thermodynamically grounded remediation guidance accessible to field practitioners without requiring specialized geochemical modeling expertise. The simulation-to-ML pipeline demonstrated here is in principle transferable beyond chelation: any remediation context where geochemical models can systematically explore parameter space is amenable to this approach. Future development will integrate reactive transport modeling for kinetic and open-system effects, expand the framework to additional metals and chelators, and validate predictions against bench-scale extraction experiments with contaminated Rhode Island soils.

---

## CITATION TRACKING — SECTION 4

No new citations introduced. Section 4 is a summary of findings already presented and cited in Sections 2 and 3.

## WORD COUNT

- **Total Section 4: ~370 words**
- Target was ~400 words; within acceptable range.

---
