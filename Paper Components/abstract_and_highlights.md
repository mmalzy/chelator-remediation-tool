# Abstract and Highlights

## HIGHLIGHTS

- 12,636 PHREEQC simulations generate ML training data for chelator selection
- Gradient Boosting predicts free metal fraction with R-squared above 0.99
- pH is the dominant control; chelator type and dose are key treatment variables
- Model validated against chemical logic rules and published extraction data
- Open-source Streamlit interface translates geochemistry to field decisions

## ABSTRACT

Heavy metal contamination in urban and coastal soils poses persistent risks to human health and ecosystem function. Chelator-assisted remediation is widely used but chelator selection remains largely empirical, with practitioners choosing agents based on experience rather than site-specific geochemical analysis. No existing tool provides rapid, multi-metal, multi-chelator comparison across the range of soil conditions encountered in practice.

We present a novel framework that couples thermodynamic geochemical simulation with machine learning to create a decision-support tool for chelator selection. Using PHREEQC with the minteq.v4.dat thermodynamic database, we generated 12,636 unique simulation scenarios spanning four priority metals (Pb, Cu, Zn, Cd), five chelating agents (EDTA, NTA, citrate, humic acid, fulvic acid), three dose levels, five pH values (5.5–7.5), three soil textures with corresponding surface complexation and organic carbon parameters, three moisture/redox conditions, three ionic strength levels reflecting coastal and road-salt-impacted conditions, and no-chelator baselines. Gradient Boosting regression models trained on simulation outputs predict percent free dissolved metal with R-squared values of 0.979–1.000 and RMSE of 0.15–0.83 percentage points across all four metals in cross-validation.

Internal validation confirms the models obey fundamental geochemical rules, and comparison with published experimental data shows agreement on chelator rankings and metal difficulty trends. The framework is deployed as an open-source Streamlit interface where practitioners input site-specific soil parameters and receive chelator recommendations with predicted effectiveness. This approach — using simulation to systematically explore parameter space that would require thousands of laboratory experiments — represents a scalable methodology for translating geochemical knowledge into practical remediation guidance, with direct applicability to the contaminated coastal soils of Rhode Island and similar legacy-contaminated environments.

---

## CORRECTIONS MADE

1. "paraemeters" → "parameters"
2. "no-chealtor" → "no-chelator"
3. "rankngs" → "rankings"
4. "site specific" → "site-specific"
5. "remdiation" → "remediation"
6. "Gradient boosting" → "Gradient Boosting"
7. "road-salt impacted" → "road-salt-impacted"
8. Hyphens in "5.5-7.5" and "0.979-1.000" changed to en dashes (5.5–7.5, 0.979–1.000) per journal style
9. Added em dash around the parenthetical clause in the final sentence for clarity
