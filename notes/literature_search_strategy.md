# Literature Search Strategy for Tier 2 Validation
## Chelator ML Remediation Project

---

## SEARCH DATABASES
- Google Scholar (free, broadest coverage)
- Web of Science (if university access available)
- Scopus (if university access available)
- PubMed (for environmental health angle)

## TARGET JOURNALS
- Journal of Hazardous Materials
- Chemosphere
- Environmental Pollution
- Science of the Total Environment
- Environmental Science & Technology
- Soil & Sediment Contamination
- Journal of Environmental Quality
- Geoderma
- Journal of Soils and Sediments

## SEARCH QUERIES (copy-paste ready)

### Primary searches:
1. "EDTA" AND "soil" AND "lead extraction" AND "chelator"
2. "chelator-assisted" AND "soil remediation" AND ("lead" OR "copper" OR "zinc" OR "cadmium")
3. "EDTA" AND "NTA" AND "citrate" AND "soil" AND "heavy metal"
4. "chelant-enhanced" AND "phytoextraction" AND "metal mobilization"
5. "soil washing" AND "chelating agent" AND ("Pb" OR "Cu" OR "Zn" OR "Cd")
6. "EDTA" AND "soil" AND "extraction efficiency" AND "pH"
7. "chelator" AND "dose" AND "metal" AND "soil" AND "batch experiment"

### Narrowing searches (if too many results):
8. "EDTA" AND "citrate" AND "comparison" AND "soil" AND "heavy metal"
9. "chelator" AND "soil" AND "speciation" AND "free metal"
10. "coastal soil" AND "heavy metal" AND "remediation" AND "chelator"

## WHAT TO LOOK FOR IN EACH PAPER

### Required information (must have ALL of these):
- Soil pH
- At least one metal concentration (Pb, Cu, Zn, or Cd) in mg/kg
- Chelator type (must be EDTA, NTA, or Citrate for best model match)
- Chelator dose with units
- Some measure of extraction/mobilization result (% extracted, mg extracted, etc.)

### Highly desirable information:
- Soil texture or particle size distribution
- Organic matter or organic carbon content
- Multiple chelators compared in same study (best for ranking validation)
- Multiple doses tested (for dose-response validation)
- pH range tested (for pH-effect validation)
- Control/no-chelator treatment for baseline comparison

### Bonus information:
- CEC (cation exchange capacity)
- Electrical conductivity or ionic strength
- Iron/aluminum oxide content
- Mineralogy
- Kinetic data (extraction at multiple time points)
- Free ion measurements (ISE or DGT)

## DATA EXTRACTION TEMPLATE

For each usable data point from a paper, record in the CSV:

| Field | Where to find it |
|-------|-----------------|
| citation | Author (Year) Journal Volume:Pages |
| doi | Usually on first page or header |
| soil_texture | Methods section, soil characterization |
| ph | Methods or Results, soil properties table |
| om_percent | Soil properties table (may be listed as TOC - multiply by 1.72) |
| pb/cu/zn/cd_mg_kg | Soil characterization table, total metals |
| chelator_used | Methods section, experimental design |
| chelator_dose | Methods section, usually in experimental design |
| dose_unit | READ CAREFULLY - mmol/kg, g/kg, mM, mg/L vary widely |
| contact_time_hr | Methods section, experimental procedure |
| liquid_solid_ratio | Methods section (e.g., "10:1 L/S ratio") |
| observed_extraction_pct | Results section, usually in tables or figures |

## TIPS

- Review papers and meta-analyses are goldmines - they compile data from
  many studies. Search for: "review" AND "chelator" AND "soil remediation"
- Look at supplementary materials - detailed data tables are often there
- If extraction % is only in figures, use a tool like WebPlotDigitizer to
  extract numerical values from graphs
- Convert TOC (total organic carbon) to OM by multiplying by 1.72
- Some papers report DTPA-extractable metals - this is different from total
  metals and from our model's pore water predictions
- Papers from China and Europe have the most batch extraction data
- Studies on phytoextraction often include chelator-soil interaction data

## TARGET: 15-25 data points from 5-10 papers

A good benchmark dataset would have:
- At least 3 studies using EDTA on Pb-contaminated soil
- At least 2 studies comparing EDTA vs citrate
- At least 2 studies with dose-response data
- At least 2 studies at different pH values
- A range of soil textures (sandy to clayey)
- Metal concentrations spanning Low to High range

---
