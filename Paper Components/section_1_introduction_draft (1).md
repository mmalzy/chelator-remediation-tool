# Section 1: Introduction

## 1.1 The Problem: Heavy Metal Contamination in Soils

Heavy metal contamination of soils from legacy industrial activity, leaded paint and gasoline, agricultural amendments, and atmospheric deposition remains one of the most persistent environmental challenges worldwide. Lead, copper, zinc, and cadmium are non-degradable, accumulate in surface soils over decades, and persist indefinitely unless physically removed or chemically stabilized, creating long-term exposure risks through direct ingestion, dermal contact, and uptake into food crops (Thornton, 1990). In urban and post-industrial settings, residential soils frequently exceed risk-based screening levels established by the U.S. Environmental Protection Agency, with lead concentrations in some neighborhoods reaching values several times higher than the 400 mg/kg action level (Sharma et al., 2015). The spatial distribution of contamination follows predictable gradients tied to proximity to industrial sites, roadways, and aging housing stock, with urban forest soils showing lead concentrations up to four times those measured at comparable rural sites (Pouyat and McDonnell, 1991).

Coastal urban environments face compounded contamination from multiple overlapping sources. In Rhode Island, the convergence of legacy residential lead from pre-1978 paint and plumbing, historic maritime and industrial activity concentrated around Narragansett Bay, and widespread application of road deicing salts creates a distinctive geochemical setting (Santschi et al., 1984; Thompson et al., 2014). Sediment records from Narragansett Bay document decades of trace metal accumulation from the industrialized Providence metropolitan area (Santschi et al., 1984), while recent spatial analyses have identified numerous hazardous and contaminated sites within coastal zones now subject to salt marsh migration under sea level rise (Burman et al., 2023). The ongoing use of sodium chloride for winter road maintenance introduces an additional dimension: elevated ionic strength in soil pore water alters metal speciation through chloride complexation and competitive ion exchange, effects that are well documented in roadside environments but rarely incorporated into remediation planning (Amrhein et al., 1992; Bäckström et al., 2003; Merrikhpour and Jalali, 2013). This combination of legacy metal contamination and high-ionic-strength pore water motivates explicit inclusion of coastal salinity as a parameter in any geochemical framework applied to Rhode Island soils.

## 1.2 Chelator-Assisted Remediation: Current State and Limitations

Chelating agents form thermodynamically stable complexes with dissolved metal cations, reducing the free — and therefore bioavailable — metal fraction in soil pore water while enhancing metal mobility for extraction or phytoextraction. The effectiveness of a given chelator depends on its stability constants with the target metals, competition from major cations such as Ca²⁺, Mg²⁺, and Fe³⁺, solution pH — which controls both the chelator's protonation state and the speciation of metal hydroxide, carbonate, and chloride complexes — and soil properties including organic matter content and mineral surface area for sorption (Lestan et al., 2008). Ethylenediaminetetraacetic acid (EDTA) has been the industry standard for chelator-assisted remediation of lead-contaminated soils, while nitrilotriacetic acid (NTA) and citric acid offer biodegradable alternatives with different metal selectivity profiles.

In practice, however, chelator selection remains largely empirical. Practitioners typically choose agents based on general recommendations or site-specific trial and error, an approach that cannot account for the multi-dimensional interactions governing chelator performance. A practitioner facing a site contaminated with both lead and zinc at moderately acidic pH in a coastal soil has no systematic way to determine whether EDTA, NTA, or citrate — and at what dose — will minimize the free fraction of both metals simultaneously, given that the optimal chelator for one metal may be suboptimal for another. The sensitivity of chelation to pH, organic matter, competing cations, and ionic strength means that laboratory trials at one site may not transfer to another, and the cost and time required to run comprehensive dose-response experiments across all relevant combinations is prohibitive for routine site assessment.

Geochemical speciation models such as PHREEQC (Parkhurst and Appelo, 2013) can predict metal speciation under specific sets of conditions with thermodynamic rigor, but running individual simulations for each site requires specialized expertise and cannot efficiently explore the parameter space relevant to chelator selection. Machine learning approaches have increasingly been applied to environmental remediation prediction: recent studies have used ensemble methods to predict electrokinetic remediation efficiency (Barkhordari et al., 2024), heavy metal extraction by leaching agents (Qiu et al., 2025), thermal desorption of organic contaminants (Chen et al., 2024), and screening of remediation strategies more broadly (Zhang et al., 2024), as reviewed by Janga et al. (2023). However, most of these ML applications train on relatively sparse experimental datasets that do not systematically cover the multi-dimensional parameter space governing chelator-metal interactions in soil pore water.

A small but growing body of work has begun to bridge geochemical modeling and machine learning. Chang et al. (2023) trained random forest models on PHREEQC-derived adsorption data to predict uranium(VI) behavior on mineral surfaces, demonstrating that simulation-generated training data can capture thermodynamic relationships more systematically than experimental datasets alone. Molina et al. (2025) applied a directly parallel approach, using PHREEQC simulations to inform ML predictions of scaling indices in produced waters. Prasianakis et al. (2025) provided a broader review of methods for coupling geochemistry with machine learning, including benchmarking frameworks for evaluating such hybrid approaches. Despite these advances, no existing study has applied the simulation-to-ML pipeline specifically to chelator-assisted remediation — the multi-metal, multi-chelator, multi-condition optimization problem that practitioners face in the field.

## 1.3 The Gap and Our Approach

We address this gap by using geochemical simulation itself as the data generation engine for machine learning. Rather than relying on limited experimental datasets, we systematically simulated 12,636 unique combinations of metal concentrations, chelator types and doses, pH, soil texture, organic carbon, moisture and redox conditions, and ionic strength using PHREEQC with the minteq.v4.dat thermodynamic database and thermodynamically consistent parameterization. The resulting dataset encodes the full complexity of aqueous metal speciation — including chelator competition, surface complexation on iron oxide sites, and ion-activity effects — across the parameter space relevant to field remediation of contaminated coastal soils. Machine learning models trained on this dataset can then provide near-instantaneous predictions for any new combination of site parameters, replacing hours of manual PHREEQC setup with a single query to a trained model.

The objectives of this study are to: (1) design and execute a systematic PHREEQC simulation framework spanning realistic soil remediation conditions for four priority metals (Pb, Cu, Zn, Cd) and five chelating agents (EDTA, NTA, citrate, humic acid, fulvic acid) under Rhode Island-relevant environmental conditions; and (2) train and validate machine learning models that faithfully reproduce the thermodynamic speciation predictions across the full parameter space. Additionally, we (3) verify model predictions against both internal chemical logic rules and published experimental data from chelator-assisted remediation studies, and (4) deploy the trained models in a practitioner-accessible decision-support interface that translates complex geochemistry into actionable chelator recommendations for field use.

---

## CITATION TRACKING — SECTION 1

### Citations used in Section 1.1:
- Thornton, I., 1990. Global Planet. Change 2(1–2), 121–140. **[VERIFIED Day 8]**
- Sharma, K., Basta, N.T., Grewal, P.S., 2015. Urban Ecosyst. 18(1), 115–132. **[VERIFIED Day 8]**
- Pouyat, R.V., McDonnell, M.J., 1991. Water Air Soil Pollut. 57(1), 797–807. **[VERIFIED Day 8]**
- Santschi, P.H., Nixon, S., Pilson, M., Hunt, C., 1984. Estuar. Coast. Shelf Sci. 19(4), 427–449. **[VERIFIED Day 8]**
- Thompson, M.R., Burdon, A., Boekelheide, K., 2014. Sci. Total Environ. 468–469, 514–522. **[VERIFIED Day 8]**
- Burman, E., Mulvaney, K., Merrill, N., Bradley, M., Wigand, C., 2023. J. Environ. Manage. 331, 117218. **[VERIFIED Day 8]**
- Amrhein, C., Strong, J.E., Mosher, P.A., 1992. Environ. Sci. Technol. 26(4), 703–709. **[VERIFIED Day 5]**
- Bäckström, M., Nilsson, U., Håkansson, K., Allard, B., Karlsson, S., 2003. Water Air Soil Pollut. 147(1), 343–366. **[VERIFIED Day 8]**
- Merrikhpour, H., Jalali, M., 2013. Water Environ. J. 27(4), 524–534. **[VERIFIED Day 8]**

### Citations used in Section 1.2:
- Lestan, D., Luo, C., Li, X., 2008. Environ. Pollut. 153(1), 3–13. **[VERIFIED Day 5]**
- Parkhurst, D.L., Appelo, C.A.J., 2013. USGS Techniques and Methods 6-A43. **[VERIFIED Day 5]**
- Barkhordari, M.S., et al., 2024. J. Environ. Chem. Eng. 12(6), 114330. **[VERIFIED Day 7]**
- Chen, S., et al., 2024. Sci. Total Environ. 927, 172173. **[VERIFIED Day 7]**
- Qiu, Y., et al., 2025. J. Environ. Chem. Eng. 14(1), 120716. **[VERIFIED Day 7]**
- Zhang, Y., et al., 2024. Processes 12(6), 1157. **[VERIFIED Day 7]**
- Janga, B., et al., 2023. Chemosphere 345, 140476. **[VERIFIED Day 7]**
- Chang, E., et al., 2023. Appl. Geochem. 155, 105731. **[VERIFIED Day 7]**
- Molina, O., et al., 2025. URTEC D021S034R003. **[VERIFIED Day 7]**
- Prasianakis, N.I., et al., 2025. Environ. Earth Sci. 84(5), 121. **[VERIFIED Day 7]**

### Citations used in Section 1.3:
- (No new citations — references PHREEQC and minteq.v4.dat already cited above)

**Total new citations introduced in Section 1:** 10 (Thornton, Sharma, Pouyat, Santschi, Thompson, Burman, Bäckström, Merrikhpour, Qiu, Prasianakis)
**Total citations reused from Sections 2–3:** 9 (Amrhein, Lestan, Parkhurst, Barkhordari, Chen, Zhang, Janga, Chang, Molina)
**All citations verified.**

---

## STYLE NOTES

- Voice: Third person passive for factual statements, first person plural ("We address...") sparingly for interpretive/framing statements per established convention
- Chemical formulas: Unicode superscripts (Ca²⁺, Mg²⁺, Fe³⁺)
- Numbers: Spelled out below 10 in prose, numerals for measurements (per convention)
- Ranges: En dashes (12,636; 5.5–7.5)
- No emojis, no bullets in paper prose
- RI context woven naturally through contamination history and ionic strength rationale

## WORD COUNT

- Section 1.1: ~310 words
- Section 1.2: ~460 words
- Section 1.3: ~250 words
- **Total Section 1: ~1,020 words**
- Target was ~1,200 words; current draft is slightly lean. Can expand 1.2 if needed, or this is within acceptable range for J Hazard Mater.

---
