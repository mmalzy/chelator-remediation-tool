# Chelator ML Remediation Project
## Rhode Island Coastal Soil Remediation Decision Support System

---

## PROJECT OVERVIEW

### Goal
Build a machine learning model trained on aqueous geochemical simulations to recommend 
optimal chelating agents for heavy metal remediation in contaminated soils. The final 
product is an interface where practitioners input site-specific soil parameters and 
receive chelator recommendations with predicted effectiveness.

### Scientific Approach
- Use PHREEQC (USGS geochemical modeling software) to simulate metal speciation 
  in soil pore water under thousands of different conditions
- Parse outputs to extract % free metal fraction (the target variable)
- Train ML model on simulation results
- Deploy as a Streamlit interface for field use

### Why % Free Fraction?
Free dissolved metal (e.g., Pb+2, Cu+2, Zn+2, Cd+2) represents the bioavailable, 
mobile fraction that poses the greatest environmental and health risk. Chelators work 
by binding free metals into stable complexes, reducing bioavailability.

---

## CO-CREATOR CONTEXT (For AI Assistant)

### Role
You are acting as both:
1. An expert soil biogeochemist who understands metal speciation, chelation chemistry,
   PHREEQC modeling, and Rhode Island coastal soil conditions
2. An expert Python developer who writes clean, well-documented, production-ready code

### Key Scientific Knowledge Needed
- Heavy metal speciation in soil pore water (pH-dependent, ligand competition)
- Chelation chemistry: EDTA > Citrate > NTA for Pb/Cu; Citrate > EDTA for Zn
- Soil surface complexation (iron/aluminum oxides as Hfo_wOH sites)
- Rhode Island context: coastal salinity, road salt, legacy contamination
- PHREEQC thermodynamic databases and reaction definitions
- DOC as proxy for organic matter in soil pore water
- pe (electron activity) as proxy for soil moisture/redox conditions

### Key Chemistry Findings So Far
- pH is the MOST important variable (higher pH = less free metal)
- High ionic strength (coastal RI) actually REDUCES free Pb/Cu (chloride complexation)
- Zn is hardest to chelate (mean 83.65% free even with chelators)
- Best scenario: EDTA 300 mg/L, pH 7.5, Clay, Low ionic = ~0% free for all metals
- Chelator ranking for Pb: Citrate > EDTA > NTA > Humic/Fulvic
- Humic/Fulvic acids modeled as additional DOC (not separate species in minteq.v4.dat)

---

## SYSTEM SETUP

### Computer
- MacBook Pro, Apple Silicon (M1/M2/M3), macOS
- Username: mallorymalz
- Terminal: zsh

### Software Installed
- Python 3.9 (system): /Library/Developer/CommandLineTools/usr/bin/python3
- Python packages: pandas, numpy (installed to user: ~/Library/Python/3.9/)
- Homebrew: /opt/homebrew/bin/brew
- PHREEQC 3.5.0: /usr/local/bin/phreeqc

### PHREEQC Details
- Executable: /usr/local/bin/phreeqc
- Version: 3.5.0-14000
- CRITICAL: phreeqc does NOT support --version flag (throws error)
- Run syntax: phreeqc <input_file> <output_file> <database_file>
- PHREEQC output files use latin-1 encoding (NOT utf-8)
  - Always open with: open(file, 'r', encoding='latin-1')

### PHREEQC Databases
Location: /usr/local/share/phreeqc_databases/
Available databases:
  - phreeqc.dat     → Default, basic species (NO organic chelators)
  - minteq.v4.dat   → USE THIS ONE (has EDTA, Citrate, NTA, surface complexation)
  - minteq.dat      → Older version of minteq
  - wateq4f.dat     → Alternative thermodynamic database
  - llnl.dat        → Lawrence Livermore database (large)
  - pitzer.dat      → High ionic strength/saline conditions
  - sit.dat         → Specific ion interaction theory

CRITICAL: Always use minteq.v4.dat for this project!
Command example:
  phreeqc input.phr output.txt /usr/local/share/phreeqc_databases/minteq.v4.dat

---

## PROJECT DIRECTORY STRUCTURE

Base path: /Users/mallorymalz/Documents/chelator_ml_project/

chelator_ml_project/
├── PROJECT_README.md           ← This file
├── phreeqc_inputs/             ← All PHREEQC .phr input files
│   ├── test1.phr               ← Simple pH test (learning exercise)
│   ├── test2.phr               ← Lead + EDTA test (had errors)
│   ├── test3.phr               ← Lead with no chelator (BASELINE)
│   ├── test4.phr               ← Lead with citrate (minteq.v4.dat)
│   ├── test4_clean.phr         ← Cleaner version of test4
│   ├── test5.phr               ← Lead with EDTA (working!)
│   ├── scenario_001-120.phr    ← Pilot study (120 scenarios, Pb only)
│   ├── full_scenario_0001-1134.phr  ← Full study (Pb/Cu/Zn, no DTPA)
│   ├── dtpa_scenario_0001+.phr ← DTPA test study (DTPA underperformed)
│   ├── RI_final_00001+.phr     ← MAIN DATASET (Mesic/Wet conditions)
│   └── dry_00001+.phr          ← Supplemental dry scenarios
│
├── phreeqc_outputs/            ← All PHREEQC output .txt files
│   ├── test1_output.txt through test5_output.txt
│   ├── scenario_001-120.txt    ← Pilot outputs
│   ├── full_scenario_*.txt     ← Full study outputs
│   ├── dtpa_scenario_*.txt     ← DTPA study outputs
│   ├── RI_final_*.txt          ← Main dataset outputs (8,100 scenarios)
│   └── dry_*.txt               ← Dry scenario outputs (4,050 scenarios)
│
├── python_scripts/             ← All Python code
│   ├── compare_chelators.py    ← Compares free Pb across test scenarios
│   ├── analyze_pilot_data.py   ← Analysis of 120-scenario pilot study
│   ├── generate_pilot_data.py  ← Generates 120-scenario pilot dataset
│   ├── generate_full_training_data.py  ← 1,134-scenario full dataset
│   ├── generate_dtpa_training_data.py  ← DTPA experiment (not ideal)
│   ├── generate_final_RI_training_data.py  ← MAIN: 8,100 RI scenarios
│   ├── generate_dry_supplement.py  ← Supplemental 4,050 dry scenarios
│   └── analyze_full_data.py    ← Analysis of full training dataset
│
├── data/                       ← All CSV training data
│   ├── pilot_training_data.csv     ← 120 scenarios (Pb only, pilot)
│   ├── full_training_data.csv      ← 1,134 scenarios (Pb/Cu/Zn)
│   ├── dtpa_training_data.csv      ← DTPA experiment data
│   ├── RI_final_training_data.csv  ← 8,100 RI scenarios (Mesic/Wet)
│   ├── dry_scenarios.csv           ← 4,050 dry scenarios only
│   └── complete_training_data.csv  ← MASTER: all 12,150 scenarios ← USE THIS
│
├── models/                     ← Trained ML models (to be created)
│   └── (empty - next phase)
│
└── notes/                      ← Documentation and reference materials
    └── (project notes)

---

## COMPLETE TRAINING DATASET

File: /Users/mallorymalz/Documents/chelator_ml_project/data/complete_training_data.csv
Dimensions: 12,150 rows × 36 columns

### Input Features (20 columns):
| Column | Description | Units | Range |
|--------|-------------|-------|-------|
| ph | Soil pore water pH | - | 5.5-7.5 |
| metal_level | Contamination level | category | Low/Medium/High |
| pb_mg_L | Lead concentration | mg/L | 25/100/300 |
| cu_mg_L | Copper concentration | mg/L | 20/80/250 |
| zn_mg_L | Zinc concentration | mg/L | 30/120/400 |
| cd_mg_L | Cadmium concentration | mg/L | 2/8/25 |
| doc_mg_L | Dissolved organic carbon | mg/L | 10/25/40 |
| ca_mg_L | Calcium (competition) | mg/L | 20/100 |
| mg_mg_L | Magnesium (competition) | mg/L | 10/50 |
| na_mg_L | Sodium (ionic strength) | mg/L | 100/500/2000 |
| cl_mg_L | Chloride (ionic strength) | mg/L | 150/700/3000 |
| chelator | Chelating agent | category | EDTA/NTA/Citrate/Humic/Fulvic |
| dose_mg_L | Chelator dose | mg/L | 50/150/300 |
| texture | Soil texture class | category | Sand/Loam/Clay |
| hfo_sites | Iron oxide surface sites | mol | 0.1/0.5/1.5 |
| moisture | Moisture condition | category | Dry/Mesic/Wet |
| pe | Redox potential proxy | - | 3/8/12 |
| ca_mg_level | Ca/Mg competition level | category | Low/High |
| ionic_level | Salinity level | category | Low/Medium/High |
| doc_mg_L | DOC tied to texture | mg/L | 10/25/40 |

### Target Variables (8 columns - PRIMARY OUTPUTS):
| Column | Description |
|--------|-------------|
| pb_percent_free | % free Pb+2 in solution ← PRIMARY |
| cu_percent_free | % free Cu+2 in solution ← PRIMARY |
| zn_percent_free | % free Zn+2 in solution ← PRIMARY |
| cd_percent_free | % free Cd+2 in solution ← PRIMARY |
| pb_sorbed_mol | Pb sorbed to soil surfaces |
| cu_sorbed_mol | Cu sorbed to soil surfaces |
| zn_sorbed_mol | Zn sorbed to soil surfaces |
| cd_sorbed_mol | Cd sorbed to soil surfaces |

---

## PHREEQC INPUT FILE FORMAT

### Basic Structure:
TITLE [description]
SOLUTION 1
    temp      25
    pH        [value]
    pe        [value]    # Redox proxy: Dry=12, Mesic=8, Wet=3
    units     mol/L
    [element] [concentration in mol/L]
    
SURFACE 1    # Soil surface complexation
    Hfo_wOH   [value]  600  0.09
    # Texture: Sand=0.1, Loam=0.5, Clay=1.5
    -equil 1
    
END

### Chelator Lines (mol/L units):
EDTA:    Edta      [mol/L]    # MW = 292.24 g/mol
NTA:     Nta       [mol/L]    # MW = 191.14 g/mol
Citrate: Citrate   [mol/L]    # MW = 189.1 g/mol
Humic:   Added as extra C(4) (DOC proxy)
Fulvic:  Added as extra C(4) × 0.8 (smaller binding capacity)

### Unit Conversion (mg/L to mol/L):
mol/L = (mg/L / 1000) / MW

### Key Species in minteq.v4.dat:
- Pb+2, PbOH+, PbCO3, PbCl+, Pb(Edta)-2, Pb(Citrate)-
- Cu+2, CuOH+, Cu(Edta)-2, Cu(Citrate)-
- Zn+2, ZnOH+, Zn(Edta)-2, Zn(Citrate)-
- Cd+2, CdOH+, Cd(Edta)-2
- Hfo_wOPb+, Hfo_wOCu+, Hfo_wOZn+, Hfo_wOCd+ (surface species)

---

## PARAMETER SCIENCE (IMPORTANT FOR ML MODEL)

### pH Effects:
- Lower pH → more H+ competing with metals for chelator binding
- Lower pH → desorption of metals from soil surfaces  
- Higher pH = better chelation (pH 7.5 optimal)
- pH is the SINGLE MOST IMPORTANT variable

### Ionic Strength Effects (RI-Specific):
- High ionic strength (coastal/road salt) → chloride complexes form
- PbCl+, PbCl2 etc. reduce free Pb+2 but create mobile complexes
- High salinity actually LOWERS % free metal counterintuitively

### Soil Texture Effects:
- More clay = more surface sites = more metal sorption
- DOC increases with clay content (tied together in our model)
- Sand: hfo=0.1, DOC=10 mg/L
- Loam: hfo=0.5, DOC=25 mg/L
- Clay: hfo=1.5, DOC=40 mg/L

### Moisture/Redox Effects (pe):
- Dry (pe=12): Oxidizing conditions, metals more mobile
- Mesic (pe=8): Moderate conditions
- Wet (pe=3): Reducing conditions, some metals can precipitate as sulfides

### Metal Difficulty Ranking (hardest to chelate):
1. Zn (mean 83.65% free) - weak binding to all chelators
2. Cd (mean 47.70% free) - moderate
3. Pb (mean 43.67% free) - good chelation especially with EDTA
4. Cu (mean 25.09% free) - best chelation, especially EDTA

### Chelator Performance Ranking (mean % free Pb):
1. Citrate: 31.5% (best overall, biodegradable)
2. EDTA: 36.0% (industry standard, good for Pb/Cu)
3. NTA: 39.6% (compromise, in database)
4. Humic/Fulvic: 55.6% (modeled as DOC, weaker)

---

## INTERFACE DESIGN (PLANNED - NEXT PHASE)

### User Inputs:
- pH (numeric slider: 5.0-8.5)
- Metal concentrations (mg/L for Pb, Cu, Zn, Cd)
- Organic matter % → Python converts to DOC mg/L
  - DOC = OM% × 10 × 0.58 (approximate conversion)
- Moisture condition (dropdown: Dry/Mesic/Wet)
- Soil texture class (dropdown: Sand/Loam/Clay)
- Chelator dose (optional: let model optimize)
- Coastal/saline conditions? (yes/no → ionic strength)

### Model Outputs:
- % free fraction for each metal
- Recommended chelator type
- Recommended dose
- Effectiveness rating (Excellent/Good/Moderate/Poor)
- Warning flags (e.g., "Zn may require additional treatment")

### Technology Stack:
- ML Model: Random Forest or Gradient Boosting (scikit-learn)
- Interface: Streamlit (Python web app)
- Deployment: Local or Streamlit Cloud

---

## NEXT STEPS

### Phase 1: COMPLETE ✓
- PHREEQC installed and working
- Test simulations validated
- 12,150 training scenarios generated
- CSV ready for ML

### Phase 2: NEXT - Machine Learning
1. Load complete_training_data.csv
2. Feature engineering (encode categorical variables)
3. Train Random Forest model
4. Evaluate model performance (R², RMSE)
5. Feature importance analysis
6. Save trained model

### Phase 3: Interface
1. Build Streamlit interface
2. Add OM% → DOC conversion
3. Add chelator recommendation logic
4. Test with real RI site data
5. Deploy

---

## IMPORTANT NOTES AND LESSONS LEARNED

1. PHREEQC does NOT have --version flag (use just 'phreeqc' to test)
2. Always use minteq.v4.dat database (not phreeqc.dat)
3. DTPA was tested but underperformed vs expectations - NOT in minteq.v4.dat,
   manual thermodynamic definition may have errors - exclude from final model
4. Output files use latin-1 encoding - always specify encoding='latin-1'
5. Humic/Fulvic acids modeled as DOC additions (not ideal but best available)
6. PHREEQC path: /usr/local/bin/phreeqc (copied from .dmg manually)
7. Database path: /usr/local/share/phreeqc_databases/minteq.v4.dat
8. NTA IS in minteq.v4.dat and works correctly
9. Arsenic excluded - exists as oxyanion (AsO4 3-), not chelated by EDTA/citrate
10. Rhode Island context: include high ionic strength (Na=2000, Cl=3000 mg/L)
    to represent coastal saline conditions and road salt impacts

---

## CONTACT/PROJECT INFO
Researcher: Mallory Malz
Institution: Rhode Island
Project: Heavy Metal Chelation ML Remediation Tool
Status: Data generation complete, ML training next
