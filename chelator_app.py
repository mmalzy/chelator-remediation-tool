"""
Chelator ML Remediation Tool — v4 (Professional Interface, Fixed Contrast)
Rhode Island Coastal Soil Remediation Decision Support System

Usage:
    cd /Users/mallorymalz/Documents/chelator_ml_project/python_scripts
    python3 -m streamlit run chelator_app_v4.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chelator Remediation Tool",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Professional CSS — forced light theme, explicit text colors everywhere
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- Import fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&family=Playfair+Display:wght@500;700&display=swap');

/* ===========================================================
   FORCE LIGHT BACKGROUND — overrides Streamlit dark mode
   =========================================================== */
.stApp,
.main,
.block-container,
[data-testid="stAppViewContainer"],
[data-testid="stMainBlockContainer"] {
    background-color: #ffffff !important;
    color: #1e293b !important;
}

/* ---- Global text ---- */
html, body, [class*="css"],
p, span, li, label, div {
    font-family: 'Source Sans Pro', sans-serif !important;
}
/* Force dark text on main area */
.stApp p, .stApp span, .stApp label, .stApp div,
.stApp li, .stApp td, .stApp th {
    color: #1e293b !important;
}
h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: #1a3a52 !important;
}

/* ---- Hide default Streamlit menu / footer ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ---- Hide the header bar text but keep sidebar toggle functional ---- */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.5rem !important;
}

/* ---- Sidebar: hide ALL close/collapse buttons inside it ---- */
button[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] button[kind="header"],
section[data-testid="stSidebar"] > div:first-child > button,
section[data-testid="stSidebar"] button:has(.material-symbols-rounded),
[data-testid="stSidebarCollapsedControl"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    position: absolute !important;
    pointer-events: none !important;
}

/* ---- If sidebar is somehow collapsed, style the reopen arrow ---- */
[data-testid="collapsedControl"] {
    color: #1a3a52 !important;
    background: #f4f7f6 !important;
    border: 1px solid #d1ddd9 !important;
    border-radius: 0 8px 8px 0 !important;
}

/* ===========================================================
   TOP BANNER — teal-slate gradient (not pure navy-on-black)
   =========================================================== */
.top-banner {
    background: linear-gradient(135deg, #1a3a52 0%, #2a7a6e 100%);
    padding: 2rem 2.5rem 1.6rem;
    border-radius: 0 0 12px 12px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.top-banner h1 {
    color: #ffffff !important;
    font-size: 1.85rem;
    margin: 0 0 0.3rem 0;
    font-family: 'Playfair Display', serif !important;
    letter-spacing: 0.3px;
}
.top-banner p {
    color: #c8e6df !important;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}

/* ===========================================================
   SIDEBAR — forced light bg, all text explicitly dark
   =========================================================== */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #f4f7f6 !important;
    border-right: 1px solid #d1ddd9;
}
/* Force ALL sidebar text dark */
section[data-testid="stSidebar"] * {
    color: #1e293b !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown h3,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown strong {
    color: #1a3a52 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] span {
    color: #334155 !important;
}
/* Sidebar slider track */
section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background-color: #2a7a6e !important;
}
/* Sidebar number inputs */
section[data-testid="stSidebar"] input {
    color: #1e293b !important;
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
}
/* Sidebar selectbox */
section[data-testid="stSidebar"] [data-baseweb="select"] {
    background-color: #ffffff !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #1e293b !important;
}
/* Sidebar checkbox */
section[data-testid="stSidebar"] .stCheckbox label span {
    color: #1e293b !important;
}
/* Sidebar warning text */
section[data-testid="stSidebar"] .stAlert p {
    color: #92400e !important;
}
/* Sidebar horizontal rule */
section[data-testid="stSidebar"] hr {
    border-color: #d1ddd9 !important;
}

/* ===========================================================
   TABS
   =========================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 2px solid #d1ddd9;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.7rem 1.4rem;
    font-weight: 600;
    font-size: 0.92rem;
    color: #64748b !important;
    border-bottom: 3px solid transparent;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #1a3a52 !important;
    border-bottom: 3px solid #2a7a6e !important;
    background: transparent !important;
}

/* ===========================================================
   RECOMMENDATION CARDS — explicit text colors inside
   =========================================================== */
.rec-card {
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    border-left: 5px solid;
}
.card-excellent { background: #ecfdf5 !important; border-color: #16a34a; }
.card-good      { background: #eff6ff !important; border-color: #2563eb; }
.card-moderate  { background: #fefce8 !important; border-color: #ca8a04; }
.card-poor      { background: #fef2f2 !important; border-color: #dc2626; }

.card-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
    color: #1a3a52 !important;
}
.card-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    /* color set inline per card */
}
.card-label {
    font-size: 0.82rem;
    color: #4a6274 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.15rem;
}
.card-detail {
    font-size: 0.88rem;
    color: #334155 !important;
    margin-top: 0.3rem;
}

/* ===========================================================
   WARNING BOX
   =========================================================== */
.warning-box {
    background: #fefce8 !important;
    border: 1px solid #ca8a04;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
    color: #713f12 !important;
}
.warning-box strong {
    color: #713f12 !important;
}

/* ===========================================================
   METRIC BOXES
   =========================================================== */
.metric-box {
    background: #f4f7f6 !important;
    border: 1px solid #d1ddd9;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-box .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1a3a52 !important;
}
.metric-box .label {
    font-size: 0.78rem;
    color: #4a6274 !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

/* ===========================================================
   TABLES — teal header, explicit cell colors
   =========================================================== */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}
.styled-table thead th {
    background: #1a3a52 !important;
    color: #ffffff !important;
    padding: 0.65rem 0.8rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.styled-table tbody td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #d1ddd9;
    color: #1e293b !important;
}
.styled-table tbody tr:nth-child(even) { background: #f4f7f6 !important; }
.styled-table tbody tr:hover { background: #e8efed !important; }

/* ===========================================================
   ABOUT SECTIONS
   =========================================================== */
.about-section {
    background: #f4f7f6 !important;
    border: 1px solid #d1ddd9;
    border-radius: 10px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1rem;
}
.about-section h3 {
    color: #1a3a52 !important;
    font-size: 1.05rem;
    margin-top: 0;
}
.about-section p,
.about-section li {
    font-size: 0.9rem;
    color: #334155 !important;
    line-height: 1.6;
}
.about-section strong {
    color: #1a3a52 !important;
}
.about-footer {
    margin-top: 1rem;
    font-size: 0.85rem;
    color: #64748b !important;
}

/* ---- Divider ---- */
.section-divider {
    border: none;
    border-top: 1px solid #d1ddd9;
    margin: 1.5rem 0;
}

/* ===========================================================
   STREAMLIT WIDGET OVERRIDES (download button, captions, etc.)
   =========================================================== */
.stDownloadButton button {
    background-color: #1a3a52 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px;
    padding: 0.5rem 1.2rem;
    font-weight: 600;
}
.stDownloadButton button:hover {
    background-color: #2a7a6e !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: #4a6274 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
st.markdown("""
<div class="top-banner">
    <h1>Chelator Remediation Decision Support Tool</h1>
    <p>Rhode Island Coastal Soil &mdash; Heavy Metal Speciation &amp; Chelation Modeling</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check for models/ in multiple locations:
# 1. Same directory as script (Streamlit Cloud / shared folder)
# 2. One level up (local project: python_scripts -> models)
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "models")

METALS = ["Pb", "Cu", "Zn", "Cd"]
TARGET_COLS = [f"{m.lower()}_percent_free" for m in METALS]

CHELATOR_DISPLAY = {
    "EDTA": "EDTA",
    "NTA": "NTA",
    "Citrate": "Citrate",
    "Humic": "Humic Acid",
    "Fulvic": "Fulvic Acid",
    "nan": "No Treatment",
}
CHELATOR_ENCODE = {v: k for k, v in CHELATOR_DISPLAY.items()}

DOSE_OPTIONS_DISPLAY = {
    0: "None",
    50: "50 mg/L (Low)",
    150: "150 mg/L (Moderate)",
    300: "300 mg/L (High)",
}


@st.cache_resource
def load_models():
    models = {}
    for metal in METALS:
        path = os.path.join(MODEL_DIR, f"{metal.lower()}_percent_free_model.joblib")
        models[metal] = joblib.load(path)
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_info.json")) as f:
        feature_info = json.load(f)
    return models, encoders, feature_info


try:
    models, label_encoders, feature_info = load_models()
    model_features = feature_info["feature_columns"]
except Exception as e:
    st.error(f"Failed to load models from {MODEL_DIR}. Error: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# USDA texture triangle helper
# ---------------------------------------------------------------------------
def classify_texture(sand, silt, clay):
    """Simplified USDA texture classification."""
    if sand >= 85:
        return "Sand"
    elif clay >= 40:
        return "Clay"
    elif silt >= 80:
        return "Clay"  # treat as fine-textured
    elif sand >= 50 and clay < 20:
        return "Sand"
    elif clay >= 27:
        return "Clay"
    else:
        return "Loam"


def texture_to_params(texture):
    """Return (hfo_sites, doc_mg_L) for a given texture class."""
    mapping = {"Sand": (0.1, 10), "Loam": (0.5, 25), "Clay": (1.5, 40)}
    return mapping.get(texture, (0.5, 25))


def moisture_to_pe(moisture):
    return {"Dry": 12, "Mesic": 8, "Wet": 3}[moisture]


def ionic_level_from_na(na):
    if na <= 200:
        return "Low"
    elif na <= 800:
        return "Medium"
    return "High"


def metal_level_from_pb(pb):
    if pb <= 50:
        return "Low"
    elif pb <= 200:
        return "Medium"
    return "High"


def ca_mg_level_from_ca(ca):
    return "Low" if ca <= 60 else "High"

# ---------------------------------------------------------------------------
# Effectiveness helpers
# ---------------------------------------------------------------------------
def effectiveness_label(pct_free):
    if pct_free <= 10:
        return "Excellent", "card-excellent"
    elif pct_free <= 30:
        return "Good", "card-good"
    elif pct_free <= 60:
        return "Moderate", "card-moderate"
    return "Poor", "card-poor"


def effectiveness_color(pct_free):
    """Returns a color that is readable on light backgrounds."""
    if pct_free <= 10:
        return "#15803d"   # dark green
    elif pct_free <= 30:
        return "#1d4ed8"   # dark blue
    elif pct_free <= 60:
        return "#a16207"   # dark amber
    return "#b91c1c"       # dark red

# ---------------------------------------------------------------------------
# Build a feature vector for prediction
# ---------------------------------------------------------------------------
def build_features(ph, pb, cu, zn, cd, doc, ca, mg, na, cl, chelator_raw,
                   dose, texture, hfo, moisture, pe, ca_mg_lvl, ionic_lvl,
                   metal_lvl):
    """Build a single-row DataFrame matching the 14 model feature columns."""
    # Encode chelator to a single integer via the label encoder
    if "chelator" in label_encoders:
        le = label_encoders["chelator"]
        if chelator_raw in le.classes_:
            chelator_enc = le.transform([chelator_raw])[0]
        else:
            chelator_enc = 0
    else:
        chelator_enc = 0

    row = {
        "ph": ph,
        "pb_mg_L": pb,
        "cu_mg_L": cu,
        "zn_mg_L": zn,
        "cd_mg_L": cd,
        "doc_mg_L": doc,
        "ca_mg_L": ca,
        "mg_mg_L": mg,
        "na_mg_L": na,
        "cl_mg_L": cl,
        "dose_mg_L": dose,
        "hfo_sites": hfo,
        "pe": pe,
        "chelator_encoded": chelator_enc,
    }

    # Return with columns in the exact order the model expects
    return pd.DataFrame([{f: row.get(f, 0) for f in model_features}])


def predict_all(features_df):
    results = {}
    for metal in METALS:
        pred = models[metal].predict(features_df)[0]
        results[metal] = max(0.0, min(100.0, pred))
    return results

# ---------------------------------------------------------------------------
# SIDEBAR — Site Parameters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Site Parameters")
    st.markdown("---")

    st.markdown("**Soil pH**")
    ph = st.slider("pH", 5.0, 8.5, 6.5, 0.1, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Metal Concentrations (mg/L)**")
    pb = st.number_input("Lead (Pb)", 0.0, 1000.0, 100.0, 10.0)
    cu = st.number_input("Copper (Cu)", 0.0, 1000.0, 80.0, 10.0)
    zn = st.number_input("Zinc (Zn)", 0.0, 1000.0, 120.0, 10.0)
    cd = st.number_input("Cadmium (Cd)", 0.0, 100.0, 8.0, 1.0)

    st.markdown("---")
    st.markdown("**Soil Texture**")
    col_s, col_si, col_c = st.columns(3)
    with col_s:
        sand_pct = st.number_input("Sand %", 0, 100, 40, 5)
    with col_si:
        silt_pct = st.number_input("Silt %", 0, 100, 40, 5)
    with col_c:
        clay_pct = st.number_input("Clay %", 0, 100, 20, 5)

    total_pct = sand_pct + silt_pct + clay_pct
    if total_pct != 100:
        st.warning(f"Percentages sum to {total_pct}%. Adjust to equal 100%.")
    texture = classify_texture(sand_pct, silt_pct, clay_pct)
    hfo, doc = texture_to_params(texture)
    st.caption(f"USDA Classification: **{texture}**")

    st.markdown("---")
    st.markdown("**Moisture Condition**")
    moisture = st.selectbox("Moisture", ["Dry", "Mesic", "Wet"], index=1,
                            label_visibility="collapsed")
    pe = moisture_to_pe(moisture)

    st.markdown("---")
    st.markdown("**Site Context**")
    is_coastal = st.checkbox("Coastal or salt-impacted site", value=True)
    if is_coastal:
        na_val = st.select_slider("Salinity level",
                                  options=[500, 1000, 2000],
                                  value=500,
                                  format_func=lambda x: {500: "Moderate",
                                                         1000: "High",
                                                         2000: "Very High"}[x])
        cl_val = na_val * 1.5
    else:
        na_val = 100
        cl_val = 150

    ca_val = st.select_slider("Calcium/Magnesium competition",
                              options=[20, 60, 100],
                              value=60,
                              format_func=lambda x: {20: "Low", 60: "Moderate",
                                                     100: "High"}[x])
    mg_val = ca_val / 2.0

    # Derived categorical levels
    ionic_lvl = ionic_level_from_na(na_val)
    metal_lvl = metal_level_from_pb(pb)
    ca_mg_lvl = ca_mg_level_from_ca(ca_val)

# ---------------------------------------------------------------------------
# Predictions for all chelator x dose combinations (+ No Treatment baseline)
# ---------------------------------------------------------------------------
chelator_options = ["EDTA", "NTA", "Citrate", "Humic", "Fulvic"]
dose_values = [50, 150, 300]

def run_full_comparison(ph, pb, cu, zn, cd, doc, ca, mg, na, cl,
                        texture, hfo, moisture, pe, ca_mg, ionic, metal):
    rows = []
    # No-treatment baseline
    feats = build_features(ph, pb, cu, zn, cd, doc, ca, mg, na, cl,
                           "nan", 0, texture, hfo, moisture, pe,
                           ca_mg, ionic, metal)
    preds = predict_all(feats)
    rows.append({"Chelator": "No Treatment", "Dose (mg/L)": 0,
                 **{f"{m} % Free": round(preds[m], 1) for m in METALS}})

    # All chelator x dose combos
    for chel in chelator_options:
        for dose in dose_values:
            feats = build_features(ph, pb, cu, zn, cd, doc, ca, mg,
                                   na, cl, chel, dose, texture, hfo,
                                   moisture, pe, ca_mg, ionic, metal)
            preds = predict_all(feats)
            rows.append({"Chelator": CHELATOR_DISPLAY.get(chel, chel),
                         "Dose (mg/L)": dose,
                         **{f"{m} % Free": round(preds[m], 1) for m in METALS}})
    return pd.DataFrame(rows)

comparison_df = run_full_comparison(ph, pb, cu, zn, cd, doc, ca_val, mg_val,
                                    na_val, cl_val, texture, hfo, moisture, pe,
                                    ca_mg_lvl, ionic_lvl, metal_lvl)

# Best overall chelator (lowest average % free across all metals)
treatment_rows = comparison_df[comparison_df["Chelator"] != "No Treatment"].copy()
metal_cols = [f"{m} % Free" for m in METALS]
treatment_rows["avg_free"] = treatment_rows[metal_cols].mean(axis=1)
best_idx = treatment_rows["avg_free"].idxmin()
best_row = treatment_rows.loc[best_idx]
best_chelator = best_row["Chelator"]
best_dose = int(best_row["Dose (mg/L)"])

# Baseline row
baseline_row = comparison_df[comparison_df["Chelator"] == "No Treatment"].iloc[0]

# ---------------------------------------------------------------------------
# MAIN CONTENT — Tabs
# ---------------------------------------------------------------------------
tab_rec, tab_compare, tab_site, tab_about = st.tabs(
    ["Recommendations", "Full Comparison", "Site Summary", "About"]
)

# ===================== TAB 1: Recommendations =====================
with tab_rec:
    # Warnings
    warnings = []
    if ph < 5.5:
        warnings.append("Low pH significantly reduces chelator effectiveness. Consider liming before chelation treatment.")
    if any(comparison_df["Zn % Free"] > 60):
        warnings.append("Zinc is resistant to chelation under these conditions. Supplemental treatment (e.g., soil amendment, phytoremediation) may be needed.")
    if ionic_lvl == "High":
        warnings.append("High salinity detected. Chloride complexation may affect metal speciation. Results account for this, but field verification is recommended.")

    if warnings:
        for w in warnings:
            st.markdown(f'<div class="warning-box"><strong>Advisory:</strong> {w}</div>',
                        unsafe_allow_html=True)

    st.markdown(f"""
    <div class="rec-card card-excellent" style="border-left-width:6px;">
        <div class="card-label">Recommended Treatment</div>
        <div class="card-title" style="font-size:1.35rem;">{best_chelator} at {best_dose} mg/L</div>
        <div class="card-detail">Best overall performance across Pb, Cu, Zn, and Cd for your site conditions.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Predicted Effectiveness by Metal")

    cols = st.columns(4)
    for i, metal in enumerate(METALS):
        pct = best_row[f"{metal} % Free"]
        baseline_pct = baseline_row[f"{metal} % Free"]
        reduction = baseline_pct - pct
        label, css_class = effectiveness_label(pct)
        color = effectiveness_color(pct)

        with cols[i]:
            st.markdown(f"""
            <div class="rec-card {css_class}">
                <div class="card-label">{metal}</div>
                <div class="card-value" style="color:{color};">{pct:.1f}%</div>
                <div class="card-detail">free in solution</div>
                <hr class="section-divider">
                <div class="card-detail">
                    Baseline: {baseline_pct:.1f}%<br>
                    Reduction: {reduction:.1f} pp<br>
                    Rating: <strong>{label}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Per-metal best chelators
    st.markdown("#### Best Chelator by Individual Metal")
    per_metal_rows = []
    for metal in METALS:
        col_name = f"{metal} % Free"
        best_for_metal = treatment_rows.loc[treatment_rows[col_name].idxmin()]
        per_metal_rows.append({
            "Metal": metal,
            "Best Chelator": best_for_metal["Chelator"],
            "Dose (mg/L)": int(best_for_metal["Dose (mg/L)"]),
            "% Free": round(best_for_metal[col_name], 1),
            "Rating": effectiveness_label(best_for_metal[col_name])[0],
        })
    per_metal_df = pd.DataFrame(per_metal_rows)

    header_html = "".join(f"<th>{c}</th>" for c in per_metal_df.columns)
    rows_html = ""
    for _, r in per_metal_df.iterrows():
        cells = "".join(f"<td>{r[c]}</td>" for c in per_metal_df.columns)
        rows_html += f"<tr>{cells}</tr>"
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ===================== TAB 2: Full Comparison =====================
with tab_compare:
    st.markdown("#### All Chelator and Dose Combinations")
    st.caption("Showing predicted % free metal for every treatment option under your site conditions.")

    # Build HTML table
    header_html = "".join(f"<th>{c}</th>" for c in comparison_df.columns)
    rows_html = ""
    for _, r in comparison_df.iterrows():
        cells = ""
        for c in comparison_df.columns:
            val = r[c]
            if isinstance(val, float):
                color = effectiveness_color(val)
                cells += f'<td style="color:{color} !important; font-weight:600;">{val:.1f}%</td>'
            else:
                cells += f"<td>{val}</td>"
        rows_html += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # CSV download
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = comparison_df.to_csv(index=False)
    st.download_button(
        label="Download comparison table as CSV",
        data=csv_data,
        file_name="chelator_comparison.csv",
        mime="text/csv",
    )

# ===================== TAB 3: Site Summary =====================
with tab_site:
    st.markdown("#### Input Parameters Summary")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="metric-box"><div class="label">pH</div><div class="value">{ph:.1f}</div></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-box"><div class="label">Texture</div><div class="value">{texture}</div></div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-box"><div class="label">Moisture</div><div class="value">{moisture}</div></div>
        """, unsafe_allow_html=True)

    st.markdown("##### Metal Concentrations")
    mc1, mc2, mc3, mc4 = st.columns(4)
    for col_w, (name, val) in zip([mc1, mc2, mc3, mc4],
                                   [("Pb", pb), ("Cu", cu), ("Zn", zn), ("Cd", cd)]):
        with col_w:
            st.markdown(f"""
            <div class="metric-box"><div class="label">{name} (mg/L)</div><div class="value">{val:.0f}</div></div>
            """, unsafe_allow_html=True)

    st.markdown("##### Geochemical Context")
    ctx_data = {
        "Parameter": ["DOC (mg/L)", "Ca (mg/L)", "Mg (mg/L)", "Na (mg/L)",
                       "Cl (mg/L)", "pe", "HFO sites (mol)", "Ionic Strength"],
        "Value": [doc, ca_val, mg_val, na_val, cl_val, pe, hfo,
                  ionic_lvl],
    }
    ctx_df = pd.DataFrame(ctx_data)
    header_html = "".join(f"<th>{c}</th>" for c in ctx_df.columns)
    rows_html = "".join(
        "<tr>" + "".join(f"<td>{r[c]}</td>" for c in ctx_df.columns) + "</tr>"
        for _, r in ctx_df.iterrows()
    )
    st.markdown(f"""
    <table class="styled-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

# ===================== TAB 4: About =====================
with tab_about:
    st.markdown("""
    <div class="about-section">
        <h3>About This Tool</h3>
        <p>
            This decision support system predicts the effectiveness of chelating agents
            for heavy metal remediation in contaminated soils, with parameters calibrated
            for Rhode Island coastal environments.
        </p>
        <p>
            <strong>How it works:</strong> The underlying machine learning model (Gradient Boosting)
            was trained on 12,636 geochemical simulations run in PHREEQC 3.5.0 using the
            minteq.v4.dat thermodynamic database. Each simulation computed metal speciation
            in soil pore water under a unique combination of pH, metal loading, chelator type
            and dose, soil texture, moisture/redox conditions, ionic strength, and competing
            cation concentrations.
        </p>
        <p>
            The target variable is <strong>% free dissolved metal</strong> (e.g., Pb<sup>+2</sup>),
            which represents the bioavailable, mobile fraction posing the greatest risk.
            Lower values indicate more effective chelation.
        </p>
    </div>

    <div class="about-section">
        <h3>Chelators Included</h3>
        <ul>
            <li><strong>EDTA</strong> &mdash; Industry standard; strong binding for Pb and Cu</li>
            <li><strong>NTA</strong> &mdash; Moderate strength; biodegradable alternative to EDTA</li>
            <li><strong>Citrate</strong> &mdash; Biodegradable; effective for Pb under favorable conditions</li>
            <li><strong>Humic Acid</strong> &mdash; Natural organic matter; weaker chelation</li>
            <li><strong>Fulvic Acid</strong> &mdash; Natural organic matter; weaker chelation</li>
            <li><strong>No Treatment</strong> &mdash; Baseline (no chelator added)</li>
        </ul>
    </div>

    <div class="about-section">
        <h3>Model Performance</h3>
        <table class="styled-table">
            <thead>
                <tr><th>Metal</th><th>R&sup2;</th><th>CV R&sup2;</th><th>RMSE</th></tr>
            </thead>
            <tbody>
                <tr><td>Lead (Pb)</td><td>0.9990</td><td>0.9788</td><td>0.83%</td></tr>
                <tr><td>Copper (Cu)</td><td>0.9997</td><td>0.9481</td><td>0.59%</td></tr>
                <tr><td>Zinc (Zn)</td><td>0.9998</td><td>0.9972</td><td>0.33%</td></tr>
                <tr><td>Cadmium (Cd)</td><td>1.0000</td><td>0.9999</td><td>0.15%</td></tr>
            </tbody>
        </table>
    </div>

    <div class="about-section">
        <h3>Key Findings from Training Data</h3>
        <ul>
            <li>pH is the single most important variable controlling chelation effectiveness.</li>
            <li>Zinc is the most resistant metal to chelation (mean 84% free across all scenarios).</li>
            <li>High ionic strength (coastal salinity) can reduce free metal through chloride complexation.</li>
            <li>Clay soils with high iron oxide content sorb more metals, reducing bioavailability.</li>
            <li>Optimal conditions: EDTA at 300 mg/L, pH 7.5, Clay texture, Low ionic strength.</li>
        </ul>
    </div>

    <div class="about-section">
        <h3>Limitations and Disclaimers</h3>
        <p>
            This tool provides predictions based on thermodynamic equilibrium modeling and should
            be used as a screening-level guide, not a substitute for site-specific investigation.
            Key limitations include:
        </p>
        <ul>
            <li>PHREEQC models assume thermodynamic equilibrium; kinetic effects are not captured.</li>
            <li>Humic and fulvic acids are modeled as DOC proxies, not as explicit species.</li>
            <li>Predictions are most reliable within the parameter ranges used in training.</li>
            <li>Field conditions (heterogeneity, preferential flow, microbial activity) are not modeled.</li>
        </ul>
        <p class="about-footer">
            Developed by Mallory Malz &mdash; University of Rhode Island<br>
            Geochemical modeling: PHREEQC 3.5.0 (USGS) with minteq.v4.dat<br>
            Machine learning: scikit-learn Gradient Boosting Regressor
        </p>
    </div>
    """, unsafe_allow_html=True)
