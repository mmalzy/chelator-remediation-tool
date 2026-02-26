#!/usr/bin/env python3
"""
chelator_app.py
================
Streamlit Decision Support Interface for Heavy Metal Remediation
Rhode Island Coastal Soil Chelator Recommendation Tool

Loads trained ML models and provides practitioners with:
- Site-specific chelator recommendations
- Predicted % free metal for each treatment option
- Comparison of all chelators at multiple doses
- Improvement over untreated baseline

Usage:
    streamlit run chelator_app.py

Author: Mallory Malz (with AI co-creator)
Project: Chelator ML Remediation - Rhode Island Coastal Soils
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import joblib

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = "/Users/mallorymalz/Documents/chelator_ml_project"
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Metal display info
METALS = {
    "pb_percent_free": {"name": "Lead (Pb)", "symbol": "Pb", "color": "#E53935"},
    "cu_percent_free": {"name": "Copper (Cu)", "symbol": "Cu", "color": "#1E88E5"},
    "zn_percent_free": {"name": "Zinc (Zn)", "symbol": "Zn", "color": "#43A047"},
    "cd_percent_free": {"name": "Cadmium (Cd)", "symbol": "Cd", "color": "#FB8C00"},
}

# Chelator info for recommendations
CHELATOR_INFO = {
    "EDTA": {
        "full_name": "Ethylenediaminetetraacetic acid",
        "biodegradable": False,
        "notes": "Industry standard. Strong binding for Pb and Cu. Persistent in environment.",
    },
    "NTA": {
        "full_name": "Nitrilotriacetic acid",
        "biodegradable": True,
        "notes": "Moderate binding strength. More biodegradable than EDTA.",
    },
    "Citrate": {
        "full_name": "Citric acid / Citrate",
        "biodegradable": True,
        "notes": "Natural, biodegradable. Effective for Pb. Food-grade available.",
    },
    "Humic": {
        "full_name": "Humic acid",
        "biodegradable": True,
        "notes": "Natural soil organic matter fraction. Gentle, low-cost option.",
    },
    "Fulvic": {
        "full_name": "Fulvic acid",
        "biodegradable": True,
        "notes": "Smaller molecular weight fraction of humic substances.",
    },
}

# USDA texture classification from sand/silt/clay percentages
def classify_texture(sand, silt, clay):
    """
    Simplified USDA texture triangle classification.
    Maps to the three classes used in the training data.
    Returns texture class and the reasoning.
    """
    if clay >= 35:
        return "Clay", f"{clay}% clay ≥ 35% → Clay class"
    elif sand >= 70:
        return "Sand", f"{sand}% sand ≥ 70% → Sand class"
    else:
        return "Loam", f"Mixed texture → Loam class"


def get_texture_properties(texture):
    """Get HFO sites and DOC for a texture class (matches training data)."""
    props = {
        "Sand": {"hfo_sites": 0.1, "doc_mg_L": 10},
        "Loam": {"hfo_sites": 0.5, "doc_mg_L": 25},
        "Clay": {"hfo_sites": 1.5, "doc_mg_L": 40},
    }
    return props[texture]


def get_ionic_values(level):
    """Get Na/Cl values for ionic strength level (matches training data)."""
    values = {
        "Low":    {"na_mg_L": 100,  "cl_mg_L": 150},
        "Medium": {"na_mg_L": 500,  "cl_mg_L": 700},
        "High":   {"na_mg_L": 2000, "cl_mg_L": 3000},
    }
    return values[level]


def get_ca_mg_values(level):
    """Get Ca/Mg values for competition level (matches training data)."""
    values = {
        "Low":  {"ca_mg_L": 20,  "mg_mg_L": 10},
        "High": {"ca_mg_L": 100, "mg_mg_L": 50},
    }
    return values[level]


def get_moisture_pe(moisture):
    """Get pe value for moisture condition (matches training data)."""
    pe_map = {"Dry": 12, "Mesic": 8, "Wet": 3}
    return pe_map[moisture]


@st.cache_resource
def load_models():
    """Load all trained models and encoders (cached so it only loads once)."""
    models = {}
    for target in METALS.keys():
        model_path = os.path.join(MODEL_DIR, f"{target}_model.joblib")
        if os.path.exists(model_path):
            models[target] = joblib.load(model_path)
        else:
            st.error(f"Model not found: {model_path}")
            return None, None, None

    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))

    with open(os.path.join(MODEL_DIR, "feature_info.json"), 'r') as f:
        feature_info = json.load(f)

    return models, encoders, feature_info


def predict_single(models, encoders, feature_info, params):
    """
    Make predictions for a single set of conditions.
    params: dict with all input feature values
    Returns: dict of {target: predicted_percent_free}
    """
    # Build feature row in correct order
    row = {}
    for col in feature_info["feature_columns"]:
        if col == "chelator_encoded":
            chelator_val = params.get("chelator", "nan")
            le = encoders["chelator"]
            if chelator_val in le.classes_:
                row[col] = le.transform([chelator_val])[0]
            else:
                row[col] = le.transform(["nan"])[0]
        else:
            row[col] = params.get(col, 0)

    X = pd.DataFrame([row])

    predictions = {}
    for target, model in models.items():
        pred = model.predict(X)[0]
        pred = max(0.0, min(100.0, pred))  # Clamp to valid range
        predictions[target] = round(pred, 1)

    return predictions


def predict_all_chelators(models, encoders, feature_info, base_params, doses):
    """
    Predict % free for all chelators at all doses.
    Returns a DataFrame with results for comparison.
    """
    chelators = ["nan", "EDTA", "NTA", "Citrate", "Humic", "Fulvic"]
    results = []

    for chelator in chelators:
        dose_list = [0] if chelator == "nan" else doses
        for dose in dose_list:
            params = base_params.copy()
            params["chelator"] = chelator
            params["dose_mg_L"] = dose

            preds = predict_single(models, encoders, feature_info, params)

            row = {
                "Chelator": "No Treatment" if chelator == "nan" else chelator,
                "Dose (mg/L)": dose,
            }
            for target, info in METALS.items():
                row[info["symbol"] + " % Free"] = preds[target]
            results.append(row)

    return pd.DataFrame(results)


def get_effectiveness_label(pct_free):
    """Convert % free to a human-readable effectiveness rating."""
    if pct_free <= 5:
        return "Excellent", "🟢"
    elif pct_free <= 20:
        return "Good", "🟡"
    elif pct_free <= 50:
        return "Moderate", "🟠"
    else:
        return "Poor", "🔴"


def find_best_recommendation(comparison_df):
    """
    Find the best chelator/dose for each metal.
    Returns dict of recommendations.
    """
    recommendations = {}
    # Exclude the "None" baseline
    treated = comparison_df[comparison_df["Chelator"] != "No Treatment"].copy()
    baseline = comparison_df[comparison_df["Chelator"] == "No Treatment"].iloc[0]

    for target, info in METALS.items():
        col = info["symbol"] + " % Free"
        best_idx = treated[col].idxmin()
        best_row = treated.loc[best_idx]
        baseline_val = baseline[col]
        best_val = best_row[col]
        reduction = baseline_val - best_val

        recommendations[target] = {
            "chelator": best_row["Chelator"],
            "dose": best_row["Dose (mg/L)"],
            "predicted_free": best_val,
            "baseline_free": baseline_val,
            "reduction": reduction,
            "effectiveness": get_effectiveness_label(best_val),
        }

    return recommendations


# ============================================================
# STREAMLIT APP
# ============================================================

def main():
    st.set_page_config(
        page_title="Chelator Remediation Tool",
        page_icon="🧪",
        layout="wide",
    )

    # --- Header ---
    st.title("🧪 Heavy Metal Chelation Remediation Tool")
    st.markdown(
        "**Rhode Island Coastal Soil Decision Support System**  \n"
        "Enter your site conditions below to receive chelator recommendations "
        "for reducing heavy metal bioavailability in contaminated soils."
    )
    st.markdown("---")

    # Load models
    models, encoders, feature_info = load_models()
    if models is None:
        st.stop()

    # ============================================================
    # SIDEBAR: Site Condition Inputs
    # ============================================================
    st.sidebar.header("📋 Site Conditions")
    st.sidebar.markdown("Enter your soil test results below.")

    # --- pH ---
    st.sidebar.subheader("Soil pH")
    ph = st.sidebar.slider(
        "Soil pore water pH",
        min_value=5.0, max_value=8.5, value=6.5, step=0.1,
        help="Measured pH of soil pore water or paste extract. "
             "Lower pH = more metal mobility. Most RI soils: 5.5-7.5"
    )

    # --- Metal Concentrations ---
    st.sidebar.subheader("Metal Concentrations (mg/L)")
    st.sidebar.markdown("*From soil pore water or extraction test*")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        pb_mg = st.number_input("Lead (Pb)", min_value=0.0, max_value=1000.0,
                                value=100.0, step=10.0)
        zn_mg = st.number_input("Zinc (Zn)", min_value=0.0, max_value=1000.0,
                                value=120.0, step=10.0)
    with col2:
        cu_mg = st.number_input("Copper (Cu)", min_value=0.0, max_value=1000.0,
                                value=80.0, step=10.0)
        cd_mg = st.number_input("Cadmium (Cd)", min_value=0.0, max_value=100.0,
                                value=8.0, step=1.0)

    # --- Soil Texture (sand/silt/clay %) ---
    st.sidebar.subheader("Soil Texture")
    st.sidebar.markdown("*Enter particle size percentages (must sum to 100%)*")

    sand_pct = st.sidebar.slider("Sand %", 0, 100, 40)
    silt_pct = st.sidebar.slider("Silt %", 0, 100, 40)
    clay_pct = 100 - sand_pct - silt_pct

    if clay_pct < 0:
        st.sidebar.error(f"Sand + Silt = {sand_pct + silt_pct}% (exceeds 100%). "
                         "Please adjust.")
        st.stop()

    st.sidebar.markdown(f"**Clay: {clay_pct}%** (calculated)")

    texture_class, texture_reason = classify_texture(sand_pct, silt_pct, clay_pct)
    tex_props = get_texture_properties(texture_class)
    st.sidebar.info(f"Texture class: **{texture_class}**  \n{texture_reason}")

    # --- Organic Matter ---
    st.sidebar.subheader("Organic Matter")
    om_pct = st.sidebar.slider(
        "Organic Matter (%)",
        min_value=0.5, max_value=15.0, value=3.0, step=0.5,
        help="From loss-on-ignition or Walkley-Black test"
    )
    # Note: DOC in training data is tied to texture, so we use the texture-based
    # value for prediction. The OM% is displayed for context.
    doc_display = om_pct * 10 * 0.58  # Approximate conversion

    # --- Moisture / Drainage ---
    st.sidebar.subheader("Moisture Condition")
    moisture = st.sidebar.selectbox(
        "Site drainage condition",
        options=["Dry", "Mesic", "Wet"],
        index=1,  # Default to Mesic
        help="Dry = well-drained upland, Mesic = moderate drainage, "
             "Wet = poorly drained/near water table"
    )
    pe = get_moisture_pe(moisture)

    # --- Salinity / Ionic Strength ---
    st.sidebar.subheader("Salinity Conditions")
    ionic_level = st.sidebar.selectbox(
        "Salinity level",
        options=["Low", "Medium", "High"],
        index=0,
        help="Low = inland site, Medium = moderate salt influence, "
             "High = coastal or road salt-impacted"
    )
    ionic_vals = get_ionic_values(ionic_level)

    # --- Competing Cations ---
    st.sidebar.subheader("Competing Cations")
    ca_mg_level = st.sidebar.selectbox(
        "Calcium/Magnesium level",
        options=["Low", "High"],
        index=0,
        help="High Ca/Mg competes with heavy metals for chelator binding. "
             "High = calcareous or limed soils"
    )
    ca_mg_vals = get_ca_mg_values(ca_mg_level)

    # ============================================================
    # BUILD BASE PARAMETERS
    # ============================================================
    base_params = {
        "ph": ph,
        "pb_mg_L": pb_mg,
        "cu_mg_L": cu_mg,
        "zn_mg_L": zn_mg,
        "cd_mg_L": cd_mg,
        "doc_mg_L": tex_props["doc_mg_L"],
        "ca_mg_L": ca_mg_vals["ca_mg_L"],
        "mg_mg_L": ca_mg_vals["mg_mg_L"],
        "na_mg_L": ionic_vals["na_mg_L"],
        "cl_mg_L": ionic_vals["cl_mg_L"],
        "hfo_sites": tex_props["hfo_sites"],
        "pe": pe,
    }

    # ============================================================
    # MAIN CONTENT: Results
    # ============================================================

    # Run predictions for all chelators at standard doses
    doses = [50, 150, 300]
    comparison_df = predict_all_chelators(
        models, encoders, feature_info, base_params, doses
    )

    # Get best recommendations
    recommendations = find_best_recommendation(comparison_df)

    # --- Tab layout ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Recommendations",
        "📊 Full Comparison",
        "🔬 Site Summary",
        "ℹ️ About"
    ])

    # ============================================================
    # TAB 1: Recommendations
    # ============================================================
    with tab1:
        st.header("Recommended Treatments")
        st.markdown("Best chelator and dose for each metal at your site conditions:")

        for target, info in METALS.items():
            rec = recommendations[target]
            eff_label, eff_icon = rec["effectiveness"]

            with st.container():
                st.subheader(f"{info['name']}")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        label="Best Chelator",
                        value=rec["chelator"],
                    )
                with c2:
                    st.metric(
                        label="Recommended Dose",
                        value=f"{rec['dose']:.0f} mg/L",
                    )
                with c3:
                    st.metric(
                        label="Predicted % Free",
                        value=f"{rec['predicted_free']:.1f}%",
                        delta=f"-{rec['reduction']:.1f}%",
                        delta_color="inverse",  # Green for negative (reduction is good)
                    )
                with c4:
                    st.metric(
                        label="Effectiveness",
                        value=f"{eff_icon} {eff_label}",
                    )

                # Baseline comparison
                st.caption(
                    f"Without treatment: {rec['baseline_free']:.1f}% free → "
                    f"With {rec['chelator']} {rec['dose']:.0f} mg/L: "
                    f"{rec['predicted_free']:.1f}% free "
                    f"({rec['reduction']:.1f} percentage point reduction)"
                )
                st.markdown("---")

        # Warning flags
        st.subheader("⚠️ Warnings & Notes")
        warnings_list = []

        zn_rec = recommendations["zn_percent_free"]
        if zn_rec["predicted_free"] > 50:
            warnings_list.append(
                "**Zinc** remains highly bioavailable even with optimal chelation. "
                "Zinc binds weakly to most chelating agents. Consider additional "
                "treatment approaches (e.g., lime amendment to raise pH, "
                "phosphate stabilization, or phytoremediation)."
            )

        if ph < 6.0:
            warnings_list.append(
                f"**Low pH ({ph})** significantly reduces chelation effectiveness. "
                "Consider lime amendment to raise pH to 6.5-7.0 before chelator application."
            )

        if ionic_level == "High":
            warnings_list.append(
                "**High salinity** conditions detected. While chloride complexes reduce "
                "free metal concentrations, the resulting metal-chloride species are still "
                "mobile. Monitor leachate and groundwater."
            )

        for target, rec in recommendations.items():
            eff_label, _ = rec["effectiveness"]
            if eff_label == "Poor":
                metal_name = METALS[target]["name"]
                warnings_list.append(
                    f"**{metal_name}** shows poor response to chelation under these conditions. "
                    "Consider alternative remediation strategies."
                )

        if not warnings_list:
            st.success("No warnings for current site conditions.")
        else:
            for w in warnings_list:
                st.warning(w)

    # ============================================================
    # TAB 2: Full Comparison Table
    # ============================================================
    with tab2:
        st.header("Full Chelator Comparison")
        st.markdown(
            "Predicted % free metal for every chelator and dose combination. "
            "**Lower is better** (less bioavailable metal)."
        )

        # Style the dataframe - highlight the lowest values
        def highlight_min(s):
            """Highlight minimum value in each column."""
            is_min = s == s.min()
            return ['background-color: #C8E6C9' if v else '' for v in is_min]

        metal_cols = [info["symbol"] + " % Free" for info in METALS.values()]
        styled_df = comparison_df.style.apply(
            highlight_min, subset=metal_cols
        ).format({col: "{:.1f}" for col in metal_cols})

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        st.caption(
            "Green highlighted cells = best (lowest) value for that metal. "
            "The 'None' row shows predicted bioavailability without any chelator treatment."
        )

        # --- Bar chart comparison ---
        st.subheader("Visual Comparison at 300 mg/L Dose")

        high_dose = comparison_df[
            (comparison_df["Dose (mg/L)"] == 300) |
            (comparison_df["Chelator"] == "No Treatment")
        ].copy()

        if not high_dose.empty:
            chart_data = high_dose.set_index("Chelator")[metal_cols]
            st.bar_chart(chart_data, height=400)
            st.caption("Lower bars = more effective treatment (less free metal)")

    # ============================================================
    # TAB 3: Site Summary
    # ============================================================
    with tab3:
        st.header("Site Condition Summary")
        st.markdown("Review of the parameters used for predictions:")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Soil Properties")
            st.markdown(f"**pH:** {ph}")
            st.markdown(f"**Texture:** {texture_class} "
                        f"({sand_pct}% sand, {silt_pct}% silt, {clay_pct}% clay)")
            st.markdown(f"**Organic Matter:** {om_pct}% "
                        f"(≈ {doc_display:.0f} mg/L DOC)")
            st.markdown(f"**Moisture:** {moisture} (pe = {pe})")
            st.markdown(f"**Iron Oxide Sites:** {tex_props['hfo_sites']} mol "
                        f"(from {texture_class} texture)")

        with c2:
            st.subheader("Water Chemistry")
            st.markdown(f"**Lead (Pb):** {pb_mg} mg/L")
            st.markdown(f"**Copper (Cu):** {cu_mg} mg/L")
            st.markdown(f"**Zinc (Zn):** {zn_mg} mg/L")
            st.markdown(f"**Cadmium (Cd):** {cd_mg} mg/L")
            st.markdown(f"**Salinity:** {ionic_level} "
                        f"(Na = {ionic_vals['na_mg_L']}, "
                        f"Cl = {ionic_vals['cl_mg_L']} mg/L)")
            st.markdown(f"**Ca/Mg Competition:** {ca_mg_level} "
                        f"(Ca = {ca_mg_vals['ca_mg_L']}, "
                        f"Mg = {ca_mg_vals['mg_mg_L']} mg/L)")

        st.markdown("---")
        st.subheader("Model Information")
        st.markdown(
            "Predictions are generated by Gradient Boosting models trained on "
            "12,636 PHREEQC geochemical simulations covering realistic Rhode Island "
            "coastal soil conditions. The models achieve R² > 0.95 with RMSE < 1% "
            "for all four metals on cross-validation."
        )

    # ============================================================
    # TAB 4: About
    # ============================================================
    with tab4:
        st.header("About This Tool")

        st.markdown("""
        ### Purpose
        This tool helps environmental practitioners select the most effective 
        chelating agent for heavy metal remediation in contaminated soils, 
        with a focus on Rhode Island coastal environments.

        ### How It Works
        1. **PHREEQC Simulations**: 12,636 geochemical simulations were run using 
           the USGS PHREEQC model with the minteq.v4.dat thermodynamic database, 
           covering realistic combinations of soil pH, metal concentrations, 
           soil texture, moisture conditions, salinity, and chelator treatments.

        2. **Machine Learning**: Gradient Boosting models were trained on the 
           simulation results to predict % free (bioavailable) metal fraction 
           for Lead, Copper, Zinc, and Cadmium under any combination of conditions.

        3. **Recommendations**: The tool compares all available chelators at 
           multiple doses and recommends the treatment predicted to most 
           effectively reduce metal bioavailability at your specific site.

        ### Chelating Agents
        """)

        for name, info in CHELATOR_INFO.items():
            bio = "✅ Biodegradable" if info["biodegradable"] else "⚠️ Persistent"
            st.markdown(f"**{name}** ({info['full_name']})  \n"
                        f"{bio} — {info['notes']}")

        st.markdown("""
        ### Key Concepts
        - **% Free Metal**: The fraction of dissolved metal present as free ions 
          (e.g., Pb²⁺). This is the most bioavailable and mobile form. 
          Lower = better remediation.
        - **Chelation**: Chelators bind free metal ions into stable complexes, 
          reducing bioavailability and mobility.
        - **Surface Sorption**: Soil iron/aluminum oxides naturally bind metals. 
          Clay soils have more sorption capacity than sandy soils.

        ### Limitations
        - Predictions are based on equilibrium thermodynamic modeling and may not 
          capture kinetic effects in the field.
        - Humic and Fulvic acid predictions are approximate (modeled as DOC additions 
          rather than discrete complexing agents in PHREEQC).
        - Metal concentrations outside the training range (Pb > 300, Cu > 250, 
          Zn > 400, Cd > 25 mg/L) may produce less reliable predictions.
        - This tool does not account for soil heterogeneity, preferential flow paths, 
          or biological activity.

        ### Citation
        Developed by Mallory Malz, University of Rhode Island.  
        PHREEQC modeling with minteq.v4.dat thermodynamic database.  
        Machine learning models: scikit-learn Gradient Boosting Regressor.
        """)


if __name__ == "__main__":
    main()
