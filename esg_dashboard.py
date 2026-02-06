import streamlit as st

# --- Custom Green & White Theme ---
st.markdown(
    """
    <link href='https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Roboto:wght@400;500&display=swap' rel='stylesheet'>
    <style>
    html, body, .stApp {
        background: #e8f5e9 !important;
        font-family: 'Montserrat', 'Roboto', sans-serif !important;
        color: #184d27 !important;
    }
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }
    .stTitle, .stSubheader, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        font-family: 'Montserrat', sans-serif !important;
        color: #217a3a !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
    }
    .stSubheader {
        font-size: 1.3rem !important;
        margin-bottom: 0.5em !important;
    }
    .stTitle {
        font-size: 2.2rem !important;
        margin-bottom: 0.2em !important;
    }
    .stDivider {
        border-top: 2px solid #217a3a !important;
        margin-bottom: 1.5em !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #217a3a 60%, #43a047 100%) !important;
        color: #fff !important;
        border-radius: 12px !important;
        border: none !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6em 2em;
        box-shadow: 0 2px 8px rgba(33,122,58,0.10);
        transition: background 0.2s, box-shadow 0.2s;
    }
    .stButton>button:hover {
        background: #184d27 !important;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(33,122,58,0.18);
    }
    .stNumberInput, .stSlider, .stTextInput, .stSelectbox {
        background: #e0ede2 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 12px rgba(33,122,58,0.13);
        font-family: 'Roboto', sans-serif !important;
        border: 1.5px solid #388e3c !important;
    }
    /* Target the input field backgrounds inside widgets */
    .stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background: #fff !important;
        color: #111 !important;
        border-radius: 10px !important;
    }
    .stSlider > div {
        background: #e0ede2 !important;
        border-radius: 10px !important;
    }
    /* Make input labels and slider labels dark */
    label, .stSlider label, .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #111 !important;
        font-weight: 600 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1.05rem !important;
    }
    label, .stSlider label, .stNumberInput label, .stTextInput label, .stSelectbox label {
        color: #184d27 !important;
        font-weight: 600 !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1.05rem !important;
    }
    .stExpanderHeader {
        color: #217a3a !important;
        font-weight: 700;
        font-family: 'Montserrat', sans-serif !important;
    }
    .stMetric {
        background: #e0ede2 !important;
        border-radius: 16px !important;
        box-shadow: 0 2px 12px rgba(33,122,58,0.10);
        padding: 0.7em 0.7em 0.7em 1.2em !important;
        font-family: 'Montserrat', sans-serif !important;
        color: #111 !important;
        border: 1.5px solid #8bc49b !important;
    }
    /* Ensure metric label is fully visible and not ellipsized */
    div[data-testid="stMetric"] > div:first-child {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        color: #111 !important;
        font-size: 1.05rem !important;
        font-family: 'Montserrat', sans-serif !important;
        line-height: 1.2 !important;
        word-break: break-word !important;
    }
    /* Make metric value text black and bold */
    div[data-testid="stMetricValue"] {
        color: #111 !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        font-family: 'Montserrat', sans-serif !important;
    }
    .stMarkdown, .stText, .stWrite, .stDataFrame, .stTable {
        color: #184d27 !important;
        font-family: 'Roboto', sans-serif !important;
        font-size: 1.08rem;
    }
    .stAlert, .stSuccess, .stWarning, .stError {
        border-radius: 14px !important;
        font-family: 'Montserrat', sans-serif !important;
        font-size: 1.08rem;
        background: #e8f5e9 !important;
        color: #184d27 !important;
        border: 1.5px solid #43a047 !important;
    }
    .stContainer, .stColumn {
        background: #e0ede2 !important;
        border-radius: 20px !important;
        box-shadow: 0 4px 24px rgba(33,122,58,0.13);
        padding: 1.2em 1.5em 1.2em 1.5em !important;
        margin-bottom: 1.5em !important;
        border: 1.5px solid #8bc49b !important;
    }
    .stExpander {
        margin-top: 1.2em !important;
        background: #e0ede2 !important;
        border-radius: 16px !important;
        border: 1.5px solid #8bc49b !important;
    }
    /* Hide Streamlit branding */
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Use a pastel theme for Seaborn plots
sns.set_theme(style='whitegrid', palette='pastel')

# Load model and encoder
with open("esg_model.pkl", "rb") as f:
    model = pickle.load(f)
st.set_page_config(page_title="ESG Risk Dashboard", layout="wide")
with open("esg_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)



# --- Header Section ---

# --- Header Section ---
st.title("🌱 ESG Risk Scoring Dashboard")
st.markdown("<span style='font-size:1.15rem; color:#256029; font-family:Montserrat,sans-serif;'>Interactive ESG risk assessment for companies using environmental data.</span>", unsafe_allow_html=True)
st.divider()

# --- Main Layout: White Cards ---
left, right = st.columns([1, 2], gap="large")

# ========== LEFT PANEL (INPUTS) ========== 
with left:
    with st.container():
        st.subheader("📥 Company Inputs")
        co2 = st.number_input("CO₂ Emissions", min_value=0.0, value=500.0)
        energy = st.number_input("Energy Consumption", min_value=1.0, value=300.0)
        renewable = st.slider("Renewable Energy Usage (%)", 0, 100, 40)
        water = st.number_input("Water Usage", min_value=0.0, value=200.0)
        waste = st.number_input("Waste Generated", min_value=0.0, value=150.0)
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔍 Predict ESG Risk", use_container_width=True)

# ========== FEATURE ENGINEERING ==========
emission_intensity = co2 / energy
renewable_risk = 1 - (renewable / 100)
water_risk = water / 500
waste_risk = waste / 500

# Only use the features the model was trained on for prediction
X_input = np.array([[emission_intensity, water_risk]])

# ========== RIGHT PANEL (OUTPUTS) ========== 
with right:
    with st.container():
        if predict_btn:
            prediction = model.predict(X_input)
            risk = encoder.inverse_transform(prediction)[0]

            st.subheader("📊 ESG Risk Result")

            if risk == "Low Risk":
                st.success("✅ LOW ESG RISK")
            elif risk == "Medium Risk":
                st.warning("⚠️ MEDIUM ESG RISK")
            else:
                st.error("❌ HIGH ESG RISK")

            # ---------- QUICK METRICS ----------
            col1, col2, col3, col4 = st.columns(4, gap="small")
            col1.metric("Emission Intensity", f"{emission_intensity:.2f}")
            col2.metric("Renewable %", f"{renewable}%")
            col3.metric("Water Risk", f"{water_risk:.2f}")
            col4.metric("Waste Risk", f"{waste_risk:.2f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------- RISK FACTOR BAR CHART ----------
            st.subheader("📈 ESG Risk Factor Breakdown")

            risk_df = pd.DataFrame(
                {
                    "Risk Factor": [
                        "Emission Intensity",
                        "Renewable Risk",
                        "Water Risk",
                        "Waste Risk",
                    ],
                    "Risk Level": [
                        emission_intensity,
                        renewable_risk,
                        water_risk,
                        waste_risk,
                    ],
                }
            )

            fig, ax = plt.subplots(figsize=(7,4))
            sns.barplot(x="Risk Level", y="Risk Factor", data=risk_df, palette=["#388e3c", "#66bb6a", "#a5d6a7", "#e8f5e9"], ax=ax)
            ax.set_xlabel("Risk Level (Higher = Worse)")
            ax.set_title("Environmental Risk Factors")
            for spine in ax.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

            # ---------- COMPARISON CHART ----------
            st.subheader("📉 Comparison with Safe Threshold")

            threshold_df = pd.DataFrame(
                {
                    "Metric": [
                        "Emission Intensity",
                        "Renewable Risk",
                        "Water Risk",
                        "Waste Risk",
                    ],
                    "Your Company": [
                        emission_intensity,
                        renewable_risk,
                        water_risk,
                        waste_risk,
                    ],
                    "Safe Threshold": [0.3, 0.3, 0.3, 0.3],
                }
            )

            long_df = threshold_df.melt(id_vars="Metric", var_name="Type", value_name="Value")

            fig2, ax2 = plt.subplots(figsize=(8,4))
            sns.barplot(x="Metric", y="Value", hue="Type", data=long_df, palette=["#388e3c", "#e8f5e9"], ax=ax2)
            ax2.set_ylabel("Risk Level")
            ax2.set_title("Company vs Safe ESG Threshold")
            ax2.legend(title="")
            for spine in ax2.spines.values():
                spine.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)

            # ---------- RECOMMENDATIONS ----------
            with st.expander("🌿 Sustainability Recommendations", expanded=True):
                if renewable < 50:
                    st.write("• Increase renewable energy usage")
                if co2 > 800:
                    st.write("• Reduce carbon emissions")
                if water > 300:
                    st.write("• Optimize water consumption")
                if waste > 300:
                    st.write("• Improve waste recycling practices")
                if renewable >= 50 and co2 <= 800 and water <= 300 and waste <= 300:
                    st.write("• Keep up the good work — your metrics are within recommended thresholds ✅")
