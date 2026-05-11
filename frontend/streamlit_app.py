import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ---------------- MODEL LOAD ----------------

MODEL_PATH = "../ML_Model/concrete_strength_model.pkl"
model = joblib.load(MODEL_PATH)

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Concrete Strength Predictor",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

.stButton>button {
    background-color: #FF4B4B;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

section[data-testid="stSidebar"] {
    background-color: #1E1E1E;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("🏗️ AI-Based Concrete Strength Predictor")
st.markdown("### End-to-End Machine Learning System for Concrete Mix Analysis")

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("📥 Input Parameters")

sap = st.sidebar.slider("SAP %", 0.0, 1.0, 0.2, 0.01)

wc = st.sidebar.slider(
    "Water-Cement Ratio",
    0.30,
    0.50,
    0.38,
    0.01
)

age = st.sidebar.selectbox(
    "Age (days)",
    [7, 14, 28, 56, 90]
)

weight = st.sidebar.number_input(
    "Weight (g)",
    1000,
    5000,
    2500
)

area = st.sidebar.number_input(
    "Area (mm²)",
    1000,
    20000,
    10000
)

slump = st.sidebar.number_input(
    "Slump (mm)",
    0,
    200,
    70
)

# ---------------- INPUT DATAFRAME ----------------
input_df = pd.DataFrame([{
    'SAP_%': sap,
    'W_C_Ratio': wc,
    'Weight_g': weight,
    'Area_mm2': area,
    'Slump_mm': slump,
    'Age_days': age
}])

# ---------------- INPUT SUMMARY ----------------
st.subheader("📋 Input Summary")
st.dataframe(input_df, use_container_width=True)

# ---------------- GAUGE FUNCTION ----------------
def show_gauge(value):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Strength (MPa)"},
        gauge={
            'axis': {'range': [0, 60]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 25], 'color': "#808080"},
                {'range': [25, 40], 'color': "#505050"},
                {'range': [40, 60], 'color': "#303030"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Strength"):

    try:

        # Predict directly using model
        strength = round(float(model.predict(input_df)[0]), 2)

        # ---------------- METRICS ----------------
        col1, col2, col3 = st.columns(3)

        col1.metric("Predicted Strength", f"{strength} MPa")
        col2.metric("Age", f"{age} days")

        # ---------------- QUALITY ----------------
        if strength > 40:
            quality = "High 💪"
            col3.metric("Quality", quality)
            st.success("💪 High Strength Concrete")

        elif strength > 25:
            quality = "Medium ⚖️"
            col3.metric("Quality", quality)
            st.warning("⚖️ Medium Strength Concrete")

        else:
            quality = "Low ⚠️"
            col3.metric("Quality", quality)
            st.error("⚠️ Low Strength Concrete")

        # ---------------- GAUGE ----------------
        st.subheader("🎯 Strength Gauge")
        show_gauge(strength)

        # =========================================================
        # STRENGTH VS AGE
        # =========================================================
        st.subheader("📈 Strength vs Age")

        age_range = [7, 14, 28, 56, 90]

        temp_df = pd.DataFrame({
            'SAP_%': sap,
            'W_C_Ratio': wc,
            'Weight_g': weight,
            'Area_mm2': area,
            'Slump_mm': slump,
            'Age_days': age_range
        })

        pred_strength = model.predict(temp_df)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(age_range, pred_strength,
                marker='o',
                linewidth=2)

        ax.set_xlabel("Age (days)")
        ax.set_ylabel("Strength (MPa)")
        ax.set_title("Strength Growth Over Time")
        ax.grid(True)

        st.pyplot(fig)

        # =========================================================
        # STRENGTH VS W/C RATIO
        # =========================================================
        st.subheader("📈 Strength vs W/C Ratio")

        wc_range = np.linspace(0.34, 0.46, 30)

        temp_df = pd.DataFrame({
            'SAP_%': sap,
            'W_C_Ratio': wc_range,
            'Weight_g': weight,
            'Area_mm2': area,
            'Slump_mm': slump,
            'Age_days': age
        })

        pred_strength = model.predict(temp_df)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(wc_range, pred_strength,
                linewidth=2)

        ax.set_xlabel("W/C Ratio")
        ax.set_ylabel("Strength (MPa)")
        ax.set_title("Effect of W/C Ratio on Strength")
        ax.grid(True)

        st.pyplot(fig)

        # =========================================================
        # STRENGTH VS SAP
        # =========================================================
        st.subheader("📈 Strength vs SAP %")

        sap_range = np.linspace(0.1, 0.6, 30)

        temp_df = pd.DataFrame({
            'SAP_%': sap_range,
            'W_C_Ratio': wc,
            'Weight_g': weight,
            'Area_mm2': area,
            'Slump_mm': slump,
            'Age_days': age
        })

        pred_strength = model.predict(temp_df)

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(sap_range,
                pred_strength,
                linewidth=2,
                color='orange')

        ax.set_xlabel("SAP %")
        ax.set_ylabel("Strength (MPa)")
        ax.set_title("Effect of SAP on Strength")
        ax.grid(True)

        st.pyplot(fig)

        # =========================================================
        # 3D VISUALIZATION
        # =========================================================
        st.subheader("🌐 3D Visualization")

        wc_values = np.linspace(0.34, 0.46, 15)
        sap_values = np.linspace(0.1, 0.6, 15)

        data_list = []

        for wc_val in wc_values:
            for sap_val in sap_values:

                temp_input = pd.DataFrame([{
                    'SAP_%': sap_val,
                    'W_C_Ratio': wc_val,
                    'Weight_g': weight,
                    'Area_mm2': area,
                    'Slump_mm': slump,
                    'Age_days': age
                }])

                pred = model.predict(temp_input)[0]

                data_list.append({
                    'W_C_Ratio': wc_val,
                    'SAP_%': sap_val,
                    'Strength': pred
                })

        plot_df = pd.DataFrame(data_list)

        fig = px.scatter_3d(
            plot_df,
            x='W_C_Ratio',
            y='SAP_%',
            z='Strength',
            color='Strength',
            size='Strength',
            title='3D Strength Prediction Surface'
        )

        fig.update_layout(
            scene=dict(
                xaxis_title='W/C Ratio',
                yaxis_title='SAP %',
                zaxis_title='Strength (MPa)'
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================================================
        # KEY INSIGHTS
        # =========================================================
        st.subheader("🧠 Key Insights")

        st.info(f"""
        • Increasing curing age improves concrete strength  
        • Higher W/C ratio generally reduces strength  
        • SAP dosage shows non-linear influence on strength  
        • Model predicts compressive strength using ML techniques  
        """)

        # =========================================================
        # DOWNLOAD REPORT
        # =========================================================
        report = f"""
AI Concrete Strength Prediction Report

-------------------------------------
INPUT PARAMETERS
-------------------------------------

SAP %           : {sap}
W/C Ratio       : {wc}
Age             : {age} days
Weight          : {weight} g
Area            : {area} mm²
Slump           : {slump} mm

-------------------------------------
PREDICTION
-------------------------------------

Predicted Strength : {strength} MPa
Concrete Quality   : {quality}

-------------------------------------
Generated using AI-Based ML Model
-------------------------------------
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="Concrete_Strength_Report.txt"
        )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")