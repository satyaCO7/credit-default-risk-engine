import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from credit_engine import train_credit_model, predict_and_explain

st.set_page_config(page_title="Retail Credit Risk Engine", layout="wide")

# Cache the model training so it only runs once when the app boots
@st.cache_resource
def load_engine():
    return train_credit_model()

model, explainer, feature_names = load_engine()

st.title(" Institutional Credit Default Prediction Engine")
st.markdown("An end-to-end XGBoost machine learning pipeline with integrated SHAP (SHapley Additive exPlanations) to predict consumer credit defaults and automatically generate regulatory reason codes for lending decisions.")

# --- SIDEBAR: CUSTOMER INPUTS ---
st.sidebar.header("Applicant Financial Profile")
c_score = st.sidebar.slider("Credit Score", min_value=300, max_value=850, value=650, step=10)
c_income = st.sidebar.slider("Annual Income ($)", min_value=20000, max_value=200000, value=65000, step=1000)
c_util = st.sidebar.slider("Credit Utilization (%)", min_value=0.0, max_value=1.0, value=0.40, step=0.05)
c_dti = st.sidebar.slider("Debt-to-Income Ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.05)
c_missed = st.sidebar.slider("Missed Payments (Last 12M)", min_value=0, max_value=5, value=0, step=1)

customer_data = [c_score, c_income, c_util, c_dti, c_missed]

# --- EXECUTION & EXPLAINABILITY ---
if st.button("Run Credit Risk Assessment"):
    with st.spinner("Executing XGBoost decision trees and calculating SHAP values..."):
        prob_default, shap_vals, customer_df = predict_and_explain(model, explainer, feature_names, customer_data)
        
        st.markdown("---")
        
        # Determine risk category
        if prob_default < 0.20:
            risk_tier = "🟢 LOW RISK (Approve)"
        elif prob_default < 0.50:
            risk_tier = "🟡 MODERATE RISK (Manual Review)"
        else:
            risk_tier = "🔴 HIGH RISK (Decline)"
            
        # Display Metrics
        col1, col2 = st.columns(2)
        col1.metric("Calculated Probability of Default", f"{prob_default * 100:.2f}%")
        col2.metric("System Recommendation", risk_tier)
        
        st.markdown("###  SHAP Explainability: Decision Drivers")
        st.markdown("This chart explains *why* the XGBoost model made its decision. Red bars push the probability of default higher (increasing risk). Blue bars push the probability lower (decreasing risk).")
        
        # Plotting the SHAP values using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 4))
        
        colors = ['red' if val > 0 else 'blue' for val in shap_vals]
        y_pos = np.arange(len(feature_names))
        
        ax.barh(y_pos, shap_vals, color=colors, edgecolor='black', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel("Impact on Default Probability (Log Odds)")
        ax.set_title("Local Feature Importance (SHAP Values)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        st.pyplot(fig)