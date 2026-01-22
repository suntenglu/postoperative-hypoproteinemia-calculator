# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# =====================================================
# Page configuration
# =====================================================
st.set_page_config(
    page_title="Postoperative Persistent Hypoproteinemia Risk Calculator",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "rf_pipeline.joblib")

# =====================================================
# ENGLISH DISPLAY NAME  ->  CHINESE MODEL COLUMN NAME
# (UI uses English; the model still receives CN column names)
# =====================================================
FEATURE_MAP = {
    "Carcinoembryonic antigen (CEA)": "癌胚抗原",
    "Total cholesterol": "胆固醇",
    "Calcium": "钙",
    "Triglycerides": "甘油三酯",
    "Indirect bilirubin": "间接胆红素",
    "Magnesium": "镁",
    "Uric acid": "尿酸",
    "Prothrombin time (PT)": "凝血酶原时间",
    "Hemoglobin": "血红蛋白",
    "Total bilirubin": "总胆红素",
}
FEATURES_CN = list(FEATURE_MAP.values())

# =====================================================
# Load model
# =====================================================
@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict) and "pipeline" in bundle:
        return bundle["pipeline"]
    return bundle

pipe = load_model()

# =====================================================
# UI - Header
# =====================================================
st.title("Postoperative Persistent Hypoproteinemia Risk Calculator")
st.caption("Based on a Random Forest machine learning model")

st.write(
    "Enter the following clinical indicators to estimate the risk probability "
    "of postoperative persistent hypoproteinemia."
)

# =====================================================
# Input section (English labels)
# =====================================================
st.subheader("Input variables")

inputs_en = {}
for en_name in FEATURE_MAP.keys():
    inputs_en[en_name] = st.number_input(
        label=en_name,
        value=float("nan"),
        step=0.1,
        format="%.4f"
    )

# =====================================================
# Prediction (Probability only; no threshold / no risk category)
# =====================================================
st.subheader("Prediction")

if st.button("Predict", type="primary"):
    missing = [k for k, v in inputs_en.items() if np.isnan(v)]
    if missing:
        st.error("The following variables are missing: " + ", ".join(missing))
        st.stop()

    # Map EN input -> CN model columns
    row_cn = {FEATURE_MAP[en]: float(inputs_en[en]) for en in FEATURE_MAP}

    # Ensure column order matches training
    X = pd.DataFrame([row_cn])[FEATURES_CN]

    # Predict probability of positive class
    prob = float(pipe.predict_proba(X)[:, 1][0])

    st.success(f"Predicted risk probability: {prob*100:.2f}%")

    st.markdown("---")
    st.write("Input values used for prediction:")
    st.dataframe(X, use_container_width=True)

st.markdown("---")
st.caption(
    "Note: This tool is intended for research and clinical decision support only "
    "and should not replace clinical judgment."
)
