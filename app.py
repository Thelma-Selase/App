
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import statsmodels.api as sm

st.set_page_config(page_title="Patient Health Risk Assessment", page_icon="🏥", layout="centered")

rf_model    = joblib.load('rf_model.pkl')
reg_model   = sm.load('regression_model.pkl')
rf_features  = joblib.load('rf_feature_names.pkl')
reg_features = joblib.load('regression_feature_names.pkl')

st.title("🏥 Patient Health Risk Assessment Tool")
st.markdown("Enter patient details at discharge to get a **predicted Health Score** and **30-day Readmission Risk**.")
st.markdown("---")
st.subheader("Patient Details")

age                  = st.number_input("Age (years)", min_value=18, max_value=100, value=52)
smoking              = st.selectbox("Smoking Status", ["No", "Yes"])
physical_activity    = st.selectbox("Physical Activity Level", ["Sedentary", "Low", "Moderate", "High"])
sleep_hours          = st.slider("Sleep Hours per Night", min_value=3.0, max_value=10.0, value=6.5, step=0.5)
medication_adherence = st.slider("Medication Adherence (1=Poor, 10=Excellent)", min_value=1, max_value=10, value=6)
bmi                  = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
sbp                  = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=140)
glucose              = st.number_input("Blood Glucose (mg/dL)", min_value=40, max_value=300, value=95)
length_stay          = st.number_input("Length of Hospital Stay (days)", min_value=1, max_value=30, value=5)
chronic_disease      = st.selectbox("Chronic Disease", ["No", "Yes"])
severity             = st.selectbox("Disease Severity", ["Mild", "Moderate", "Severe", "Critical"])
health_awareness     = st.selectbox("Health Awareness Level", ["Very Low", "Low", "Moderate", "High", "Very High"])
diet_quality         = st.selectbox("Diet Quality", ["Very Poor", "Poor", "Fair", "Good", "Excellent"])

st.markdown("---")

if st.button("Generate Risk Assessment"):

    reg_dict = {col: 0 for col in reg_features if col != "const"}
    reg_dict["Age"]                 = age
    reg_dict["MedicationAdherence"] = medication_adherence
    reg_dict["SleepHours"]          = sleep_hours
    reg_dict["LengthStay"]          = length_stay
    reg_dict["SBP"]                 = sbp
    if smoking == "Yes":
        reg_dict["Smoking_Yes"] = 1
    if physical_activity == "High":
        reg_dict["PhysicalActivity_High"] = 1
    reg_df = pd.DataFrame([reg_dict])
    reg_df = sm.add_constant(reg_df, has_constant="add")
    reg_df = reg_df[reg_features]

    rf_dict = {col: 0 for col in rf_features}
    rf_dict["Age"]                 = age
    rf_dict["BMI"]                 = bmi
    rf_dict["SBP"]                 = sbp
    rf_dict["LengthStay"]          = length_stay
    rf_dict["MedicationAdherence"] = medication_adherence
    rf_dict["SleepHours"]          = sleep_hours
    rf_dict["Glucose"]             = glucose
    if chronic_disease == "Yes":
        rf_dict["ChronicDisease_Yes"] = 1
    severity_map = {"Moderate": "Severity_Moderate", "Severe": "Severity_Severe", "Critical": "Severity_Critical"}
    if severity in severity_map:
        rf_dict[severity_map[severity]] = 1
    awareness_map = {"Low": "HealthAwareness_Low", "Moderate": "HealthAwareness_Moderate",
                     "High": "HealthAwareness_High", "Very High": "HealthAwareness_Very High"}
    if health_awareness in awareness_map:
        rf_dict[awareness_map[health_awareness]] = 1
    diet_map = {"Poor": "DietQuality_Poor", "Fair": "DietQuality_Fair",
                "Good": "DietQuality_Good", "Excellent": "DietQuality_Excellent"}
    if diet_quality in diet_map:
        rf_dict[diet_map[diet_quality]] = 1
    rf_df = pd.DataFrame([rf_dict])
    rf_df = rf_df[rf_features]

    health_score     = float(reg_model.predict(reg_df).iloc[0])
    health_score     = round(max(0, min(100, health_score)), 1)
    readmission_prob = float(rf_model.predict_proba(rf_df)[0][1])
    readmission_pct  = round(readmission_prob * 100, 1)

    if readmission_prob < 0.35:
        risk_tier = "Low Risk"
        action    = "Standard discharge"
        color     = "green"
    elif readmission_prob < 0.60:
        risk_tier = "Medium Risk"
        action    = "Schedule 7-day post-discharge follow-up call"
        color     = "orange"
    else:
        risk_tier = "High Risk"
        action    = "Enrol in chronic disease management programme"
        color     = "red"

    st.markdown("## Assessment Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Health Score", value=f"{health_score} / 100")
        if health_score >= 75:
            st.success("Good Health Status")
        elif health_score >= 60:
            st.warning("Moderate Health Status")
        else:
            st.error("Poor Health Status")
    with col2:
        st.metric(label="30-Day Readmission Risk", value=f"{readmission_pct}%")
        if color == "green":
            st.success(risk_tier)
        elif color == "orange":
            st.warning(risk_tier)
        else:
            st.error(risk_tier)

    st.markdown("---")
    st.markdown(f"**Recommended Action:** {action}")
    st.markdown("---")
    st.caption("This tool supports clinical judgement. Built on simulated data for DSCD 603, University of Ghana.")
