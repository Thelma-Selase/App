import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Patient Health Risk Assessment",
    page_icon="🏥",
    layout="centered"
)

# ── Load and prepare data ─────────────────────────────────────────────────────
# We load the raw dataset and rebuild the models every time the app starts
# This avoids needing any external .pkl files

@st.cache_resource
def train_models():

    # Load data
    df = pd.read_csv('DSCD603_Data.csv')

    # Recode binary columns
    for col in ['Smoking', 'Alcohol', 'ChronicDisease', 'Readmission', 'HighRisk']:
        df[col] = df[col].replace({0: 'No', 1: 'Yes'})

    df['Gender']        = df['Gender'].replace({1: 'Male', 2: 'Female'})
    df['MaritalStatus'] = df['MaritalStatus'].replace({1: 'Single', 2: 'Married'})

    df['SES']             = df['SES'].replace({1:'Very Low',2:'Low',3:'Medium',4:'High',5:'Very High'})
    df['PhysicalActivity']= df['PhysicalActivity'].replace({1:'Sedentary',2:'Low',3:'Moderate',4:'High'})
    df['Severity']        = df['Severity'].replace({1:'Mild',2:'Moderate',3:'Severe',4:'Critical'})
    df['ExerciseFreq']    = df['ExerciseFreq'].replace({1:'Never',2:'Rarely',3:'Sometimes',4:'Often',5:'Always'})
    df['DietQuality']     = df['DietQuality'].replace({1:'Very Poor',2:'Poor',3:'Fair',4:'Good',5:'Excellent'})
    df['HealthAwareness'] = df['HealthAwareness'].replace({1:'Very Low',2:'Low',3:'Moderate',4:'High',5:'Very High'})

    # ── Regression Model — HealthScore ────────────────────────────────────────
    reg_data = df[['HealthScore','Age','BMI','SBP','DBP','SleepHours',
                   'MedicationAdherence','LengthStay','Smoking',
                   'ChronicDisease','PhysicalActivity']].copy()

    reg_data = pd.get_dummies(reg_data,
                               columns=['Smoking','ChronicDisease','PhysicalActivity'],
                               drop_first=True)

    bool_cols = reg_data.select_dtypes(include='bool').columns
    reg_data[bool_cols] = reg_data[bool_cols].astype(int)
    reg_data = reg_data.astype(float)

    X_reg = reg_data.drop(columns=['HealthScore'])
    y_reg = reg_data['HealthScore']
    X_reg = sm.add_constant(X_reg)

    reg_model = sm.OLS(y_reg, X_reg).fit()

    # Keep only significant predictors
    sig_vars = reg_model.pvalues[reg_model.pvalues < 0.05].index.tolist()
    X_reg_final = X_reg[sig_vars]
    reg_model_final = sm.OLS(y_reg, X_reg_final).fit()
    reg_features = list(X_reg_final.columns)

    # ── Random Forest Model — Readmission ─────────────────────────────────────
    clf_data = df[['Readmission','Age','BMI','SBP','LengthStay',
                   'MedicationAdherence','ChronicDisease','Severity',
                   'HealthAwareness','DietQuality','SleepHours','Glucose']].copy()

    clf_data['Readmission'] = clf_data['Readmission'].map({'Yes':1,'No':0})

    clf_data = pd.get_dummies(clf_data,
                               columns=['ChronicDisease','Severity',
                                        'HealthAwareness','DietQuality'],
                               drop_first=True)

    bool_cols = clf_data.select_dtypes(include='bool').columns
    clf_data[bool_cols] = clf_data[bool_cols].astype(int)
    clf_data = clf_data.astype(float)

    X_clf = clf_data.drop(columns=['Readmission'])
    y_clf = clf_data['Readmission']
    rf_features = list(X_clf.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    rf_model = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        max_depth=6,
        min_samples_leaf=5,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    return reg_model_final, reg_features, rf_model, rf_features


# Train models on app startup — cached so it only runs once
with st.spinner("Loading models... please wait"):
    reg_model, reg_features, rf_model, rf_features = train_models()

# ── App Header ────────────────────────────────────────────────────────────────

st.title("Patient Health Risk Assessment Tool")
st.markdown("Enter patient details at discharge to get a predicted **Health Score** and **30-day Readmission Risk**.")
st.markdown("---")
st.subheader("Patient Details")

# ── Inputs ────────────────────────────────────────────────────────────────────

col1, col2 = st.columns(2)

with col1:
    age                  = st.number_input("Age (years)", min_value=18, max_value=100, value=52)
    smoking              = st.selectbox("Smoking Status", ["No", "Yes"])
    physical_activity    = st.selectbox("Physical Activity Level", ["Sedentary", "Low", "Moderate", "High"])
    sleep_hours          = st.slider("Sleep Hours per Night", min_value=3.0, max_value=10.0, value=6.5, step=0.5)
    medication_adherence = st.slider("Medication Adherence (1=Poor, 10=Excellent)", min_value=1, max_value=10, value=6)
    length_stay          = st.number_input("Length of Hospital Stay (days)", min_value=1, max_value=30, value=5)

with col2:
    bmi              = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0, step=0.1)
    sbp              = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=220, value=140)
    glucose          = st.number_input("Blood Glucose (mg/dL)", min_value=40, max_value=300, value=95)
    chronic_disease  = st.selectbox("Chronic Disease", ["No", "Yes"])
    severity         = st.selectbox("Disease Severity", ["Mild", "Moderate", "Severe", "Critical"])
    health_awareness = st.selectbox("Health Awareness Level", ["Very Low", "Low", "Moderate", "High", "Very High"])
    diet_quality     = st.selectbox("Diet Quality", ["Very Poor", "Poor", "Fair", "Good", "Excellent"])

st.markdown("---")

# ── Predict Button ────────────────────────────────────────────────────────────

if st.button("Generate Risk Assessment"):

    # ── Build regression input ─────────────────────────────────────────────────
    reg_dict = {col: 0 for col in reg_features if col != 'const'}

    if 'Age' in reg_dict:                    reg_dict['Age'] = age
    if 'MedicationAdherence' in reg_dict:    reg_dict['MedicationAdherence'] = medication_adherence
    if 'SleepHours' in reg_dict:             reg_dict['SleepHours'] = sleep_hours
    if 'LengthStay' in reg_dict:             reg_dict['LengthStay'] = length_stay
    if 'SBP' in reg_dict:                    reg_dict['SBP'] = sbp
    if 'BMI' in reg_dict:                    reg_dict['BMI'] = bmi
    if 'DBP' in reg_dict:                    reg_dict['DBP'] = sbp - 50
    if smoking == 'Yes' and 'Smoking_Yes' in reg_dict:
        reg_dict['Smoking_Yes'] = 1
    if physical_activity == 'High' and 'PhysicalActivity_High' in reg_dict:
        reg_dict['PhysicalActivity_High'] = 1
    if chronic_disease == 'Yes' and 'ChronicDisease_Yes' in reg_dict:
        reg_dict['ChronicDisease_Yes'] = 1

    reg_df = pd.DataFrame([reg_dict])
    reg_df = sm.add_constant(reg_df, has_constant='add')
    reg_df = reg_df[reg_features]

    # ── Build RF input ─────────────────────────────────────────────────────────
    rf_dict = {col: 0 for col in rf_features}

    rf_dict['Age']                 = age
    rf_dict['BMI']                 = bmi
    rf_dict['SBP']                 = sbp
    rf_dict['LengthStay']          = length_stay
    rf_dict['MedicationAdherence'] = medication_adherence
    rf_dict['SleepHours']          = sleep_hours
    rf_dict['Glucose']             = glucose

    if chronic_disease == 'Yes' and 'ChronicDisease_Yes' in rf_dict:
        rf_dict['ChronicDisease_Yes'] = 1

    severity_map = {'Moderate':'Severity_Moderate','Severe':'Severity_Severe','Critical':'Severity_Critical'}
    if severity in severity_map and severity_map[severity] in rf_dict:
        rf_dict[severity_map[severity]] = 1

    awareness_map = {'Low':'HealthAwareness_Low','Moderate':'HealthAwareness_Moderate',
                     'High':'HealthAwareness_High','Very High':'HealthAwareness_Very High'}
    if health_awareness in awareness_map and awareness_map[health_awareness] in rf_dict:
        rf_dict[awareness_map[health_awareness]] = 1

    diet_map = {'Poor':'DietQuality_Poor','Fair':'DietQuality_Fair',
                'Good':'DietQuality_Good','Excellent':'DietQuality_Excellent'}
    if diet_quality in diet_map and diet_map[diet_quality] in rf_dict:
        rf_dict[diet_map[diet_quality]] = 1

    rf_df = pd.DataFrame([rf_dict])[rf_features]

    # ── Run predictions ────────────────────────────────────────────────────────
    health_score     = float(reg_model.predict(reg_df).iloc[0])
    health_score     = round(max(0, min(100, health_score)), 1)
    readmission_prob = float(rf_model.predict_proba(rf_df)[0][1])
    readmission_pct  = round(readmission_prob * 100, 1)

    # ── Risk tier ──────────────────────────────────────────────────────────────
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

    # ── Display results ────────────────────────────────────────────────────────
    st.markdown("## Assessment Results")

    res1, res2 = st.columns(2)

    with res1:
        st.metric("Predicted Health Score", f"{health_score} / 100")
        if health_score >= 75:
            st.success("Good Health Status")
        elif health_score >= 60:
            st.warning("Moderate Health Status")
        else:
            st.error("Poor Health Status")

    with res2:
        st.metric("30-Day Readmission Risk", f"{readmission_pct}%")
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

