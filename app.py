"""
app.py - Streamlit app for Heart Disease Risk Assessment
Run: streamlit run app.py
This app loads a saved model 'heart_model.joblib' if present; otherwise it will
train a model on heart_disease_sample.csv automatically.
"""
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import subprocess
import sys

MODEL_PATH = Path('heart_model.joblib')
DATA_PATH = Path('heart_disease_sample.csv')

def train_if_needed():
    if not MODEL_PATH.exists():
        st.info("No pre-trained model found. Training model (may take a moment)...")
        subprocess.check_call([sys.executable, 'train_model.py', '--data', str(DATA_PATH), '--model_out', str(MODEL_PATH)])
    return joblib.load(MODEL_PATH)

st.set_page_config(page_title="Heart Disease Risk Assessment", layout="centered")
st.title("❤️ Heart Disease Risk Assessment")
st.write("Enter patient data to predict heart disease risk (binary classification).")

with st.form("input_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options={1: "Male", 0: "Female"}, format_func=lambda x: {1:"Male",0:"Female"}[x])
    cp = st.selectbox("Chest Pain Type (cp)", options=[0,1,2,3], index=1)
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options={0:"No",1:"Yes"}, format_func=lambda x: {0:"No",1:"Yes"}[x])
    restecg = st.selectbox("Resting ECG (restecg)", options=[0,1,2], index=0)
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options={0:"No",1:"Yes"}, format_func=lambda x: {0:"No",1:"Yes"}[x])
    oldpeak = st.number_input("ST depression induced by exercise (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", options=[0,1,2], index=1)
    ca = st.selectbox("Number of major vessels colored (ca)", options=[0,1,2,3,4], index=0)
    thal = st.selectbox("Thal (3 = normal, 6 = fixed defect, 7 = reversible defect)", options=[3,6,7], index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    model = train_if_needed()
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else None
    if pred == 1:
        st.error(f"High risk of heart disease (probability ≈ {prob:.2f})" if prob is not None else "High risk of heart disease")
    else:
        st.success(f"Low risk of heart disease (probability ≈ {prob:.2f})" if prob is not None else "Low risk of heart disease")
    st.write("### Input data")
    st.write(input_df)
