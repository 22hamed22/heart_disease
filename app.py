import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="centered"
)

# =========================
# ✅ CUSTOM CSS
# =========================
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.big-title {
    font-size: 36px;
    font-weight: bold;
    color: #e63946;
    text-align: center;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
    color: #155724;
    font-weight: bold;
}
.danger-box {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 10px;
    color: #721c24;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# ✅ PATHS
# =========================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "heart.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# =========================
# ✅ LOAD OR TRAIN MODEL
# =========================
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        return train_model()

def train_model():
    st.warning("Training model for the first time...")

    # ✅ If dataset doesn't exist, create it automatically
    if not DATA_PATH.exists():
        url = "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv"
        df = pd.read_csv(url)
        df.to_csv(DATA_PATH, index=False)
    else:
        df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    return model

model = load_model()

# =========================
# ✅ TITLE
# =========================
st.markdown("<div class='big-title'>❤️ Heart Disease Risk Assessment</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI-Based Heart Attack Prediction System</div>", unsafe_allow_html=True)
st.markdown("---")

# =========================
# ✅ INPUT FORM
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 250, 130)
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

    with col2:
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thal", [3, 6, 7])

    submitted = st.form_submit_button("🔍 Predict Heart Disease Risk")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ✅ PREDICTION OUTPUT
# =========================
if submitted:
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
        "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("## 🩺 Prediction Result")

    if prediction == 1:
        st.markdown(
            f"<div class='danger-box'>⚠️ High Risk of Heart Disease<br>Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='success-box'>✅ Low Risk of Heart Disease<br>Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )

    with st.expander("📊 View Entered Patient Data"):
        st.write(input_df)

# =========================
# ✅ FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Machine Learning & Streamlit | Internship Project</center>",
    unsafe_allow_html=True
)
