import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler
model = tf.keras.models.load_model("heart_disease_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction using CNN–BiLSTM")

# Feature input form
with st.form("input_form"):
    st.subheader("Enter Patient Details")
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol", min_value=100, max_value=600, value=240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0–4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

    submitted = st.form_submit_button("Predict")

# Predict if form is submitted
if submitted:
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]], dtype=np.float32)

    # Scale the data
    scaled_features = scaler.transform(input_data)
    reshaped_input = scaled_features.reshape((scaled_features.shape[0], scaled_features.shape[1], 1))

    # Predict
    prediction = model.predict(reshaped_input, verbose=0)[0][0]
    result = "✅ No Heart Disease Detected" if prediction < 0.5 else "⚠️ Risk of Heart Disease Detected"

    st.subheader("Prediction Result:")
    st.success(result)

