import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load your trained model and label encoder
with open('diabetes_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le_class = pickle.load(f)

# Title of the web app
st.title("🩺 Diabetes Prediction Web App")
st.subheader("ReDI School - Hamburg : Machine Learning Class Project")
st.markdown("Provide your health metrics to predict if you are at risk of diabetes. Please ensure values are accurate for the best results.")

# Input fields with better organization and user-friendliness
st.subheader("Personal Information")
gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
age = st.slider("Age (years)", min_value=20, max_value=79, value=30, help="Enter your age.")

st.subheader("Medical Metrics")
col1, col2 = st.columns(2)

with col1:
    urea = st.number_input("Urea (mg/dL)", min_value=0.5, max_value=38.9, value=30.0, help="Normal range: 7-20 mg/dL.")
    cr = st.number_input("Creatinine (mg/dL)", min_value=6.0, max_value=800.0, value=50.0, help="Normal range: 0.6-1.2 mg/dL.")
    hba1c = st.number_input("HbA1c (%)", min_value=0.9, max_value=15.0, value=7.0, help="Normal range: below 5.7%.")

with col2:
    chol = st.number_input("Cholesterol (mmol/L)", min_value=0.0, max_value=10.3, value=5.0, help="Desirable level: below 5.2 mmol/L.")
    tg = st.number_input("Triglycerides (mmol/L)", min_value=0.3, max_value=13.8, value=8.0, help="Normal range: below 1.7 mmol/L.")
    bmi = st.slider("BMI (kg/m²)", min_value=19.0, max_value=47.75, value=25.0, step=0.1, help="Healthy range: 18.5–24.9 kg/m².")

st.subheader("Lipoprotein Metrics")
hdl = st.number_input("HDL (mmol/L)", min_value=0.2, max_value=9.9, value=5.0, help="Optimal level: above 1.0 mmol/L.")
ldl = st.number_input("LDL (mmol/L)", min_value=0.3, max_value=9.9, value=5.0, help="Optimal level: below 2.6 mmol/L.")
vldl = st.number_input("VLDL (mmol/L)", min_value=0.1, max_value=35.0, value=20.0, help="Normal range: 0.1–1.0 mmol/L.")

# Encoding and prediction
gender_encoded = 1 if gender == "Male" else 0
custom_input = [gender_encoded, age, urea, cr, hba1c, chol, tg, hdl, ldl, vldl, bmi]

if st.button("Predict"):
    custom_input_array = np.array(custom_input).reshape(1, -1)
    prediction = model.predict(custom_input_array)
    predicted_class = le_class.inverse_transform(prediction)[0]

    # Display results
    if predicted_class == 'Y':
        st.warning("⚠️ You have a high chance of diabetes. Please consult a doctor.")
    elif predicted_class == 'N':
        st.success("✅ Good news! You have a low chance of diabetes. Maintain a healthy lifestyle.")
    elif predicted_class == 'P':
        st.info("⚠️ You may be prediabetic. Monitor your sugar intake and consult a doctor.")
    else:
        st.error("Unexpected result. Please re-check your inputs.")
