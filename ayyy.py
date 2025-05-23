import os
import pandas as pd
import joblib
import gdown
import streamlit as st

# Path to the model
MODEL_PATH = "heart_attack_risk_model.pkl"
FILE_ID = "1D6tor14R0jFqhvAhiOxPO_q2u26lK-kw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# Load the model
model = joblib.load(MODEL_PATH)

# Expected input features
expected_columns = [
    "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
    "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
    "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
    "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
    "Max_Heart_Rate_Achieved"
]

# Title
st.title("Heart Attack Risk Prediction")

# Form
with st.form("risk_form"):
    user_input = {}
    for col in expected_columns:
        user_input[col] = st.text_input(f"Enter {col}")
    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Heart Attack Risk: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
