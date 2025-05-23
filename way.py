import os
import pandas as pd
import joblib
import gdown
import streamlit as st

# Model download info
MODEL_PATH = "heart_attack_risk_model.pkl"
FILE_ID = "1D6tor14R0jFqhvAhiOxPO_q2u26lK-kw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

# Load model
model = joblib.load(MODEL_PATH)

# Input fields
st.title("Heart Attack Risk Prediction")

# Page title
st.title("💓 Heart Attack Risk Prediction")

# Form to collect user input
with st.form("prediction_form"):
    st.header("Enter your health information:")
    
    Age = st.number_input("Age", min_value=1, max_value=120, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Smoking = st.selectbox("Smoking", [1, 0])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", [1, 0])
    Physical_Activity_Level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
    Diabetes = st.selectbox("Diabetes", [1, 0])
    Hypertension = st.selectbox("Hypertension", [1, 0])
    Cholesterol_Level = st.selectbox("Cholesterol Level", ["Low", "Moderate", "High"])
    Resting_BP = st.number_input("Resting BP", min_value=50, max_value=200)
    Heart_Rate = st.number_input("Heart Rate", min_value=30, max_value=200)
    Family_History = st.selectbox("Family History", [1, 0])
    Stress_Level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
    Chest_Pain_Type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    Thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    Fasting_Blood_Sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
    ECG_Results = st.selectbox("ECG Results", ["Normal", "Abnormal", "Hypertrophy"])
    Exercise_Induced_Angina = st.selectbox("Exercise Induced Angina", [1, 0])
    Max_Heart_Rate_Achieved = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220)
    
    submit = st.form_submit_button("Predict Risk")

# Expected column order
expected_columns = [
    "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
    "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
    "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
    "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
    "Max_Heart_Rate_Achieved"
]

if submit:
    # Prepare the data
    input_data = [[
        Age, Gender, Smoking, Alcohol_Consumption, Physical_Activity_Level,
        BMI, Diabetes, Hypertension, Cholesterol_Level, Resting_BP,
        Heart_Rate, Family_History, Stress_Level, Chest_Pain_Type,
        Thalassemia, Fasting_Blood_Sugar, ECG_Results, Exercise_Induced_Angina,
        Max_Heart_Rate_Achieved
    ]]

    df = pd.DataFrame(input_data, columns=expected_columns)

    # Predict
    prediction = model.predict(df)[0]
    
    st.success(f"🩺 Predicted Heart Attack Risk: **{prediction}**")
