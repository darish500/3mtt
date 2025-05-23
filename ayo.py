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

# Define input form
with st.form("prediction_form"):
    Age = st.number_input("Age", min_value=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Smoking = st.selectbox("Smoking", ["Yes", "No"])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    Physical_Activity_Level = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    BMI = st.number_input("BMI")
    Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    Hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    Cholesterol_Level = st.selectbox("Cholesterol Level", ["Normal", "High", "Very High"])
    Resting_BP = st.number_input("Resting Blood Pressure")
    Heart_Rate = st.number_input("Heart Rate")
    Family_History = st.selectbox("Family History", ["Yes", "No"])
    Stress_Level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    Chest_Pain_Type = st.selectbox("Chest Pain Type", ["Type 0", "Type 1", "Type 2", "Type 3"])
    Thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed defect", "Reversible defect"])
    Fasting_Blood_Sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    ECG_Results = st.selectbox("ECG Results", ["Normal", "Abnormal", "Probable or definite"])
    Exercise_Induced_Angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    Max_Heart_Rate_Achieved = st.number_input("Max Heart Rate Achieved")

    submit = st.form_submit_button("Predict")

# Convert categorical values to encoded format if needed
def preprocess_input():
    return [
        Age,
        1 if Gender == "Male" else 0,
        1 if Smoking == "Yes" else 0,
        1 if Alcohol_Consumption == "Yes" else 0,
        {"Low": 0, "Medium": 1, "High": 2}[Physical_Activity_Level],
        BMI,
        1 if Diabetes == "Yes" else 0,
        1 if Hypertension == "Yes" else 0,
        {"Normal": 0, "High": 1, "Very High": 2}[Cholesterol_Level],
        Resting_BP,
        Heart_Rate,
        1 if Family_History == "Yes" else 0,
        {"Low": 0, "Medium": 1, "High": 2}[Stress_Level],
        int(Chest_Pain_Type.split()[-1]),
        {"Normal": 1, "Fixed defect": 2, "Reversible defect": 3}[Thalassemia],
        1 if Fasting_Blood_Sugar == "Yes" else 0,
        {"Normal": 0, "Abnormal": 1, "Probable or definite": 2}[ECG_Results],
        1 if Exercise_Induced_Angina == "Yes" else 0,
        Max_Heart_Rate_Achieved
    ]

# Predict
if submit:
    input_data = pd.DataFrame([preprocess_input()], columns=[
        "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
        "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
        "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
        "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
        "Max_Heart_Rate_Achieved"
    ])
    try:
            #input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Heart Attack Risk: {prediction}")
    except Exception as e:
            st.error(f"Error during prediction: {e}")
