import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("heart_attack_risk_model.pkl")

# Expected columns
expected_columns = [
    "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
    "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
    "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
    "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
    "Max_Heart_Rate_Achieved"
]

# Streamlit UI
st.set_page_config(page_title="Heart Attack Risk Prediction")
st.title("Heart Attack Risk Prediction")

# Create input form
with st.form("risk_form"):
    Age = st.number_input("Age", min_value=0, max_value=120)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Smoking = st.selectbox("Smoking", [1, 0])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", [1, 0])
    Physical_Activity_Level = st.selectbox("Physical Activity Level", ["Low", "Moderate", "High"])
    BMI = st.number_input("BMI", min_value=0.0, format="%.1f")
    Diabetes = st.selectbox("Diabetes", [1, 0])
    Hypertension = st.selectbox("Hypertension", [1, 0])
    Cholesterol_Level = st.selectbox("Cholesterol Level", ["Low", "Moderate", "High"])
    Resting_BP = st.number_input("Resting BP", min_value=0)
    Heart_Rate = st.number_input("Heart Rate", min_value=0)
    Family_History = st.selectbox("Family History", [1, 0])
    Stress_Level = st.selectbox("Stress Level", ["Low", "Moderate", "High"])
    Chest_Pain_Type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal", "Asymptomatic"])
    Thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    Fasting_Blood_Sugar = st.selectbox("Fasting Blood Sugar", [1, 0])
    ECG_Results = st.selectbox("ECG Results", ["Normal", "Abnormal", "Hypertrophy"])
    Exercise_Induced_Angina = st.selectbox("Exercise Induced Angina", [1, 0])
    Max_Heart_Rate_Achieved = st.number_input("Max Heart Rate Achieved", min_value=0)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = [
            Age, Gender, Smoking, Alcohol_Consumption, Physical_Activity_Level,
            BMI, Diabetes, Hypertension, Cholesterol_Level, Resting_BP,
            Heart_Rate, Family_History, Stress_Level, Chest_Pain_Type,
            Thalassemia, Fasting_Blood_Sugar, ECG_Results, Exercise_Induced_Angina,
            Max_Heart_Rate_Achieved
        ]

        df = pd.DataFrame([input_data], columns=expected_columns)
        prediction = model.predict(df)[0]
        st.success(f"Predicted Risk Level: {prediction}")
