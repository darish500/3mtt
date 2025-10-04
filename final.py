import os
import streamlit as st

MODEL_PATH = "heart_attack_risk_model.pkl"
FILE_ID = "1D6tor14R0jFqhvAhiOxPO_q2u26lK-kw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Ensure the model file exists; download lazily only when needed
def _ensure_model_file_present() -> None:
    if os.path.exists(MODEL_PATH):
        return
    with st.spinner("Downloading model from Google Drive..."):
        # Import gdown only when a download is actually required
        import gdown as _gdown
        _gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=True)

# Lazily load and cache the model across reruns
@st.cache_resource(show_spinner=False)
def get_model():
    _ensure_model_file_present()
    # Import joblib only when loading the model
    import joblib as _joblib
    # Use memory mapping to reduce load time and RAM when possible
    return _joblib.load(MODEL_PATH, mmap_mode="r")

st.title("Heart Attack Risk Predictor")

# Form inputs
with st.form("risk_form"):
    st.subheader("Enter Patient Information:")

    Age = st.number_input("Age", min_value=1, max_value=120)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Smoking = st.selectbox("Smoking", ["Yes", "No"])
    Alcohol_Consumption = st.selectbox("Alcohol Consumption", ["Yes", "No"])
    Physical_Activity_Level = st.selectbox("Physical Activity Level", ["Low", "Medium", "High"])
    BMI = st.number_input("BMI", min_value=10.0, max_value=60.0)
    Diabetes = st.selectbox("Diabetes", ["Yes", "No"])
    Hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    Cholesterol_Level = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"])
    Resting_BP = st.number_input("Resting Blood Pressure")
    Heart_Rate = st.number_input("Heart Rate")
    Family_History = st.selectbox("Family History of Heart Disease", ["Yes", "No"])
    Stress_Level = st.selectbox("Stress Level", ["Low", "Medium", "High"])
    Chest_Pain_Type = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    Thalassemia = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
    Fasting_Blood_Sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
    ECG_Results = st.selectbox("ECG Results", ["Normal", "Abnormal", "Hypertrophy"])
    Exercise_Induced_Angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    Max_Heart_Rate_Achieved = st.number_input("Max Heart Rate Achieved")

    submit = st.form_submit_button("Predict")

if submit:
    # Defer heavy imports until prediction is requested
    import pandas as pd
    input_data = pd.DataFrame([[
        Age, Gender, Smoking, Alcohol_Consumption, Physical_Activity_Level,
        BMI, Diabetes, Hypertension, Cholesterol_Level, Resting_BP,
        Heart_Rate, Family_History, Stress_Level, Chest_Pain_Type,
        Thalassemia, Fasting_Blood_Sugar, ECG_Results, Exercise_Induced_Angina,
        Max_Heart_Rate_Achieved
    ]], columns=[
        "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
        "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
        "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
        "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
        "Max_Heart_Rate_Achieved"
    ])

    model = get_model()
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Risk: {prediction}")
