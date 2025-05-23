import os
import pandas as pd
import joblib
import gdown
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

MODEL_PATH = "heart_attack_risk_model.pkl"
FILE_ID = "1D6tor14R0jFqhvAhiOxPO_q2u26lK-kw"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

model = joblib.load(MODEL_PATH)

expected_columns = [
    "Age", "Gender", "Smoking", "Alcohol_Consumption", "Physical_Activity_Level",
    "BMI", "Diabetes", "Hypertension", "Cholesterol_Level", "Resting_BP",
    "Heart_Rate", "Family_History", "Stress_Level", "Chest_Pain_Type",
    "Thalassemia", "Fasting_Blood_Sugar", "ECG_Results", "Exercise_Induced_Angina",
    "Max_Heart_Rate_Achieved"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = [request.form[key] for key in request.form.keys()]
        df_input = pd.DataFrame([data], columns=expected_columns)
        prediction = model.predict(df_input)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
