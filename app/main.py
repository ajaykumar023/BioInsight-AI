from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="BioInsight AI",
    description="Heart Disease Prediction API",
    version="1.0",
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "best_model.joblib"
)

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# INPUT SCHEMA
# --------------------------------------------------

class PatientData(BaseModel):
    age: int
    sex: str
    chest_pain_type: str
    resting_blood_pressure: float
    cholesterol: float
    fasting_blood_sugar: bool
    resting_ecg: str
    max_heart_rate: float
    exercise_induced_angina: bool
    st_depression: float
    st_slope: str
    major_vessels: float
    thalassemia: str

# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "BioInsight AI API is running",
        "model_loaded": True,
    }

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.post("/predict")
def predict(data: PatientData):

    df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(df)[0]

    probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": int(prediction),
        "probability": round(probability, 4),
        "risk": (
            "Heart Disease"
            if prediction == 1
            else "No Heart Disease"
        ),
    }