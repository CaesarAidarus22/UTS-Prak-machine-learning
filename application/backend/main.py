from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

# =========================
# Config Path
# =========================
MODEL_PATH = "../../model/pipeline.pkl"

model = None
use_pipeline = False

# =========================
# Load Model
# =========================
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)

        if hasattr(model, "predict"):
            use_pipeline = True

except Exception as e:
    print("Error loading model:", e)


# =========================
# Request Schema (SUDAH FIX)
# =========================
class InputData(BaseModel):
    bulk_density: float
    organic_matter_pct: float   # ✅ FIX
    cation_exchange_capacity: float
    salinity_ec: float          # ✅ FIX


# =========================
# Root Endpoint
# =========================
@app.get("/")
def read_root():
    return {"message": "Agro ML API is running"}


# =========================
# Prediction Endpoint
# =========================
@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {"error": "Model not loaded"}

    try:
        # URUTAN HARUS SAMA DENGAN TRAINING
        input_array = np.array([[
            data.bulk_density,
            data.organic_matter_pct,
            data.cation_exchange_capacity,
            data.salinity_ec
        ]])

        prediction = model.predict(input_array)[0]

        return {
            "prediction": int(prediction),
            "interpretation": "Suitable" if prediction == 0 else "Not Suitable"
        }

    except Exception as e:
        return {"error": str(e)}