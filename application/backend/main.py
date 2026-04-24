import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title="Agro ML API",
    description="API untuk prediksi kelayakan kondisi agro-environmental.",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "model" / "pipeline.pkl"))
DATASET_PATH = Path(
    os.getenv("DATASET_PATH", PROJECT_ROOT / "dataset" / "agro_environmental_dataset.csv")
)

FEATURES = [
    "bulk_density",
    "organic_matter_pct",
    "cation_exchange_capacity",
    "salinity_ec",
]
TARGET = "failure_flag"

FIELD_LIMITS = {
    "bulk_density": (0.0, 3.0),
    "organic_matter_pct": (0.0, 100.0),
    "cation_exchange_capacity": (0.0, 100.0),
    "salinity_ec": (0.0, 50.0),
}

model = None
model_status = "Model has not been initialized yet."


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bulk_density: float = Field(ge=0.0, le=3.0, description="Bulk density tanah")
    organic_matter_pct: float = Field(
        ge=0.0,
        le=100.0,
        description="Persentase bahan organik",
    )
    cation_exchange_capacity: float = Field(
        ge=0.0,
        le=100.0,
        description="Cation exchange capacity",
    )
    salinity_ec: float = Field(ge=0.0, le=50.0, description="Salinity electrical conductivity")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()], columns=FEATURES)


class PredictionResponse(BaseModel):
    prediction: int
    interpretation: str
    confidence: float | None = None
    model_loaded: bool = True


class HealthResponse(BaseModel):
    model_loaded: bool
    model_path: str
    dataset_path: str
    model_exists: bool
    dataset_exists: bool
    features: list[str]
    field_limits: dict[str, tuple[float, float]]
    model_status: str


class ReloadResponse(BaseModel):
    model_loaded: bool
    model_status: str


def build_pipeline(use_smote: bool):
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=-1,
    )

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [
                ("scaler", StandardScaler()),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )

    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def train_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, usecols=FEATURES + [TARGET]).dropna()
    X = df[FEATURES]
    y = df[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    sampling_note = "trained without SMOTE"
    try:
        pipeline = build_pipeline(use_smote=True)
        sampling_note = "trained with SMOTE"
    except ImportError:
        pipeline = build_pipeline(use_smote=False)

    pipeline.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, f"{sampling_note}; saved to {MODEL_PATH}"


def load_or_train_model(force_retrain: bool = False) -> None:
    global model, model_status

    try:
        if MODEL_PATH.exists() and not force_retrain:
            model = joblib.load(MODEL_PATH)
            model_status = f"Loaded model from {MODEL_PATH}"
            return

        model, training_status = train_model()
        model_status = f"Model was initialized by training a new pipeline: {training_status}"
    except Exception as exc:
        model = None
        model_status = f"Failed to initialize model: {exc}"


def build_health_payload() -> HealthResponse:
    return HealthResponse(
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        dataset_path=str(DATASET_PATH),
        model_exists=MODEL_PATH.exists(),
        dataset_exists=DATASET_PATH.exists(),
        features=FEATURES,
        field_limits=FIELD_LIMITS,
        model_status=model_status,
    )


def build_prediction_response(data: InputData) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=model_status)

    try:
        input_frame = data.to_frame()
        prediction = int(model.predict(input_frame)[0])

        confidence = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_frame)[0]
            confidence = round(float(max(probabilities)), 4)

        return PredictionResponse(
            prediction=prediction,
            interpretation="Suitable" if prediction == 0 else "Not Suitable",
            confidence=confidence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


load_or_train_model()


@app.get("/")
def read_root() -> dict[str, Any]:
    return {
        "message": "Agro ML API is running",
        "model_loaded": model is not None,
        "model_status": model_status,
    }


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return build_health_payload()


@app.post("/reload-model", response_model=ReloadResponse)
def reload_model(force_retrain: bool = False) -> ReloadResponse:
    load_or_train_model(force_retrain=force_retrain)

    if model is None:
        raise HTTPException(status_code=500, detail=model_status)

    return ReloadResponse(model_loaded=True, model_status=model_status)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData) -> PredictionResponse:
    return build_prediction_response(data)
