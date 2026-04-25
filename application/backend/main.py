import os
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI(
    title="Agro ML API",
    description="API untuk prediksi kelayakan kondisi agro-environmental.",
    version="1.1.0",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "model" / "pipeline.pkl"))
DATASET_PATH = Path(
    os.getenv("DATASET_PATH", PROJECT_ROOT / "dataset" / "agro_environmental_dataset.csv")
)
N_JOBS = 1
MAX_TRAIN_ROWS = int(os.getenv("MAX_TRAIN_ROWS", "120000"))

NUMERIC_FEATURES = [
    "bulk_density",
    "organic_matter_pct",
    "cation_exchange_capacity",
    "salinity_ec",
    "buffering_capacity",
    "soil_moisture_pct",
    "moisture_limit_dry",
    "moisture_limit_wet",
    "soil_temp_c",
    "air_temp_c",
    "light_intensity_par",
    "soil_ph",
    "ph_stress_flag",
    "nitrogen_ppm",
    "phosphorus_ppm",
    "potassium_ppm",
]
CATEGORICAL_FEATURES = [
    "soil_type",
    "moisture_regime",
    "thermal_regime",
    "nutrient_balance",
    "plant_category",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "failure_flag"

FIELD_LIMITS = {
    "bulk_density": (0.7, 1.6),
    "organic_matter_pct": (1.2, 18.0),
    "cation_exchange_capacity": (5.0, 40.0),
    "salinity_ec": (0.2, 4.0),
    "buffering_capacity": (0.3, 0.9),
    "soil_moisture_pct": (5.0, 70.0),
    "moisture_limit_dry": (8.0, 30.0),
    "moisture_limit_wet": (28.0, 65.0),
    "soil_temp_c": (10.0, 40.0),
    "air_temp_c": (7.0, 49.6),
    "light_intensity_par": (200.0, 1200.0),
    "soil_ph": (4.0, 8.8),
    "ph_stress_flag": (0.0, 1.0),
    "nitrogen_ppm": (20.0, 220.0),
    "phosphorus_ppm": (10.0, 159.0),
    "potassium_ppm": (20.0, 220.0),
}
CATEGORICAL_OPTIONS = {
    "soil_type": [
        "Alluvial",
        "Chalky",
        "Clayey",
        "Laterite",
        "Loamy",
        "Peaty",
        "Saline",
        "Sandy",
        "Silty",
    ],
    "moisture_regime": ["dry", "optimal", "waterlogged"],
    "thermal_regime": ["cold", "heat_stress", "optimal"],
    "nutrient_balance": ["deficient", "excessive", "optimal"],
    "plant_category": ["cereal", "legume", "vegetable"],
}

model = None
model_status = "Model has not been initialized yet."


class InputData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bulk_density: float = Field(ge=0.7, le=1.6)
    organic_matter_pct: float = Field(ge=1.2, le=18.0)
    cation_exchange_capacity: float = Field(ge=5.0, le=40.0)
    salinity_ec: float = Field(ge=0.2, le=4.0)
    buffering_capacity: float = Field(ge=0.3, le=0.9)
    soil_moisture_pct: float = Field(ge=5.0, le=70.0)
    moisture_limit_dry: float = Field(ge=8.0, le=30.0)
    moisture_limit_wet: float = Field(ge=28.0, le=65.0)
    soil_temp_c: float = Field(ge=10.0, le=40.0)
    air_temp_c: float = Field(ge=7.0, le=49.6)
    light_intensity_par: float = Field(ge=200.0, le=1200.0)
    soil_ph: float = Field(ge=4.0, le=8.8)
    ph_stress_flag: int = Field(ge=0, le=1)
    nitrogen_ppm: float = Field(ge=20.0, le=220.0)
    phosphorus_ppm: float = Field(ge=10.0, le=159.0)
    potassium_ppm: float = Field(ge=20.0, le=220.0)
    soil_type: Literal[
        "Alluvial",
        "Chalky",
        "Clayey",
        "Laterite",
        "Loamy",
        "Peaty",
        "Saline",
        "Sandy",
        "Silty",
    ]
    moisture_regime: Literal["dry", "optimal", "waterlogged"]
    thermal_regime: Literal["cold", "heat_stress", "optimal"]
    nutrient_balance: Literal["deficient", "excessive", "optimal"]
    plant_category: Literal["cereal", "legume", "vegetable"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([self.model_dump()], columns=ALL_FEATURES)


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
    numeric_features: list[str]
    categorical_features: list[str]
    field_limits: dict[str, tuple[float, float]]
    categorical_options: dict[str, list[str]]
    model_status: str


class ReloadResponse(BaseModel):
    model_loaded: bool
    model_status: str


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = SklearnPipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(use_smote: bool):
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=N_JOBS,
    )
    preprocessor = build_preprocessor()

    if use_smote:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )

    return SklearnPipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def model_matches_expected_features(candidate_model) -> bool:
    feature_names = getattr(candidate_model, "feature_names_in_", None)
    if feature_names is None:
        return False
    return list(feature_names) == ALL_FEATURES


def train_model():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH, usecols=ALL_FEATURES + [TARGET]).dropna()
    if len(df) > MAX_TRAIN_ROWS:
        sample_fraction = MAX_TRAIN_ROWS / len(df)
        df = (
            df.groupby(TARGET, group_keys=False)
            .apply(
                lambda frame: frame.sample(
                    max(1, int(round(len(frame) * sample_fraction))),
                    random_state=42,
                )
            )
            .reset_index(drop=True)
        )
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Backend fallback training prioritizes startup speed and stability.
    sampling_note = f"trained without SMOTE using up to {MAX_TRAIN_ROWS} rows"
    pipeline = build_pipeline(use_smote=False)

    pipeline.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline, f"{sampling_note}; saved to {MODEL_PATH}"


def load_or_train_model(force_retrain: bool = False) -> None:
    global model, model_status

    try:
        if MODEL_PATH.exists() and not force_retrain:
            loaded_model = joblib.load(MODEL_PATH)
            if model_matches_expected_features(loaded_model):
                model = loaded_model
                model_status = f"Loaded model from {MODEL_PATH}"
                return

            model_status = "Existing model did not match the current feature set. Retraining model."

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
        features=ALL_FEATURES,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        field_limits=FIELD_LIMITS,
        categorical_options=CATEGORICAL_OPTIONS,
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
