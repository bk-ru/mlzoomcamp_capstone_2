"""FastAPI app for overtime prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, PredictionRequest, PredictionResponse


APP_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = APP_ROOT / "models" / "best_model.joblib"
META_PATH = APP_ROOT / "models" / "best_model_meta.json"

app = FastAPI(title="NYC Overtime Prediction Service")
app.state.model = None
app.state.meta = None
app.state.model_error = None


def load_artifacts() -> tuple[Any, Dict[str, Any]]:
    """Load model and metadata from disk."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train the model or add files to models/."
        )
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


@app.on_event("startup")
def startup_event() -> None:
    """Load model artifacts at startup."""
    app.state.model = None
    app.state.meta = None
    app.state.model_error = None
    try:
        model, meta = load_artifacts()
        app.state.model = model
        app.state.meta = meta
    except Exception as exc:  # noqa: BLE001
        app.state.model_error = str(exc)


def get_model() -> tuple[Any, Dict[str, Any]]:
    """Return the loaded model and metadata or raise a 503 error."""
    model = app.state.model
    meta = app.state.meta
    if model is None or meta is None:
        detail = app.state.model_error or "Model is not loaded"
        raise HTTPException(status_code=503, detail=detail)
    return model, meta


def build_dataframe(payload: PredictionRequest, feature_cols: list[str]) -> pd.DataFrame:
    """Build a dataframe aligned to the model feature columns."""
    row = {col: getattr(payload, col, None) for col in feature_cols}
    return pd.DataFrame([row], columns=feature_cols)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict overtime probability for a single record."""
    model, meta = get_model()
    feature_cols = meta.get("feature_cols")
    if not feature_cols:
        raise HTTPException(status_code=500, detail="Model metadata is missing")

    df = build_dataframe(payload, feature_cols)
    prob = float(model.predict_proba(df)[:, 1][0])
    threshold = float(meta.get("threshold", 0.5))
    model_name = str(meta.get("best_model_name", "model"))

    return PredictionResponse(
        prob_ot=prob,
        pred_ot=int(prob >= threshold),
        threshold=threshold,
        model=model_name,
    )
