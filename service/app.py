"""FastAPI app for overtime prediction."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from src.predict.predictor import load_artifacts, predict_with_model
from src.predict.schemas import HealthResponse, PredictionRequest, PredictionResponse

app = FastAPI(title="NYC Overtime Prediction Service")
app.state.model = None
app.state.meta = None
app.state.model_error = None

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


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict overtime probability for a single record."""
    model, meta = get_model()
    try:
        outputs = predict_with_model([payload.model_dump()], model, meta)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output = outputs[0]
    return PredictionResponse(**output)
