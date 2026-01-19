"""Prediction utilities for local inference and service usage."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import joblib
import pandas as pd

from ..config import META_PATH, MODEL_PATH


def load_artifacts() -> tuple[Any, Dict[str, Any]]:
    """Load model pipeline and metadata."""
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Train the model or place files into models/."
        )
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta


def prepare_dataframe(payload: List[Dict[str, Any]], feature_cols: List[str]) -> pd.DataFrame:
    """Prepare a dataframe with the expected feature columns."""
    normalized: List[Dict[str, Any]] = []
    for row in payload:
        cleaned = {col: row.get(col) for col in feature_cols}
        normalized.append(cleaned)
    return pd.DataFrame(normalized, columns=feature_cols)


def predict_with_model(
    records: List[Dict[str, Any]],
    model: Any,
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Generate predictions for records using a pre-loaded model."""
    feature_cols = meta.get("feature_cols", [])
    if not feature_cols:
        raise ValueError("Feature columns not found in metadata")

    df = prepare_dataframe(records, feature_cols)
    probas = model.predict_proba(df)[:, 1]
    threshold = float(meta.get("threshold", 0.5))
    model_name = str(meta.get("best_model_name", "model"))

    outputs = []
    for prob in probas:
        outputs.append(
            {
                "prob_ot": float(prob),
                "pred_ot": int(prob >= threshold),
                "threshold": threshold,
                "model": model_name,
            }
        )
    return outputs
