"""CLI for local predictions using a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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


def main() -> None:
    """Run a prediction from a JSON file."""
    parser = argparse.ArgumentParser(description="Run local predictions")
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text())
    if isinstance(payload, dict):
        records = [payload]
        single = True
    elif isinstance(payload, list):
        records = payload
        single = False
    else:
        raise ValueError("Input JSON must be an object or a list of objects")

    model, meta = load_artifacts()
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

    print(json.dumps(outputs[0] if single else outputs, indent=2))


if __name__ == "__main__":
    main()
