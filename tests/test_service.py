"""Service endpoint tests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLS = [
    "fiscal_year",
    "agency_name",
    "title_description",
    "work_location_borough",
    "leave_status_as_of_june_30",
    "pay_basis",
    "base_salary",
]

CAT_COLS = [
    "agency_name",
    "title_description",
    "work_location_borough",
    "leave_status_as_of_june_30",
    "pay_basis",
]

NUM_COLS = ["fiscal_year", "base_salary"]


def ensure_model() -> None:
    """Create a small model artifact for tests if missing."""
    model_path = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
    meta_path = Path(__file__).resolve().parents[1] / "models" / "best_model_meta.json"

    if model_path.exists() and meta_path.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    data = pd.DataFrame(
        [
            [2021, "Police", "Officer", "Queens", "Active", "per annum", 80000],
            [2022, "Fire", "Firefighter", "Bronx", "Active", "per annum", 90000],
            [2023, "Police", "Detective", "Brooklyn", "On Leave", "per annum", 95000],
            [2024, "Parks", "Supervisor", "Manhattan", "Active", "per annum", 70000],
        ],
        columns=FEATURE_COLS,
    )
    target = [0, 1, 1, 0]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CAT_COLS),
            ("numeric", numeric_pipeline, NUM_COLS),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )
    pipeline.fit(data, target)

    joblib.dump(pipeline, model_path)

    probas = pipeline.predict_proba(data)[:, 1]
    threshold = 0.5
    preds = (probas >= threshold).astype(int)
    metrics = {
        "f1": float(f1_score(target, preds, zero_division=0)),
        "precision": float(precision_score(target, preds, zero_division=0)),
        "recall": float(recall_score(target, preds, zero_division=0)),
        "accuracy": float(accuracy_score(target, preds)),
    }

    meta = {
        "seed": 42,
        "feature_cols": FEATURE_COLS,
        "best_model_name": "log_reg_test",
        "threshold": threshold,
        "metrics": {"val": metrics, "test": metrics},
        "data_source": "synthetic",
        "split_type": "synthetic",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


ensure_model()

from service.app import app  # noqa: E402


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


def test_predict() -> None:
    payload = {
        "fiscal_year": 2024,
        "agency_name": "Police",
        "title_description": "Officer",
        "work_location_borough": "Queens",
        "leave_status_as_of_june_30": "Active",
        "pay_basis": "per annum",
        "base_salary": 85000,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"prob_ot", "pred_ot", "threshold", "model"}
        assert isinstance(data["prob_ot"], float)
        assert 0.0 <= data["prob_ot"] <= 1.0
        assert isinstance(data["pred_ot"], int)
        assert isinstance(data["threshold"], float)
        assert isinstance(data["model"], str)
