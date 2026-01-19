"""Training pipeline implementation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import (
    DEFAULT_MAX_ROWS,
    DEFAULT_PAGE_SIZE,
    FEATURE_COLUMNS,
    META_PATH,
    MODEL_PATH,
    MODELS_DIR,
    RAW_TARGET_COLUMN,
    SEED,
    TARGET_COLUMN,
)
from ..data import load_dataset
from ..features import RareCategoryGrouper, enforce_columns, normalize_columns


@dataclass
class EvalResult:
    name: str
    model: Pipeline
    threshold: float
    metrics: Dict[str, float]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def build_preprocessor(
    categorical_features: List[str],
    numeric_features: List[str],
) -> ColumnTransformer:
    """Build the preprocessing transformer."""
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("rare", RareCategoryGrouper(min_fraction=0.01, min_count=20)),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numeric", numeric_pipeline, numeric_features),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
    """Create candidate model pipelines."""
    models: Dict[str, Pipeline] = {
        "dummy": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "log_reg": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier  # type: ignore

        models["xgboost"] = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=SEED,
                        tree_method="hist",
                    ),
                ),
            ]
        )
    except Exception:  # noqa: BLE001
        models["random_forest"] = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return models


def find_best_threshold(y_true: np.ndarray, probas: np.ndarray) -> Tuple[float, float]:
    """Find a threshold that maximizes F1 on validation data."""
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.1, 0.9, 17):
        preds = (probas >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold, float(best_f1)


def evaluate_predictions(
    y_true: np.ndarray,
    probas: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute evaluation metrics for a given threshold."""
    preds = (probas >= threshold).astype(int)
    return {
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def split_data(
    df: pd.DataFrame,
    target: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, str]:
    """Split data into train/val/test using fiscal year or stratified random."""
    if "fiscal_year" in df.columns and df["fiscal_year"].notna().any():
        year_series = df["fiscal_year"]
        years = sorted(year_series.dropna().unique().tolist())
        if len(years) >= 3:
            test_year = int(max(years))
            val_year = int(sorted(years)[-2])
            train_mask = (year_series < val_year) | year_series.isna()
            val_mask = year_series == val_year
            test_mask = year_series == test_year

            train_df = df[train_mask]
            val_df = df[val_mask]
            test_df = df[test_mask]

            if not train_df.empty and not val_df.empty and not test_df.empty:
                return (
                    train_df,
                    val_df,
                    test_df,
                    target.loc[train_df.index],
                    target.loc[val_df.index],
                    target.loc[test_df.index],
                    "time",
                )

    X_train, X_temp, y_train, y_temp = train_test_split(
        df,
        target,
        test_size=0.4,
        stratify=target,
        random_state=SEED,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, "random"


def tune_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Run randomized search for a candidate model."""
    if name == "log_reg":
        params = {
            "model__C": [0.05, 0.1, 0.3, 1.0, 3.0, 10.0],
        }
    elif name == "random_forest":
        params = {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
    elif name == "xgboost":
        params = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1, 0.2],
            "model__subsample": [0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.8, 0.9, 1.0],
        }
    else:
        return pipeline

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=params,
        n_iter=min(10, sum(len(v) for v in params.values())),
        scoring="f1",
        cv=3,
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: List[str],
    numeric_features: List[str],
) -> EvalResult:
    """Train and select the best model using validation performance."""
    preprocessor = build_preprocessor(categorical_features, numeric_features)
    models = build_models(preprocessor)

    results: List[EvalResult] = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        probas = pipeline.predict_proba(X_val)[:, 1]
        threshold, best_f1 = find_best_threshold(y_val.to_numpy(), probas)
        metrics = evaluate_predictions(y_val.to_numpy(), probas, threshold)
        metrics["f1"] = best_f1
        results.append(
            EvalResult(
                name=name,
                model=pipeline,
                threshold=threshold,
                metrics=metrics,
            )
        )

    results_sorted = sorted(results, key=lambda r: r.metrics["f1"], reverse=True)
    top_candidates = results_sorted[:2]

    tuned_results: List[EvalResult] = []
    for candidate in top_candidates:
        if candidate.name == "dummy":
            tuned_results.append(candidate)
            continue
        tuned_model = tune_model(
            candidate.name,
            candidate.model,
            X_train,
            y_train,
        )
        probas = tuned_model.predict_proba(X_val)[:, 1]
        threshold, best_f1 = find_best_threshold(y_val.to_numpy(), probas)
        metrics = evaluate_predictions(y_val.to_numpy(), probas, threshold)
        metrics["f1"] = best_f1
        tuned_results.append(
            EvalResult(
                name=f"{candidate.name}_tuned",
                model=tuned_model,
                threshold=threshold,
                metrics=metrics,
            )
        )

    combined = results + tuned_results
    best = sorted(combined, key=lambda r: r.metrics["f1"], reverse=True)[0]
    return best


def run_training(
    max_rows: int = DEFAULT_MAX_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Dict[str, Any]:
    """Run model training and save the best model artifacts."""
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, source = load_dataset(max_rows=max_rows, page_size=page_size)
    df = normalize_columns(df)

    enforce_columns(df, [RAW_TARGET_COLUMN])
    enforce_columns(df, FEATURE_COLUMNS)

    df[TARGET_COLUMN] = df[RAW_TARGET_COLUMN].fillna(0) > 0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df["fiscal_year"] = pd.to_numeric(df["fiscal_year"], errors="coerce")
    df["base_salary"] = pd.to_numeric(df["base_salary"], errors="coerce")

    feature_df = df[FEATURE_COLUMNS].copy()
    target = df[TARGET_COLUMN].copy()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        split_type,
    ) = split_data(feature_df, target)

    categorical_features = [
        col for col in FEATURE_COLUMNS if col not in {"fiscal_year", "base_salary"}
    ]
    numeric_features = [col for col in FEATURE_COLUMNS if col in {"fiscal_year", "base_salary"}]

    best_result = train_models(
        X_train,
        y_train,
        X_val,
        y_val,
        categorical_features,
        numeric_features,
    )

    final_model = best_result.model
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_probas = final_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_predictions(
        y_test.to_numpy(),
        test_probas,
        best_result.threshold,
    )

    joblib.dump(final_model, MODEL_PATH)

    meta = {
        "seed": SEED,
        "feature_cols": FEATURE_COLUMNS,
        "best_model_name": best_result.name,
        "threshold": best_result.threshold,
        "metrics": {
            "val": best_result.metrics,
            "test": test_metrics,
        },
        "data_source": source,
        "split_type": split_type,
        "trained_at": datetime.utcnow().isoformat() + "Z",
    }

    META_PATH.write_text(json.dumps(meta, indent=2))

    print("Training complete")
    print(f"Best model: {best_result.name}")
    print(f"Threshold: {best_result.threshold:.2f}")
    print(f"Validation metrics: {best_result.metrics}")
    print(f"Test metrics: {test_metrics}")

    return meta
