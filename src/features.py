"""Feature engineering utilities."""

from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Group infrequent categorical values into a single label."""

    def __init__(
        self,
        min_fraction: float = 0.01,
        min_count: int = 20,
        other_label: str = "Other",
    ) -> None:
        self.min_fraction = min_fraction
        self.min_count = min_count
        self.other_label = other_label
        self.categories_: Dict[str, set] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RareCategoryGrouper":
        df = self._ensure_dataframe(X)
        self.categories_ = {}
        for col in df.columns:
            counts = df[col].value_counts(dropna=True)
            allowed = counts[
                (counts >= self.min_count)
                | (counts / counts.sum() >= self.min_fraction)
            ].index
            self.categories_[col] = set(allowed.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_dataframe(X)
        for col, allowed in self.categories_.items():
            df[col] = df[col].where(df[col].isin(allowed), self.other_label)
        return df

    @staticmethod
    def _ensure_dataframe(X: pd.DataFrame) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case."""
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower()
        normalized = re.sub(r"[^0-9a-z]+", "_", normalized)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        rename_map[col] = normalized
    return df.rename(columns=rename_map)


def enforce_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required columns exist in the dataframe."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
