"""Data loading utilities for training."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests

from .config import DATA_RAW_DIR, DEFAULT_MAX_ROWS, DEFAULT_PAGE_SIZE, SOCRATA_URL


def load_local_data(data_dir: Path = DATA_RAW_DIR) -> Optional[pd.DataFrame]:
    """Load and concatenate CSV files from the local data directory."""
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        return None

    frames = []
    for csv_path in csv_files:
        frames.append(pd.read_csv(csv_path, low_memory=False))

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def download_socrata(
    max_rows: int = DEFAULT_MAX_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
    url: str = SOCRATA_URL,
) -> pd.DataFrame:
    """Download a limited dataset from the Socrata API with paging."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    frames = []
    offset = 0
    while offset < max_rows:
        limit = min(page_size, max_rows - offset)
        params = {"$limit": limit, "$offset": offset}
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()

        if not response.text.strip():
            break

        chunk = pd.read_csv(io.StringIO(response.text))
        if chunk.empty:
            break

        frames.append(chunk)
        offset += len(chunk)
        if len(chunk) < limit:
            break

    if not frames:
        raise RuntimeError("Socrata download returned no rows")

    return pd.concat(frames, ignore_index=True)


def load_dataset(
    max_rows: int = DEFAULT_MAX_ROWS,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Tuple[pd.DataFrame, str]:
    """Load data from local CSVs or fall back to the Socrata API."""
    local_df = load_local_data()
    if local_df is not None:
        return local_df, "local"

    try:
        remote_df = download_socrata(max_rows=max_rows, page_size=page_size)
        return remote_df, "socrata"
    except Exception as exc:  # noqa: BLE001
        message = (
            "No CSV files found in data/raw and Socrata download failed. "
            "Place the dataset into data/raw or check network access."
        )
        raise RuntimeError(message) from exc