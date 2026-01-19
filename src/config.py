"""Configuration constants for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
META_PATH = MODELS_DIR / "best_model_meta.json"

SEED = 42

SOCRATA_DATASET_ID = "k397-673e"
SOCRATA_URL = f"https://data.cityofnewyork.us/resource/{SOCRATA_DATASET_ID}.csv"
DEFAULT_MAX_ROWS = 50000
DEFAULT_PAGE_SIZE = 20000

FEATURE_COLUMNS = [
    "fiscal_year",
    "agency_name",
    "title_description",
    "work_location_borough",
    "leave_status_as_of_june_30",
    "pay_basis",
    "base_salary",
]

TARGET_COLUMN = "target_ot"
RAW_TARGET_COLUMN = "total_ot_paid"