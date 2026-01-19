# NYC Overtime Prediction (ML Zoomcamp Capstone)

Predict whether a NYC employee will receive overtime pay (`total_ot_paid > 0`) using the Citywide Payroll dataset. This repo includes training scripts, a FastAPI service, tests, Docker setup, and Cloud Run deployment instructions.

## Problem description

This project predicts whether a NYC employee will receive overtime pay in a given fiscal year. The target is binary: `target_ot = (total_ot_paid > 0)`. The task is imbalanced, so model selection emphasizes F1 and PR-AUC, and the decision threshold is tuned rather than fixed at 0.5. Inputs are a mix of categorical attributes (agency, title, borough, leave status, pay basis) and numeric values (fiscal_year, base_salary).

## Project structure

- `src/` training and inference code
- `service/` FastAPI service
- `tests/` API tests
- `models/` saved model artifacts
- `notebooks/` EDA and experimentation
- `screenshots/` EDA and evaluation figures
- `data/` raw and processed data folders

## Data

You have two options:

1) **Local CSV (Kaggle or manual download)**
- Download the Citywide Payroll CSV from Kaggle (or another source with the same schema).
- Place the file(s) into `data/raw/` (multiple CSVs are supported).

2) **Socrata API (NYC Open Data)**
- Dataset: https://data.cityofnewyork.us/City-Government/Citywide-Payroll-Data-Fiscal-Year-/k397-673e
- The training script can download a limited sample from the Socrata API (`k397-673e`).
- Default limit is 50k rows; adjust with `--max-rows` and `--page-size`.

Example:
```bash
python -m src.train --max-rows 80000 --page-size 20000
```

If no CSVs are present in `data/raw/`, the script falls back to Socrata automatically.

## Reproducibility

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

### Training details

`src/train.py`:
- Normalizes column names (lowercase, underscores)
- Builds target `target_ot = total_ot_paid > 0`
- Splits by `fiscal_year` (last year = test, previous year = val)
- Preprocessing: imputer + rare category grouping + one-hot for categoricals; imputer + scaler for numericals
- Models: DummyClassifier, LogisticRegression, RandomForest (or XGBoost if installed)
- Randomized search on top candidates
- Saves artifacts to `models/best_model.joblib` and `models/best_model_meta.json`

## EDA and model diagnostics (screenshots)

### Target Distribution (Pie + Bar)

OT=0 is ~79.3% and OT>0 is ~20.7% (imbalance ~3.83:1). Accuracy alone is misleading, so we focus on F1 and PR-AUC and tune the decision threshold.

<img src="screenshots/Target Distribution (Pie + Bar).png" alt="Target Distribution (Pie + Bar)" width="900">

### Numeric Distributions (log1p): base_salary and total_ot_paid

Both salary and overtime amount are heavy-tailed, so we use log1p for analysis. This is typical for financial data and motivates careful handling of outliers and scaling.

<img src="screenshots/Numeric Distributions.png" alt="Numeric Distributions (log1p)" width="900">

### Base Salary Distribution by Target

The base_salary distribution differs between OT=0 and OT>0, confirming base_salary as an informative signal for overtime prediction.

<img src="screenshots/Base Salary Distribution.png" alt="Base Salary Distribution by Target" width="900">

### Top 15 Agencies by Row Count

Most rows come from a small set of agencies, so `agency_name` is a high-cardinality categorical feature and needs careful encoding with rare-category handling.

<img src="screenshots/Top 15 Agencies.png" alt="Top 15 Agencies by Row Count" width="900">

### Model Comparison on Validation Set

Tuned XGBoost provides the best balance of F1/PR-AUC/ROC-AUC compared with RandomForest and LogisticRegression, while a Dummy baseline shows the task is non-trivial.

<img src="screenshots/Model Comparison on Validation Set.png" alt="Model Comparison on Validation Set" width="900">

### Confusion Matrix (XGBoost tuned, threshold=0.5)

The confusion matrix highlights the FP/FN trade-off at threshold 0.5. We select the final threshold based on the target metric rather than defaulting to 0.5.

<img src="screenshots/Confusion Matrix.png" alt="Confusion Matrix (XGBoost tuned)" width="900">

### ROC Curve + Precision-Recall Curve (Test)

On the test set, the model separates classes well (ROC-AUC ~0.98) and achieves strong PR-AUC (AP ~0.92), which is especially important under class imbalance.

<img src="screenshots/ROC Curve + Precision–Recall.png" alt="ROC and Precision-Recall Curves" width="900">

### Threshold Tuning (Test)

Precision/recall/F1 change substantially with threshold, enabling selection for a business objective such as higher recall (fewer missed overtime cases) or higher precision.

<img src="screenshots/hreshold Tuning.png" alt="Threshold Tuning" width="900">

### Metrics Table (Final comparison)

Final metrics summarize the best candidates; tuned XGBoost provides the strongest overall balance (notably F1 and PR-AUC) and is selected for deployment.

<img src="screenshots/Metrics Table.png" alt="Metrics Table (Final comparison)" width="900">

## Local prediction (CLI)

```bash
python -m src.predict --input sample.json
```

`sample.json` should include the following fields (optional values are allowed):
- `fiscal_year`
- `agency_name`
- `title_description`
- `work_location_borough`
- `leave_status_as_of_june_30`
- `pay_basis`
- `base_salary`

## Service

Start the service:

```bash
uvicorn service.app:app --host 0.0.0.0 --port 8000
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fiscal_year": 2024,
    "agency_name": "Police",
    "title_description": "Officer",
    "work_location_borough": "Queens",
    "leave_status_as_of_june_30": "Active",
    "pay_basis": "per annum",
    "base_salary": 85000
  }'
```

Health check:
```bash
curl http://localhost:8000/health
```

If the model artifacts are missing, `/predict` returns HTTP 503 with a clear error message.

## Tests

```bash
pytest -q
```

## Docker

```bash
docker build -t nyc-ot .
docker run -p 8000:8000 nyc-ot
```

Then call:
```bash
curl http://localhost:8000/health
```

## Cloud deployment (Google Cloud Run)

1) Authenticate and set your project:
```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

2) Build and push the image:
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/nyc-ot
```

3) Deploy to Cloud Run:
```bash
gcloud run deploy nyc-ot \
  --image gcr.io/YOUR_PROJECT_ID/nyc-ot \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000
```

4) Use the service URL from the deploy output to call `/predict`.

## Notes

- To use XGBoost, install it separately: `pip install xgboost`.
- The training script explains how to get data if local CSVs are missing.
