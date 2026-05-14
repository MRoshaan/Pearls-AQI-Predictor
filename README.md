# Pearls AQI Predictor

End-to-end PM2.5 forecasting service with AQI-style risk bands focused on Karachi, Pakistan.
Current phase: **Week 5 - Serving API and Dashboard Interface**.

## Project Status

### Completed

- AQICN current ingestion pipeline for Karachi (`src/data/ingestion.py`)
- Open-Meteo historical hourly ingestion from `2021-01-01` to today (`src/data/historical_ingestion.py`)
- EDA workflow for historical data with saved figures (`src/data/eda_karachi.py`)
- Feature engineering pipeline with targets for +24h/+48h/+72h (`src/features/build_features.py`)
- Hopsworks feature store uploader (`src/features/push_to_hopsworks.py`)
- Model training and evaluation pipeline (`src/models/train_model.py`)
- Dependency management with `requirements.txt`
- Secret handling via `.env` and `.gitignore`

### Data Outputs

- Current/raw AQI snapshot: `data/raw/karachi_aqi_raw.csv`
- Historical hourly AQI data: `data/raw/karachi_historical_aqi.csv`
- Engineered feature dataset: `data/processed/karachi_features.csv`
- EDA figures:
  - `reports/figures/karachi_pm25_pm10_trend.png`
  - `reports/figures/karachi_pollutant_correlation_heatmap.png`

## Repository Structure

```text
.
├── notebooks/
│   └── 01_eda_karachi.py
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── historical_ingestion.py
│   │   └── eda_karachi.py
│   └── features/
│       ├── build_features.py
│       └── push_to_hopsworks.py
│   └── models/
│       └── train_model.py
├── requirements.txt
└── README.md
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Add environment variable for AQICN ingestion (`.env` in repo root):

```env
API_KEY=your_aqicn_token
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT=your_hopsworks_project_name
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
HOPSWORKS_PORT=443
HOPSWORKS_MODEL_NAME=karachi_aqi_forecaster
```

## Run Pipelines

### 1) AQICN Current Data Ingestion

```bash
python src/data/ingestion.py
```

### 2) Historical Data Ingestion (Open-Meteo)

```bash
python src/data/historical_ingestion.py
```

### 3) EDA on Historical Data

```bash
python src/data/eda_karachi.py
```

### 4) Build Feature Dataset (Week 2)

```bash
python src/features/build_features.py
```

This step now merges:
- historical Open-Meteo hourly data
- latest AQICN live snapshot (`data/raw/karachi_aqi_raw.csv`)

so the most recent timestamp can stay closer to real-time.

### 5) Push Features to Hopsworks

```bash
python src/features/push_to_hopsworks.py
```

### 6) Train Models and Upload Winner to Model Registry (Week 3)

```bash
python src/models/train_model.py
```

### 7) Run FastAPI Serving Layer (Week 5)

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:
- `GET /health`
- `GET /predict/latest`
- `GET /predict/latest/explain?max_features=10`

`/predict/latest` will:
- load the latest model from Hopsworks Model Registry (fallback to local artifacts)
- read latest features from Hopsworks Feature Store (fallback to local `data/processed/karachi_features.csv`)
- predict PM2.5 concentration (ug/m^3) for +24h, +48h, +72h
- return a hazardous alert flag when forecast crosses PM2.5 hazardous threshold

### 8) Run Streamlit Dashboard (Week 5)

```bash
streamlit run src/app/dashboard.py
```

If your API is running on a different host/port:

```bash
API_BASE_URL=http://127.0.0.1:8000 streamlit run src/app/dashboard.py
```

Dashboard capabilities:
- shows +24h/+48h/+72h PM2.5 forecasts (ug/m^3) from FastAPI backend
- converts PM2.5 to AQI index using US EPA PM2.5 breakpoints for display
- displays hazardous AQI visual alert
- shows a data freshness warning when latest feature timestamp is older than 24 hours
- renders LIME top feature contributions for latest +24h prediction

Serving reliability notes:
- FastAPI caches fetched feature data in-memory for a short TTL to reduce repeated Feature Store reads.
- Configure cache window with `FEATURE_CACHE_TTL_SECONDS` (default: `120`).

This script:
- pulls historical features/targets from Hopsworks
- trains baseline models (Ridge, Random Forest)
- optionally trains a TensorFlow MLP (`ENABLE_TENSORFLOW=true`)
- evaluates using RMSE, MAE, R2
- saves local artifacts under `artifacts/models/`
- uploads the best model to Hopsworks Model Registry

## Next Suggested Milestones

- End-to-end integration tests for API and dashboard paths
- CI reliability hardening and failure alerting
- PEP 8 cleanup and modular refactoring across scripts
- Final comprehensive architecture and experiment report
