# Pearls AQI Predictor

End-to-end Air Quality Index (AQI) prediction service focused on Karachi, Pakistan.
Current phase: **Week 2 - Feature Engineering and Feature Store Integration**.

## Project Status

### Completed

- AQICN current ingestion pipeline for Karachi (`src/data/ingestion.py`)
- Open-Meteo historical hourly ingestion from `2021-01-01` to today (`src/data/historical_ingestion.py`)
- EDA workflow for historical data with saved figures (`src/data/eda_karachi.py`)
- Feature engineering pipeline with targets for +24h/+48h/+72h (`src/features/build_features.py`)
- Hopsworks feature store uploader (`src/features/push_to_hopsworks.py`)
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

### 5) Push Features to Hopsworks

```bash
python src/features/push_to_hopsworks.py
```

## Next Suggested Milestones

- Train/validation split and baseline model training pipeline
- Model registry integration and experiment tracking
- Hourly/daily CI pipeline automation
- Prediction API + dashboard + AQI alerting
