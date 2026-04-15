# Pearls AQI Predictor

End-to-end Air Quality Index (AQI) prediction service focused on Karachi, Pakistan.
Current phase: **Week 1 - Data Ingestion and EDA**.

## Project Status

### Completed

- AQICN current ingestion pipeline for Karachi (`src/data/ingestion.py`)
- Open-Meteo historical hourly ingestion from `2021-01-01` to today (`src/data/historical_ingestion.py`)
- EDA workflow for historical data with saved figures (`src/data/eda_karachi.py`)
- Dependency management with `requirements.txt`
- Secret handling via `.env` and `.gitignore`

### Data Outputs

- Current/raw AQI snapshot: `data/raw/karachi_aqi_raw.csv`
- Historical hourly AQI data: `data/raw/karachi_historical_aqi.csv`
- EDA figures:
  - `reports/figures/karachi_pm25_pm10_trend.png`
  - `reports/figures/karachi_pollutant_correlation_heatmap.png`

## Repository Structure

```text
.
├── notebooks/
│   └── 01_eda_karachi.py
├── src/
│   └── data/
│       ├── ingestion.py
│       ├── historical_ingestion.py
│       └── eda_karachi.py
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

## Next Suggested Milestones

- Feature engineering (lags, rolling windows, seasonal signals)
- Train/validation split and baseline model
- Model evaluation and experiment tracking
- Serverless deployment architecture for inference API
