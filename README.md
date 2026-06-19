# Pearls AQI Predictor

**A serverless 3-day Air Quality Index forecasting platform for Karachi, Pakistan.**

> **Live Demo:** [https://roshaan727-pearls.hf.space](https://roshaan727-pearls.hf.space)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-black)

---

## Executive Summary

This project is an end-to-end MLOps pipeline that predicts daily-average PM2.5 concentrations for Karachi across three forecast horizons (Day 1, Day 2, Day 3), converts them to US EPA AQI values, and serves them through a FastAPI backend and a Streamlit dashboard — all running on a 100% serverless stack.

The system fetches real-time and historical air quality + weather data, engineers 120+ features (lags, rolling statistics, weather forecasts, seasonal signals), trains horizon-specific ensemble models (VotingRegressor combining HistGradientBoostingRegressor + RandomForestRegressor), and exposes predictions with built-in explainability.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Sources   │────▶│  Feature Pipeline │────▶│  Feature Store  │
│  (AQICN + Open-  │     │  (GitHub Actions) │     │ (Hopsworks/Local│
│   Meteo APIs)    │     └──────────────────┘     └────────┬────────┘
└─────────────────┘                                        │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit App   │◀────│   FastAPI Server  │◀────│ Training Pipeline│
│  (Dashboard)     │     │  (REST API)       │     │  (GitHub Actions)│
└────────┬────────┘     └──────────────────┘     └────────┬────────┘
         │                                                │
         ▼                                                ▼
  ┌──────────────┐                                ┌──────────────┐
  │  Live Demo   │                                │ Model Registry│
  │  (HuggingFace│                                │ (Hopsworks/   │
  │   Spaces)    │                                │  Local)       │
  └──────────────┘                                └──────────────┘
```

### Components

| Layer | Technology | Description |
|-------|-----------|-------------|
| **Data Ingestion** | Python, Open-Meteo API, AQICN API | Fetches historical hourly AQI, live snapshots, and weather data |
| **Feature Engineering** | pandas, NumPy | 120+ features: lags, rolling stats, weather forecasts, seasonal signals |
| **Model Training** | scikit-learn (VotingRegressor) | Horizon-specific ensembles with walk-forward validation |
| **Serving** | FastAPI + CORS | REST API with `/predict/karachi` endpoint returning Day 1/2/3 forecasts |
| **Dashboard** | Streamlit | Metric cards, trajectory charts, AQI warnings, feature importance |
| **CI/CD** | GitHub Actions | Automated feature + training pipelines |
| **Feature Store** | Hopsworks / Local CSV | Dual-mode with local-first fallback |
| **Deployment** | HuggingFace Spaces | Live demo at [roshaan727-pearls.hf.space](https://roshaan727-pearls.hf.space) |

---

## Live Demo

**Dashboard:** [https://roshaan727-pearls.hf.space](https://roshaan727-pearls.hf.space)

The live demo shows:
- **Day 1 / Day 2 / Day 3** daily-average PM2.5 forecasts in metric cards
- **AQI conversion** using US EPA breakpoints with color-coded bands
- **Hazardous alert system** (st.error for AQI >= 150, st.warning for AQI >= 100)
- **3-Day Forecast Trajectory** line chart
- **"What is driving today's AQI?"** feature importance bar chart (explainability)

---

## Models Used

We train **3 horizon-specific models**, each a **VotingRegressor ensemble** combining **2 algorithms**:

| Horizon | Target | Algorithm 1 | Algorithm 2 |
|---------|--------|-------------|-------------|
| **Day 1** | Avg PM2.5 (t+1 to t+24) | HistGradientBoostingRegressor | RandomForestRegressor (n=50) |
| **Day 2** | Avg PM2.5 (t+25 to t+48) | HistGradientBoostingRegressor | RandomForestRegressor (n=50) |
| **Day 3** | Avg PM2.5 (t+49 to t+72) | HistGradientBoostingRegressor | RandomForestRegressor (n=50) |

**Why VotingRegressor?**
- HistGradientBoostingRegressor captures non-linear weather/pollutant dynamics
- RandomForestRegressor adds stability and provides `feature_importances_` for explainability
- Combining both reduces variance and improves robustness over single-algorithm models

---

## Feature Engineering

### Input Data Sources
- **AQICN**: Live PM2.5, PM10, O3, NO2, SO2, CO, temperature, humidity, wind speed
- **Open-Meteo Historical**: Hourly pollutant concentrations from 2021 to present
- **Open-Meteo Weather**: Historical temperature, humidity, wind speed, precipitation
- **Open-Meteo Forecast**: 7-day weather forecast for future covariates

### Engineered Features (120+)
| Category | Features |
|----------|----------|
| **Pollutant Lags** | PM2.5, PM10, O3, NO2 at 1h, 3h, 6h, 12h, 24h, 48h, 72h |
| **Rolling Statistics** | Mean, std, min, max over 6h, 12h, 24h, 48h, 72h windows |
| **Weather Lags** | Temperature, humidity, wind speed at 1h, 3h, 6h, 12h, 24h |
| **Weather Forecasts** | Temperature, humidity, wind at t+24h, t+48h, t+72h |
| **Trend/Volatility** | EWM means, diff features (1h, 3h, 24h) |
| **Seasonal** | Hour sin/cos, month sin/cos, day-of-year sin/cos |
| **Interaction** | Temperature * humidity, PM2.5/PM10 ratio |
| **Calendar** | Hour, day, month, day-of-week, is_weekend |

### Target Variables (Daily Averages)
- **Day 1**: Mean PM2.5 from t+1 to t+24
- **Day 2**: Mean PM2.5 from t+25 to t+48
- **Day 3**: Mean PM2.5 from t+49 to t+72

---

## Performance Results

| Horizon | RMSE | MAE | R² |
|---------|------|-----|-----|
| **Day 1** (t+1 to t+24) | 6.25 | 4.46 | **0.614** |
| **Day 2** (t+25 to t+48) | 8.93 | 6.61 | 0.214 |
| **Day 3** (t+49 to t+72) | 9.84 | 7.30 | 0.046 |
| **Overall** | 8.34 | 6.12 | 0.291 |

Day 1 achieved **R² = 0.614** — the most operationally valuable horizon for public health decisions.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root health check |
| `/health` | GET | Service health status |
| `/predict/karachi` | GET | **Main endpoint**: Day 1/2/3 daily-average PM2.5 + AQI + feature importance |
| `/predict/latest` | GET | Legacy hourly +24h/+48h/+72h PM2.5 forecast |
| `/predict/latest/explain` | GET | LIME explanation for +24h prediction |
| `/debug/runtime` | GET | Runtime metadata (model source, data source, freshness) |

### Example Response (`/predict/karachi`)

```json
{
  "model_source": "local_artifact:horizon_specific_ensemble_20260605_235044",
  "city": "Karachi",
  "generated_from_timestamp": "2026-05-25T23:00:00+00:00",
  "prediction_unit": "ug/m^3",
  "daily_avg_pm2_5_forecast": {
    "day_1": 28.5,
    "day_2": 35.2,
    "day_3": 42.1
  },
  "daily_avg_aqi_forecast": {
    "day_1": 82,
    "day_2": 98,
    "day_3": 115
  },
  "feature_importance": {
    "pm2_5_lag_1h": 32.5,
    "pm2_5_roll_mean_24h": 24.1,
    "temperature_c_lag_6h": 15.8,
    "humidity_pct_lag_12h": 14.2,
    "pm10_lag_3h": 13.4
  },
  "unsafe_day1_alert": false
}
```

---

## Quick Start (Local)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables (`.env` in repo root)

```env
API_KEY=your_aqicn_token
USE_HOPSWORKS=false
UPLOAD_TO_HOPSWORKS=false
```

### 3. Run Full Pipeline

```bash
# Data ingestion
python src/data/historical_ingestion.py
python src/data/historical_weather_ingestion.py
python src/data/weather_forecast_ingestion.py
python src/data/ingestion.py

# Feature engineering
python src/features/build_features.py

# Model training (local-first, fast mode)
USE_HOPSWORKS=false UPLOAD_TO_HOPSWORKS=false FAST_TRAIN=true python src/models/train_model.py

# Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start Dashboard (in a separate terminal)
streamlit run src/app/dashboard.py
```

### 4. Verify

- API docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Prediction: http://localhost:8000/predict/karachi

---

## Repository Structure

```text
.
├── src/
│   ├── data/
│   │   ├── ingestion.py                    # AQICN live snapshot
│   │   ├── historical_ingestion.py         # Open-Meteo historical AQI
│   │   ├── historical_weather_ingestion.py # Open-Meteo historical weather
│   │   ├── weather_forecast_ingestion.py   # Open-Meteo forecast weather
│   │   └── eda_karachi.py                  # Exploratory data analysis
│   ├── features/
│   │   ├── build_features.py               # Feature engineering pipeline
│   │   └── push_to_hopsworks.py            # Hopsworks feature store upload
│   ├── models/
│   │   └── train_model.py                  # Model training + evaluation
│   ├── api/
│   │   └── main.py                         # FastAPI serving layer
│   └── app/
│       └── dashboard.py                    # Streamlit dashboard
├── docs/
│   └── Final_Report.md                     # Detailed project report
├── artifacts/models/                       # Trained model artifacts
├── data/
│   ├── raw/                                # Ingested raw data
│   └── processed/                          # Engineered features
├── reports/figures/                        # EDA visualizations
├── notebooks/                              # EDA notebooks
├── .github/workflows/                      # CI/CD pipelines
├── requirements.txt                        # Pinned dependencies
└── README.md                               # This file
```

---

## Automation (GitHub Actions)

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| **Feature Pipeline** | Manual / Scheduled | Ingests data → builds features → pushes to feature store |
| **Training Pipeline** | Daily / Manual | Fetches features → trains models → uploads to model registry |

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_HOPSWORKS` | `false` | Enable Hopsworks cloud feature store |
| `UPLOAD_TO_HOPSWORKS` | `false` | Upload trained models to Hopsworks registry |
| `FAST_TRAIN` | `false` | Use minimal model grid for faster training |
| `ENFORCE_FRESH_FEATURES` | `false` | Block inference on stale data |
| `MAX_FEATURE_AGE_HOURS` | `48` | Maximum allowed feature age |
| `ALLOW_LOCAL_FEATURE_FALLBACK` | `true` | Fall back to local CSV on Hopsworks failure |
| `FEATURE_CACHE_TTL_SECONDS` | `120` | In-memory feature cache TTL |
| `API_BASE_URL` | `http://localhost:8000` | Backend URL for dashboard |
| `HAZARDOUS_PM25_THRESHOLD` | `250.5` | PM2.5 hazardous alert threshold |

---

## Tech Stack

- **Python 3.11** — Core language
- **scikit-learn** — ML models (VotingRegressor, HistGradientBoosting, RandomForest)
- **pandas / NumPy** — Data manipulation and feature engineering
- **FastAPI** — REST API serving layer
- **Streamlit** — Interactive dashboard
- **Hopsworks** — Cloud feature store and model registry (optional)
- **GitHub Actions** — CI/CD automation
- **HuggingFace Spaces** — Live deployment
- **LIME** — Model explainability

---

## License

This project was developed as part of the **10Pearls Internship Program**.

---

*Built with by Muhammad Roshaan — 10Pearls AQI Predictor Project*
