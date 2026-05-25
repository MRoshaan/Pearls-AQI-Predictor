# Karachi AQI Forecasting Platform - Final Report

## Executive Summary

This project delivers an end-to-end AQI forecasting platform for Karachi, Pakistan, focused on practical reliability and explainability for real-world use. The system predicts three daily-average horizons (Day 1, Day 2, Day 3) from engineered air-quality and weather features, then serves results through FastAPI and a Streamlit dashboard.

During development, we shifted from point-in-time targets (exact t+24/t+48/t+72 hour values) to daily-average targets. This reduced target noise and improved the most business-critical horizon (Day 1) from roughly 39% R2 to about 61% R2. The final solution also includes stale-data guards, local-first resiliency, and model decision transparency through feature importance visualization.

## Serverless Architecture

### CI/CD and Data/Model Automation

- GitHub Actions orchestrates scheduled and manual data/model jobs.
- Data and feature scripts produce reproducible artifacts locally.
- Model training produces versioned artifacts in `artifacts/models/`.

### Serving Layer (FastAPI)

- FastAPI exposes forecast endpoints for applications and dashboard clients.
- CORS is enabled for browser-based consumers.
- The presentation endpoint `/predict/karachi` returns:
  - Day 1/2/3 daily-average PM2.5 forecasts
  - AQI-converted values
  - Day-1 risk flag
  - top feature-importance drivers

### Frontend (Streamlit)

- Streamlit consumes FastAPI over HTTP (`/predict/karachi`).
- Dashboard provides:
  - Day 1/2/3 metric cards
  - AQI warning banner
  - 3-day trajectory chart
  - "What is driving today's AQI?" feature-importance bar chart

## Feature Engineering and Modeling Decisions

## Initial Challenge

The original target design predicted single hourly spikes at exactly +24h, +48h, +72h. These targets were highly volatile and reduced model stability and explainability.

## Target Redesign (Daily Averages)

Targets were redefined as future daily windows:

- Day 1 target: mean PM2.5 from t+1 to t+24
- Day 2 target: mean PM2.5 from t+25 to t+48
- Day 3 target: mean PM2.5 from t+49 to t+72

This smoother target formulation improved robustness, especially for Day 1 planning decisions.

## Data Quality Controls

- Stale AQICN snapshots are rejected when source timestamp age exceeds 24 hours.
- Snapshot source/ingestion misalignment beyond 24 hours is rejected.
- Low-signal precipitation feature family is removed when variance is near zero.

## Final Model Strategy

- Horizon-specific ensemble models are trained for each target horizon.
- Each horizon uses a robust `VotingRegressor` combining:
  - `HistGradientBoostingRegressor` (non-linear weather/pollutant dynamics)
  - `RandomForestRegressor(n_estimators=50)` (stability and interpretability)

## Explainability

For each Day-1 prediction, the API extracts feature importance from the RandomForest component inside the VotingRegressor and computes a local weighted score using the current input row. The top five normalized contributors are returned in the API response and rendered in the dashboard.

## Performance Snapshot

After moving to daily-average targets and the final ensemble:

- Day 1 R2: ~0.61
- Day 2 R2: ~0.21
- Day 3 R2: lower, reflecting increasing uncertainty with horizon length

Day 1 was prioritized for operational decision value and presentation impact.

## How to Run Locally

## 1) Build Features

```bash
python src/data/historical_ingestion.py
python src/data/historical_weather_ingestion.py
python src/data/weather_forecast_ingestion.py
python src/features/build_features.py
```

## 2) Train Model (Local-First Fast Mode)

```bash
USE_HOPSWORKS=false UPLOAD_TO_HOPSWORKS=false FAST_TRAIN=true python src/models/train_model.py
```

## 3) Start API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Key endpoint:

- `GET /predict/karachi`

## 4) Start Dashboard

```bash
streamlit run src/app/dashboard.py
```

If needed, set backend URL:

```bash
API_BASE_URL=http://127.0.0.1:8000 streamlit run src/app/dashboard.py
```

## Conclusion

The final platform meets the internship deliverables with a production-style pipeline, explainable daily forecasts, and a polished user-facing dashboard. The architecture is robust enough for demos and extensible for further research, including longer-history weather forecasts, richer exogenous signals, and advanced uncertainty estimation.
