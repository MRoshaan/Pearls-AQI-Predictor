"""Streamlit dashboard for PM2.5 forecasting and explainability."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Karachi PM2.5 Forecast", page_icon="AQI", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def pm25_to_aqi(pm25: float) -> int:
    """Convert PM2.5 concentration (ug/m^3) to US EPA AQI."""
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    clamped = max(0.0, min(pm25, 500.4))
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= clamped <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (clamped - c_low) + i_low
            return int(round(aqi))
    return 500


def aqi_category(aqi: int) -> tuple[str, str]:
    """Map AQI numeric value to category and color."""
    if aqi <= 50:
        return "Good", "#2e7d32"
    if aqi <= 100:
        return "Moderate", "#ef6c00"
    if aqi <= 150:
        return "Unhealthy (Sensitive)", "#f9a825"
    if aqi <= 200:
        return "Unhealthy", "#d32f2f"
    if aqi <= 300:
        return "Very Unhealthy", "#6a1b9a"
    return "Hazardous", "#4e342e"


def get_json(endpoint: str) -> dict[str, Any]:
    """Fetch JSON from FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def render_header() -> None:
    """Render dashboard heading and context."""
    st.title("Karachi PM2.5 3-Day Forecast")
    st.caption("Live PM2.5 concentration forecast from Hopsworks model registry and feature store")
    st.info("Predictions show PM2.5 in ug/m^3 and converted AQI index (US EPA formula).")


def render_hazard_banner(is_hazardous: bool) -> None:
    """Render visual alert for hazardous forecast."""
    if is_hazardous:
        st.error("Hazard alert: forecast reaches hazardous PM2.5 levels. Avoid outdoor exposure.")
    else:
        st.success("Forecast is below hazardous PM2.5 threshold.")


def render_freshness_banner(payload: dict[str, Any]) -> None:
    """Show warning if source data is stale."""
    raw_ts = payload.get("generated_from_timestamp")
    if not raw_ts:
        return

    try:
        ts = pd.to_datetime(raw_ts, utc=True)
    except Exception:
        return

    now = datetime.now(timezone.utc)
    age_hours = (now - ts.to_pydatetime()).total_seconds() / 3600

    if age_hours > 24:
        st.warning(
            f"Data freshness warning: latest feature timestamp is about {age_hours:.1f} hours old "
            f"({ts.strftime('%Y-%m-%d %H:%M:%S %Z')})."
        )


def render_forecast_cards(payload: dict[str, Any]) -> None:
    """Render forecast cards for +24h/+48h/+72h horizons."""
    cols = st.columns(3)
    order = ["+24h", "+48h", "+72h"]

    for col, horizon in zip(cols, order):
        pm25_value = float(payload["pm2_5_forecast"][horizon])
        aqi_value = pm25_to_aqi(pm25_value)
        band, color = aqi_category(aqi_value)
        with col:
            st.markdown(
                (
                    "<div style='padding:1rem;border-radius:14px;border:1px solid #ddd;'>"
                    f"<div style='font-size:0.9rem;color:#666'>{horizon}</div>"
                    f"<div style='font-size:2rem;font-weight:700'>{pm25_value:.2f} ug/m^3</div>"
                    f"<div style='font-size:1rem;color:#99a'>AQI: {aqi_value}</div>"
                    f"<div style='color:{color};font-weight:600'>{band}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_metadata(payload: dict[str, Any]) -> None:
    """Render model provenance and inference context."""
    st.markdown("### Inference Context")
    st.write(f"City: `{payload['city']}`")
    st.write(f"Source timestamp: `{payload['generated_from_timestamp']}`")
    st.write(f"Model source: `{payload['model_source']}`")
    st.write(f"Feature source: `{payload.get('feature_source', 'unknown')}`")
    st.write(f"Prediction unit: `{payload.get('prediction_unit', 'ug/m^3')}`")


def render_explainability(payload: dict[str, Any]) -> None:
    """Render LIME top-contributor table and chart."""
    st.markdown("### Why the model predicted this")
    st.caption("Feature contribution scores are from LIME for the +24h forecast horizon.")
    top = payload.get("top_features", [])
    if not top:
        st.info("No explainability results returned.")
        return

    df = pd.DataFrame(top)
    st.dataframe(df, use_container_width=True)

    chart_df = df.sort_values("abs_explanation_score", ascending=True)
    st.bar_chart(chart_df.set_index("feature")["explanation_score"], horizontal=True)


def main() -> None:
    """Dashboard entry point."""
    render_header()

    with st.sidebar:
        st.markdown("### Backend")
        st.code(API_BASE_URL)
        st.caption("Set API_BASE_URL env var if running backend on another host/port.")

    try:
        prediction = get_json("/predict/latest")
        explanation = get_json("/predict/latest/explain?max_features=10")
    except Exception as exc:
        st.error(f"Failed to fetch forecast data from backend: {exc}")
        st.stop()

    render_hazard_banner(bool(prediction["hazardous_alert"]))
    render_freshness_banner(prediction)
    render_forecast_cards(prediction)
    render_metadata(prediction)
    render_explainability(explanation)


if __name__ == "__main__":
    main()
