"""Streamlit dashboard for AQI forecasting and explainability."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Karachi AQI Forecast", page_icon="AQI", layout="wide")

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def aqi_band(pm25: float) -> tuple[str, str]:
    """Map PM2.5 concentration to AQI-style health band."""
    if pm25 <= 12:
        return "Good", "#2e7d32"
    if pm25 <= 35.4:
        return "Moderate", "#ef6c00"
    if pm25 <= 55.4:
        return "Unhealthy (Sensitive)", "#f9a825"
    if pm25 <= 150.4:
        return "Unhealthy", "#d32f2f"
    if pm25 <= 250.4:
        return "Very Unhealthy", "#6a1b9a"
    return "Hazardous", "#4e342e"


def get_json(endpoint: str) -> dict[str, Any]:
    """Fetch JSON from FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=30)
    response.raise_for_status()
    return response.json()


def render_header() -> None:
    """Render dashboard heading and context."""
    st.title("Karachi AQI 3-Day Forecast")
    st.caption("Live PM2.5 forecast from Hopsworks model registry and feature store")


def render_hazard_banner(is_hazardous: bool) -> None:
    """Render visual alert for hazardous forecast."""
    if is_hazardous:
        st.error("Hazard alert: forecast reaches hazardous PM2.5 levels. Avoid outdoor exposure.")
    else:
        st.success("Forecast is below hazardous PM2.5 threshold.")


def render_forecast_cards(payload: dict[str, Any]) -> None:
    """Render forecast cards for +24h/+48h/+72h horizons."""
    cols = st.columns(3)
    order = ["+24h", "+48h", "+72h"]

    for col, horizon in zip(cols, order):
        value = float(payload["pm2_5_forecast"][horizon])
        band, color = aqi_band(value)
        with col:
            st.markdown(
                (
                    f"<div style='padding:1rem;border-radius:14px;border:1px solid #ddd;'>"
                    f"<div style='font-size:0.9rem;color:#666'>{horizon}</div>"
                    f"<div style='font-size:2rem;font-weight:700'>{value:.2f}</div>"
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


def render_explainability(payload: dict[str, Any]) -> None:
    """Render SHAP top-contributor table and chart."""
    st.markdown("### Why the model predicted this")
    top = payload.get("top_features", [])
    if not top:
        st.info("No explainability results returned.")
        return

    df = pd.DataFrame(top)
    st.dataframe(df, use_container_width=True)

    chart_df = df.sort_values("abs_shap_value", ascending=True)
    st.bar_chart(
        chart_df.set_index("feature")["shap_value"],
        horizontal=True,
    )


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
    render_forecast_cards(prediction)
    render_metadata(prediction)
    render_explainability(explanation)


if __name__ == "__main__":
    main()
