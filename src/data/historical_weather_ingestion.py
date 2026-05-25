"""Historical weather ingestion for Karachi using Open-Meteo Weather API."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Response
from requests.exceptions import HTTPError, RequestException, Timeout


OPEN_METEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_PATH = Path("data/raw/karachi_historical_weather.csv")


def fetch_historical_weather() -> dict[str, Any]:
    """Fetch hourly historical weather data for Karachi."""
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": (
            "temperature_2m,relativehumidity_2m,windspeed_10m,precipitation"
        ),
        "timezone": "Asia/Karachi",
    }

    try:
        response: Response = requests.get(
            OPEN_METEO_WEATHER_URL,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    except Timeout as exc:
        raise TimeoutError(
            "Open-Meteo request timed out after "
            f"{REQUEST_TIMEOUT_SECONDS} seconds."
        ) from exc

    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "Unknown"
        response_body = exc.response.text[:500] if exc.response is not None else "No body"
        raise RuntimeError(
            f"Open-Meteo HTTP error {status_code}. Response: {response_body}"
        ) from exc

    except RequestException as exc:
        raise ConnectionError(f"Open-Meteo request failed: {exc}") from exc


def parse_weather_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten Open-Meteo hourly arrays into a tabular dataframe."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Invalid payload: expected 'hourly' dictionary.")

    expected_keys = [
        "time",
        "temperature_2m",
        "relativehumidity_2m",
        "windspeed_10m",
        "precipitation",
    ]

    missing_keys = [key for key in expected_keys if key not in hourly]
    if missing_keys:
        raise ValueError(f"Missing expected hourly fields: {missing_keys}")

    df = pd.DataFrame(hourly)
    df = df.rename(
        columns={
            "time": "timestamp",
            "temperature_2m": "temperature_c",
            "relativehumidity_2m": "humidity_pct",
            "windspeed_10m": "wind_speed_kph",
            "precipitation": "precipitation_mm",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df = df.rename(columns={"wind_speed_kph": "wind_speed"})
    df = df.rename(columns={"precipitation_mm": "precipitation"})
    return df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataframe to CSV, creating parent directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run full historical weather ingestion workflow for Karachi."""
    payload = fetch_historical_weather()
    df = parse_weather_payload(payload)
    save_dataframe(df, OUTPUT_PATH)
    print(f"Saved {len(df)} hourly weather rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
