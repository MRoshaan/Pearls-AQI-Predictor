"""Data ingestion script for Karachi AQI and weather data.

This module fetches current and historical air-pollution data from OpenWeather,
joins it with weather observations/forecast data, and stores raw combined data
to a local CSV file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import HTTPError, RequestException, Timeout


@dataclass(frozen=True)
class CityConfig:
    """Configuration for a city location."""

    name: str
    lat: float
    lon: float


KARACHI = CityConfig(name="Karachi", lat=24.8607, lon=67.0011)

OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_SECONDS = 20
HISTORY_DAYS = 5


def load_api_key() -> str:
    """Load OpenWeather API key from environment."""
    load_dotenv()
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing OPENWEATHER_API_KEY. Add it to your .env file before running."
        )

    return api_key


def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a GET request and return parsed JSON with robust handling."""
    try:
        response: Response = requests.get(
            url=url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    except Timeout as exc:
        raise TimeoutError(
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s for: {url}"
        ) from exc

    except HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "Unknown"
        body = exc.response.text[:500] if exc.response is not None else "No body"
        raise RuntimeError(
            f"HTTP error {status} for {url}. Response body: {body}"
        ) from exc

    except RequestException as exc:
        raise ConnectionError(f"Request failed for {url}: {exc}") from exc


def fetch_current_air_pollution(city: CityConfig, api_key: str) -> dict[str, Any]:
    """Fetch current AQI/pollutant data."""
    url = f"{OPENWEATHER_BASE_URL}/air_pollution"
    params = {"lat": city.lat, "lon": city.lon, "appid": api_key}
    return get_json(url, params)


def fetch_historical_air_pollution(
    city: CityConfig,
    api_key: str,
    start_unix: int,
    end_unix: int,
) -> dict[str, Any]:
    """Fetch historical AQI/pollutant data for a time range."""
    url = f"{OPENWEATHER_BASE_URL}/air_pollution/history"
    params = {
        "lat": city.lat,
        "lon": city.lon,
        "start": start_unix,
        "end": end_unix,
        "appid": api_key,
    }
    return get_json(url, params)


def fetch_current_weather(city: CityConfig, api_key: str) -> dict[str, Any]:
    """Fetch current weather data for the city."""
    url = f"{OPENWEATHER_BASE_URL}/weather"
    params = {"lat": city.lat, "lon": city.lon, "appid": api_key, "units": "metric"}
    return get_json(url, params)


def fetch_weather_forecast(city: CityConfig, api_key: str) -> dict[str, Any]:
    """Fetch 5-day/3-hour weather forecast data for the city."""
    url = f"{OPENWEATHER_BASE_URL}/forecast"
    params = {"lat": city.lat, "lon": city.lon, "appid": api_key, "units": "metric"}
    return get_json(url, params)


def normalize_air_pollution_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize OpenWeather air-pollution payload into tabular rows."""
    rows: list[dict[str, Any]] = []

    for item in payload.get("list", []):
        components = item.get("components", {})
        row = {
            "timestamp": pd.to_datetime(item.get("dt"), unit="s", utc=True),
            "aqi": item.get("main", {}).get("aqi"),
            "co": components.get("co"),
            "no": components.get("no"),
            "no2": components.get("no2"),
            "o3": components.get("o3"),
            "so2": components.get("so2"),
            "pm2_5": components.get("pm2_5"),
            "pm10": components.get("pm10"),
            "nh3": components.get("nh3"),
        }
        rows.append(row)

    return rows


def normalize_weather_rows(
    current_weather: dict[str, Any],
    forecast_weather: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize current and forecast weather payloads into tabular rows."""
    rows: list[dict[str, Any]] = []

    current_row = {
        "timestamp": pd.to_datetime(current_weather.get("dt"), unit="s", utc=True),
        "temp_c": current_weather.get("main", {}).get("temp"),
        "feels_like_c": current_weather.get("main", {}).get("feels_like"),
        "humidity_pct": current_weather.get("main", {}).get("humidity"),
        "pressure_hpa": current_weather.get("main", {}).get("pressure"),
        "wind_speed_mps": current_weather.get("wind", {}).get("speed"),
        "clouds_pct": current_weather.get("clouds", {}).get("all"),
    }
    rows.append(current_row)

    for item in forecast_weather.get("list", []):
        rows.append(
            {
                "timestamp": pd.to_datetime(item.get("dt"), unit="s", utc=True),
                "temp_c": item.get("main", {}).get("temp"),
                "feels_like_c": item.get("main", {}).get("feels_like"),
                "humidity_pct": item.get("main", {}).get("humidity"),
                "pressure_hpa": item.get("main", {}).get("pressure"),
                "wind_speed_mps": item.get("wind", {}).get("speed"),
                "clouds_pct": item.get("clouds", {}).get("all"),
            }
        )

    return rows


def build_dataset(city: CityConfig, api_key: str) -> pd.DataFrame:
    """Fetch, normalize, and merge AQI and weather datasets."""
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=HISTORY_DAYS)

    historical_air = fetch_historical_air_pollution(
        city=city,
        api_key=api_key,
        start_unix=int(start_time.timestamp()),
        end_unix=int(end_time.timestamp()),
    )
    current_air = fetch_current_air_pollution(city=city, api_key=api_key)
    current_weather = fetch_current_weather(city=city, api_key=api_key)
    forecast_weather = fetch_weather_forecast(city=city, api_key=api_key)

    air_rows = normalize_air_pollution_rows(historical_air)
    air_rows.extend(normalize_air_pollution_rows(current_air))
    weather_rows = normalize_weather_rows(current_weather, forecast_weather)

    if not air_rows:
        raise ValueError("No AQI records returned by API; cannot build dataset.")

    air_df = pd.DataFrame(air_rows).drop_duplicates(subset=["timestamp"])
    weather_df = pd.DataFrame(weather_rows).drop_duplicates(subset=["timestamp"])

    air_df = air_df.sort_values("timestamp").reset_index(drop=True)
    weather_df = weather_df.sort_values("timestamp").reset_index(drop=True)

    merged_df = pd.merge_asof(
        left=air_df,
        right=weather_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("3h"),
    )

    merged_df["city"] = city.name
    return merged_df


def save_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    """Persist dataframe to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run data ingestion for Karachi and save local raw CSV."""
    output_file = Path("data/raw/karachi_aqi_raw.csv")

    try:
        api_key = load_api_key()
        dataset = build_dataset(city=KARACHI, api_key=api_key)
        save_raw_data(dataset, output_file)

        print(
            "Successfully saved Karachi AQI raw data "
            f"({len(dataset)} rows) to {output_file}"
        )

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Data ingestion failed: {exc}")
        raise


if __name__ == "__main__":
    main()
