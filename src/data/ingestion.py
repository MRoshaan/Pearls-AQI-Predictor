"""Data ingestion script for Karachi AQI data using AQICN API."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from requests import Response
from requests.exceptions import HTTPError, RequestException, Timeout


AQICN_BASE_URL = "https://api.waqi.info/feed/karachi/"
REQUEST_TIMEOUT_SECONDS = 20
OUTPUT_PATH = Path("data/raw/karachi_aqi_raw.csv")


def load_api_key() -> str:
    """Load AQICN API token from .env/environment."""
    load_dotenv()
    api_key = os.getenv("API_KEY") or os.getenv("AQICN_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing AQICN API token. Set API_KEY (or AQICN_API_KEY) in .env."
        )

    return api_key


def get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute an HTTP GET request and return parsed JSON response."""
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
            f"Request timed out after {REQUEST_TIMEOUT_SECONDS}s for URL: {url}"
        ) from exc

    except HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "Unknown"
        response_body = exc.response.text[:500] if exc.response is not None else "No body"
        raise RuntimeError(
            f"HTTP error {status_code} for URL {url}. Response: {response_body}"
        ) from exc

    except RequestException as exc:
        raise ConnectionError(f"Request failed for URL {url}: {exc}") from exc


def fetch_karachi_aqi(api_key: str) -> dict[str, Any]:
    """Fetch current AQI payload for Karachi from AQICN API."""
    payload = get_json(AQICN_BASE_URL, {"token": api_key})

    if payload.get("status") != "ok":
        raise ValueError(
            "AQICN API returned non-ok status: "
            f"{payload.get('status')} | data: {payload.get('data')}"
        )

    return payload


def parse_karachi_aqi(payload: dict[str, Any]) -> pd.DataFrame:
    """Parse AQICN JSON into a tabular dataframe with relevant fields."""
    data = payload.get("data", {})
    iaqi = data.get("iaqi", {})
    city = data.get("city", {})
    time_info = data.get("time", {})

    row = {
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": city.get("name", "Karachi"),
        "aqi": data.get("aqi"),
        "dominant_pollutant": data.get("dominentpol"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "temperature_c": iaqi.get("t", {}).get("v"),
        "humidity_pct": iaqi.get("h", {}).get("v"),
        "pressure_hpa": iaqi.get("p", {}).get("v"),
        "wind_speed": iaqi.get("w", {}).get("v"),
        "aqicn_time_iso": time_info.get("iso"),
        "aqicn_time_s": time_info.get("s"),
        "aqicn_timezone": time_info.get("tz"),
    }

    df = pd.DataFrame([row])
    df["aqicn_time_iso"] = pd.to_datetime(df["aqicn_time_iso"], errors="coerce")
    return df


def save_raw_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataframe to CSV, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run Karachi AQI ingestion and save to local raw CSV."""
    try:
        api_key = load_api_key()
        payload = fetch_karachi_aqi(api_key)
        karachi_df = parse_karachi_aqi(payload)
        save_raw_data(karachi_df, OUTPUT_PATH)

        print(
            "Karachi AQI data saved successfully "
            f"({len(karachi_df)} row) to {OUTPUT_PATH}"
        )

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Data ingestion failed: {exc}")
        raise


if __name__ == "__main__":
    main()
