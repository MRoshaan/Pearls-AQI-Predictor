"""Historical AQI ingestion for Karachi using Open-Meteo Air Quality API."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests import Response
from requests.exceptions import HTTPError, RequestException, Timeout


OPEN_METEO_AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
REQUEST_TIMEOUT_SECONDS = 30
OUTPUT_PATH = Path("data/raw/karachi_historical_aqi.csv")


def fetch_historical_air_quality() -> dict[str, Any]:
    """Fetch hourly historical air quality data for Karachi."""
    params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "start_date": "2021-01-01",
        "end_date": date.today().isoformat(),
        "hourly": (
            "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,"
            "sulphur_dioxide,ozone"
        ),
        "timezone": "Asia/Karachi",
    }

    try:
        response: Response = requests.get(
            OPEN_METEO_AIR_QUALITY_URL,
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


def parse_hourly_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Flatten Open-Meteo hourly arrays into a tabular dataframe."""
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Invalid payload: expected 'hourly' dictionary.")

    expected_keys = [
        "time",
        "pm10",
        "pm2_5",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]

    missing_keys = [key for key in expected_keys if key not in hourly]
    if missing_keys:
        raise ValueError(f"Missing expected hourly fields: {missing_keys}")

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataframe to CSV, creating parent directory if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run full historical ingestion workflow for Karachi."""
    payload = fetch_historical_air_quality()
    df = parse_hourly_payload(payload)
    save_dataframe(df, OUTPUT_PATH)
    print(f"Saved {len(df)} hourly rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
