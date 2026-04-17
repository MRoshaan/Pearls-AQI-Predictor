"""Build feature dataset for Karachi AQI forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/karachi_historical_aqi.csv")
OUTPUT_PATH = Path("data/processed/karachi_features.csv")


def load_historical_data(file_path: Path) -> pd.DataFrame:
    """Load historical hourly pollutant data and validate schema."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Historical dataset not found at {file_path}. "
            "Run src/data/historical_ingestion.py first."
        )

    df = pd.read_csv(file_path)

    time_col = "timestamp" if "timestamp" in df.columns else "time"
    if time_col not in df.columns:
        raise ValueError("Input file must contain 'timestamp' or 'time' column.")

    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required_cols = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required pollutant columns: {missing_cols}")

    return df


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values with forward and backward fill for time series."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features."""
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["month"] = ts.dt.month
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and AQI-change style derived features."""
    df["aqi_proxy"] = df["pm2_5"]
    df["aqi_change_rate"] = df["aqi_proxy"].pct_change()

    lag_hours = [1, 3, 24]
    for lag in lag_hours:
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
        df[f"pm10_lag_{lag}h"] = df["pm10"].shift(lag)

    df["pm2_5_roll_mean_6h"] = df["pm2_5"].rolling(window=6).mean()
    df["pm2_5_roll_mean_24h"] = df["pm2_5"].rolling(window=24).mean()
    df["pm10_roll_mean_6h"] = df["pm10"].rolling(window=6).mean()
    df["pm10_roll_mean_24h"] = df["pm10"].rolling(window=24).mean()

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add forecasting targets for next 24/48/72 hours."""
    horizons = [24, 48, 72]
    for horizon in horizons:
        df[f"target_pm2_5_t_plus_{horizon}h"] = df["pm2_5"].shift(-horizon)

    return df


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save engineered features to local processed CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run full feature engineering pipeline for Karachi AQI."""
    df = load_historical_data(INPUT_PATH)
    df = clean_time_series(df)
    df = add_time_features(df)
    df = add_derived_features(df)
    df = add_targets(df)

    df = df.dropna().reset_index(drop=True)
    save_features(df, OUTPUT_PATH)

    print(f"Saved {len(df)} engineered rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
