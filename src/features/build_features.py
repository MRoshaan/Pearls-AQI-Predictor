"""Build feature dataset for Karachi AQI forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/karachi_historical_aqi.csv")
CURRENT_INPUT_PATH = Path("data/raw/karachi_aqi_raw.csv")
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


def load_current_snapshot(file_path: Path) -> pd.DataFrame:
    """Load latest AQICN snapshot and align columns to historical schema."""
    if not file_path.exists():
        return pd.DataFrame()

    current_df = pd.read_csv(file_path)
    if current_df.empty:
        return pd.DataFrame()

    current_df = current_df.rename(
        columns={
            "pm25": "pm2_5",
            "o3": "ozone",
            "no2": "nitrogen_dioxide",
            "so2": "sulphur_dioxide",
            "co": "carbon_monoxide",
        }
    )

    if "aqicn_time_iso" not in current_df.columns:
        current_df["aqicn_time_iso"] = np.nan
    if "ingested_at_utc" not in current_df.columns:
        current_df["ingested_at_utc"] = np.nan

    aqicn_ts = pd.to_datetime(current_df["aqicn_time_iso"], errors="coerce", utc=True)
    ingested_ts = pd.to_datetime(current_df["ingested_at_utc"], errors="coerce", utc=True)

    max_aqicn = aqicn_ts.max()
    max_ingested = ingested_ts.max()

    use_ingested_time = pd.isna(max_aqicn)
    if not use_ingested_time and not pd.isna(max_ingested):
        lag_hours = (max_ingested - max_aqicn).total_seconds() / 3600
        use_ingested_time = lag_hours > 24

    if use_ingested_time:
        current_df["timestamp"] = ingested_ts.dt.tz_convert(None)
    else:
        current_df["timestamp"] = aqicn_ts.dt.tz_convert(None)

    required_cols = [
        "timestamp",
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]
    for col in required_cols:
        if col not in current_df.columns:
            current_df[col] = np.nan

    current_df = cast(pd.DataFrame, current_df[required_cols].copy())
    mask = pd.notna(current_df["timestamp"])
    current_df = cast(pd.DataFrame, current_df.loc[mask].copy())
    if current_df.empty:
        return pd.DataFrame()

    return current_df.sort_values("timestamp").tail(1).reset_index(drop=True)


def merge_historical_with_current(historical_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Append latest current snapshot to historical data and deduplicate by timestamp."""
    if current_df.empty:
        return historical_df

    merged = pd.concat([historical_df, current_df], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def clean_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values with forward and backward fill for time series."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan)
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
    """Add lag, rolling, volatility, and trend features."""
    df["aqi_proxy"] = df["pm2_5"]
    df["aqi_change_rate"] = df["aqi_proxy"].pct_change()
    df["pm2_5_to_pm10_ratio"] = df["pm2_5"] / (df["pm10"] + 1e-6)

    lag_hours = [1, 3, 6, 12, 24, 48, 72]
    for lag in lag_hours:
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)
        df[f"pm10_lag_{lag}h"] = df["pm10"].shift(lag)
        df[f"ozone_lag_{lag}h"] = df["ozone"].shift(lag)
        df[f"nitrogen_dioxide_lag_{lag}h"] = df["nitrogen_dioxide"].shift(lag)

    roll_windows = [6, 12, 24, 48, 72]
    for window in roll_windows:
        df[f"pm2_5_roll_mean_{window}h"] = df["pm2_5"].rolling(window=window).mean()
        df[f"pm2_5_roll_std_{window}h"] = df["pm2_5"].rolling(window=window).std()
        df[f"pm2_5_roll_min_{window}h"] = df["pm2_5"].rolling(window=window).min()
        df[f"pm2_5_roll_max_{window}h"] = df["pm2_5"].rolling(window=window).max()

        df[f"pm10_roll_mean_{window}h"] = df["pm10"].rolling(window=window).mean()
        df[f"pm10_roll_std_{window}h"] = df["pm10"].rolling(window=window).std()

    df["pm2_5_ewm_mean_12h"] = df["pm2_5"].ewm(span=12, adjust=False).mean()
    df["pm2_5_ewm_mean_24h"] = df["pm2_5"].ewm(span=24, adjust=False).mean()
    df["pm2_5_diff_1h"] = df["pm2_5"].diff(1)
    df["pm2_5_diff_3h"] = df["pm2_5"].diff(3)
    df["pm2_5_diff_24h"] = df["pm2_5"].diff(24)

    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

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
    current_df = load_current_snapshot(CURRENT_INPUT_PATH)
    df = merge_historical_with_current(df, current_df)
    df = clean_time_series(df)
    df = add_time_features(df)
    df = add_derived_features(df)
    df = add_targets(df)

    feature_only_cols = [
        col
        for col in df.columns
        if col not in [
            "timestamp",
            "target_pm2_5_t_plus_24h",
            "target_pm2_5_t_plus_48h",
            "target_pm2_5_t_plus_72h",
        ]
    ]
    df = df.dropna(subset=feature_only_cols).reset_index(drop=True)
    save_features(df, OUTPUT_PATH)

    print(f"Saved {len(df)} engineered rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
