"""Build feature dataset for Karachi AQI forecasting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


INPUT_PATH = Path("data/raw/karachi_historical_aqi.csv")
CURRENT_INPUT_PATH = Path("data/raw/karachi_aqi_raw.csv")
HISTORICAL_WEATHER_PATH = Path("data/raw/karachi_historical_weather.csv")
WEATHER_FORECAST_PATH = Path("data/raw/karachi_weather_forecast.csv")
OUTPUT_PATH = Path("data/processed/karachi_features.csv")

ORIGINAL_POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "ozone",
]

WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed",
    "precipitation",
]

FORECAST_HORIZONS = [24, 48, 72]
MAX_CURRENT_SNAPSHOT_AGE_HOURS = 24.0
PRECIP_STD_DROP_THRESHOLD = 1e-3


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

    df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    missing_cols = [col for col in ORIGINAL_POLLUTANTS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required pollutant columns: {missing_cols}")

    return df


def load_historical_weather(file_path: Path) -> pd.DataFrame:
    """Load historical weather covariates aligned by timestamp."""
    if not file_path.exists():
        return pd.DataFrame()

    weather_df = pd.read_csv(file_path)
    if weather_df.empty:
        return pd.DataFrame()

    weather_df = weather_df.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relativehumidity_2m": "humidity_pct",
            "windspeed_10m": "wind_speed",
            "wind_speed_kph": "wind_speed",
            "precipitation_mm": "precipitation",
        }
    )

    time_col = "timestamp" if "timestamp" in weather_df.columns else "time"
    if time_col not in weather_df.columns:
        return pd.DataFrame()

    weather_df["timestamp"] = pd.to_datetime(weather_df[time_col], errors="coerce", format="mixed")
    weather_df = weather_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if weather_df.empty:
        return pd.DataFrame()

    for col in WEATHER_COLUMNS:
        if col not in weather_df.columns:
            weather_df[col] = np.nan

    weather_df = cast(pd.DataFrame, weather_df[["timestamp", *WEATHER_COLUMNS]].copy())
    weather_df = weather_df.drop_duplicates(subset=["timestamp"], keep="last")
    return weather_df.reset_index(drop=True)


def add_external_weather_to_current_snapshot(
    current_df: pd.DataFrame,
    weather_hist_df: pd.DataFrame,
) -> pd.DataFrame:
    """Backfill current snapshot weather from historical weather by nearest timestamp."""
    if current_df.empty or weather_hist_df.empty:
        return current_df

    snapshot_ts = pd.to_datetime(current_df.loc[current_df.index[-1], "timestamp"], errors="coerce")
    if pd.isna(snapshot_ts):
        return current_df

    deltas = (weather_hist_df["timestamp"] - snapshot_ts).abs()
    if deltas.empty:
        return current_df

    nearest_idx = deltas.idxmin()
    nearest_delta = deltas.loc[nearest_idx]
    if pd.isna(nearest_delta) or nearest_delta > pd.Timedelta(hours=2):
        return current_df

    nearest_weather = weather_hist_df.loc[nearest_idx]
    for col in WEATHER_COLUMNS:
        if col in current_df.columns and pd.notna(current_df.loc[current_df.index[-1], col]):
            continue
        current_df.loc[current_df.index[-1], col] = nearest_weather.get(col, np.nan)

    return current_df


def load_current_snapshot(file_path: Path) -> pd.DataFrame:
    """Load latest AQICN snapshot and align columns to training schema."""
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
    if not pd.isna(max_aqicn):
        aqicn_age_hours = (datetime.now(timezone.utc) - max_aqicn.to_pydatetime()).total_seconds() / 3600
        if aqicn_age_hours > MAX_CURRENT_SNAPSHOT_AGE_HOURS:
            print(
                "Skipping AQICN snapshot due to stale source timestamp: "
                f"aqicn_time={max_aqicn.isoformat()} age_hours={aqicn_age_hours:.2f} "
                f"limit={MAX_CURRENT_SNAPSHOT_AGE_HOURS:.2f}"
            )
            return pd.DataFrame()

    if not use_ingested_time and not pd.isna(max_ingested):
        lag_hours = (max_ingested - max_aqicn).total_seconds() / 3600
        if lag_hours > MAX_CURRENT_SNAPSHOT_AGE_HOURS:
            print(
                "Skipping AQICN snapshot due to source/ingestion misalignment: "
                f"lag_hours={lag_hours:.2f} limit={MAX_CURRENT_SNAPSHOT_AGE_HOURS:.2f}"
            )
            return pd.DataFrame()

    if use_ingested_time:
        current_df["timestamp"] = ingested_ts.dt.tz_convert(None)
    else:
        current_df["timestamp"] = aqicn_ts.dt.tz_convert(None)

    required_cols = ["timestamp", *ORIGINAL_POLLUTANTS, "temperature_c", "humidity_pct", "wind_speed"]
    for col in required_cols:
        if col not in current_df.columns:
            current_df[col] = np.nan

    current_df = cast(pd.DataFrame, current_df[required_cols].copy())
    mask = pd.notna(current_df["timestamp"])
    current_df = cast(pd.DataFrame, current_df.loc[mask].copy())
    if current_df.empty:
        return pd.DataFrame()

    current_df = current_df.sort_values("timestamp").tail(1).reset_index(drop=True)

    snapshot_ts = pd.to_datetime(current_df.loc[0, "timestamp"], errors="coerce", utc=True)
    if pd.isna(snapshot_ts):
        return pd.DataFrame()

    age_hours = (datetime.now(timezone.utc) - snapshot_ts.to_pydatetime()).total_seconds() / 3600
    if age_hours > MAX_CURRENT_SNAPSHOT_AGE_HOURS:
        print(
            "Skipping stale AQICN snapshot: "
            f"timestamp={snapshot_ts.isoformat()} age_hours={age_hours:.2f} "
            f"limit={MAX_CURRENT_SNAPSHOT_AGE_HOURS:.2f}"
        )
        return pd.DataFrame()

    return current_df


def load_weather_forecast(file_path: Path) -> pd.DataFrame:
    """Load weather forecast table used for future covariates."""
    if not file_path.exists():
        return pd.DataFrame()

    forecast_df = pd.read_csv(file_path)
    if forecast_df.empty:
        return pd.DataFrame()

    forecast_df = forecast_df.rename(
        columns={
            "temperature_2m": "temperature_c",
            "relativehumidity_2m": "humidity_pct",
            "windspeed_10m": "wind_speed",
            "wind_speed_kph": "wind_speed",
            "precipitation_mm": "precipitation",
        }
    )

    if "timestamp" not in forecast_df.columns:
        return pd.DataFrame()

    forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"], errors="coerce", format="mixed")
    forecast_df = forecast_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if forecast_df.empty:
        return pd.DataFrame()

    for col in WEATHER_COLUMNS:
        if col not in forecast_df.columns:
            forecast_df[col] = np.nan

    return cast(pd.DataFrame, forecast_df[["timestamp", *WEATHER_COLUMNS]].copy())


def merge_historical_with_current(historical_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    """Append latest current snapshot to historical data and deduplicate by timestamp."""
    if current_df.empty:
        return historical_df

    merged = pd.concat([historical_df, current_df], ignore_index=True)
    merged = merged.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return merged.reset_index(drop=True)


def merge_weather_on_timestamp(base_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join weather covariates onto base pollutant dataframe."""
    if base_df.empty:
        return base_df

    if weather_df.empty:
        for col in WEATHER_COLUMNS:
            if col not in base_df.columns:
                base_df[col] = np.nan
        return base_df

    merged = base_df.merge(weather_df, on="timestamp", how="left", suffixes=("", "_weather"))
    for col in WEATHER_COLUMNS:
        weather_col = f"{col}_weather"
        if weather_col in merged.columns:
            if col in merged.columns:
                merged[col] = cast(pd.Series, merged[col]).fillna(cast(pd.Series, merged[weather_col]))
            else:
                merged[col] = merged[weather_col]
            merged = merged.drop(columns=[weather_col])

    for col in WEATHER_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged.sort_values("timestamp").reset_index(drop=True)


def validate_input_quality(df: pd.DataFrame) -> None:
    """Fail fast when input data quality is insufficient for training features."""
    missing = [col for col in ORIGINAL_POLLUTANTS if col not in df.columns]
    if missing:
        raise ValueError(f"Input data missing required pollutants: {missing}")

    null_values = [float(df[col].isna().mean()) for col in ORIGINAL_POLLUTANTS]
    null_ratio = float(max(null_values))
    if null_ratio > 0.35:
        raise ValueError(
            "Input pollutant missingness too high for reliable feature generation: "
            f"max_null_ratio={null_ratio:.3f}"
        )

    duplicated_timestamps = int(df["timestamp"].duplicated().sum())
    if duplicated_timestamps > 0:
        raise ValueError(
            "Input contains duplicated timestamps before feature build: "
            f"duplicates={duplicated_timestamps}"
        )


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

    df["temp_humidity_interaction"] = df["temperature_c"] * df["humidity_pct"]
    for lag in [1, 3, 6, 12, 24]:
        df[f"temperature_c_lag_{lag}h"] = df["temperature_c"].shift(lag)
        df[f"humidity_pct_lag_{lag}h"] = df["humidity_pct"].shift(lag)
        df[f"wind_speed_lag_{lag}h"] = df["wind_speed"].shift(lag)
        df[f"precipitation_lag_{lag}h"] = df["precipitation"].shift(lag)

    return df


def add_weather_forecast_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Create horizon-specific future weather covariates for model input."""
    for horizon in FORECAST_HORIZONS:
        for col in WEATHER_COLUMNS:
            df[f"weather_forecast_{col}_t_plus_{horizon}h"] = df[col].shift(-horizon)
    return df


def inject_latest_weather_forecast(df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Inject external forecast values into the latest row future covariates."""
    if df.empty or forecast_df.empty:
        return df

    latest_idx = df.index[-1]
    latest_ts = pd.to_datetime(df.loc[latest_idx, "timestamp"], errors="coerce")
    if pd.isna(latest_ts):
        return df

    for horizon in FORECAST_HORIZONS:
        target_ts = latest_ts + pd.Timedelta(hours=horizon)
        deltas = (forecast_df["timestamp"] - target_ts).abs()
        if deltas.empty:
            continue

        nearest_idx = deltas.idxmin()
        nearest_delta = deltas.loc[nearest_idx]
        if pd.isna(nearest_delta) or nearest_delta > pd.Timedelta(hours=1):
            continue

        forecast_row = forecast_df.loc[nearest_idx]
        for col in WEATHER_COLUMNS:
            out_col = f"weather_forecast_{col}_t_plus_{horizon}h"
            if out_col in df.columns:
                df.loc[latest_idx, out_col] = forecast_row.get(col, np.nan)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily-average forecasting targets for next 3 days.

    Targets are computed as future-window means:
    - +24h target: mean(pm2_5[t+1 ... t+24])
    - +48h target: mean(pm2_5[t+25 ... t+48])
    - +72h target: mean(pm2_5[t+49 ... t+72])
    """

    windows = {
        "target_pm2_5_t_plus_24h": (1, 24),
        "target_pm2_5_t_plus_48h": (25, 48),
        "target_pm2_5_t_plus_72h": (49, 72),
    }

    for target_col, (start_h, end_h) in windows.items():
        shifted = [df["pm2_5"].shift(-hour) for hour in range(start_h, end_h + 1)]
        window_matrix = pd.concat(shifted, axis=1)
        df[target_col] = window_matrix.mean(axis=1, skipna=False)

    return df


def drop_low_signal_precipitation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop precipitation feature family when variance is near zero."""
    if "precipitation" not in df.columns:
        return df

    precip_std = float(df["precipitation"].std(skipna=True))
    if np.isnan(precip_std) or precip_std > PRECIP_STD_DROP_THRESHOLD:
        return df

    drop_cols = [
        col
        for col in df.columns
        if col == "precipitation"
        or col.startswith("precipitation_lag_")
        or col.startswith("weather_forecast_precipitation_")
    ]
    print(
        "Dropping low-signal precipitation features: "
        f"std={precip_std:.6f} threshold={PRECIP_STD_DROP_THRESHOLD:.6f} cols={len(drop_cols)}"
    )
    return df.drop(columns=drop_cols)


def save_features(df: pd.DataFrame, output_path: Path) -> None:
    """Save engineered features to local processed CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run full feature engineering pipeline for Karachi AQI."""
    df = load_historical_data(INPUT_PATH)
    historical_weather_df = load_historical_weather(HISTORICAL_WEATHER_PATH)
    df = merge_weather_on_timestamp(df, historical_weather_df)

    current_df = load_current_snapshot(CURRENT_INPUT_PATH)
    current_df = add_external_weather_to_current_snapshot(current_df, historical_weather_df)
    df = merge_historical_with_current(df, current_df)

    validate_input_quality(df)
    df = clean_time_series(df)
    df = add_time_features(df)
    df = add_derived_features(df)
    df = add_weather_forecast_covariates(df)

    weather_forecast_df = load_weather_forecast(WEATHER_FORECAST_PATH)
    df = inject_latest_weather_forecast(df, weather_forecast_df)

    df = add_targets(df)
    df = drop_low_signal_precipitation_features(df)

    feature_only_cols = [
        col
        for col in df.columns
        if col
        not in [
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
