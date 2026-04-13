"""Exploratory Data Analysis for Karachi AQI dataset.

This script reads the ingested raw CSV, handles missing values,
prints summary statistics, and generates core visualizations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("data/raw/karachi_aqi_raw.csv")
OUTPUT_DIR = Path("data/processed/eda")


def load_data(file_path: Path) -> pd.DataFrame:
    """Load dataset from CSV with timestamp parsing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input data file not found at {file_path}. Run ingestion first."
        )

    df = pd.read_csv(file_path, parse_dates=["timestamp"])
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Inspect and handle missing values with simple median imputation."""
    print("\nMissing values before handling:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.median()))

    print("\nMissing values after handling:")
    print(df.isna().sum())

    return df


def print_summary_statistics(df: pd.DataFrame) -> None:
    """Print high-level summary statistics."""
    print("\nDataset info:")
    print(df.info())

    print("\nSummary statistics:")
    print(df.describe(include="all"))


def plot_aqi_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate AQI trend over time chart."""
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df.sort_values("timestamp"), x="timestamp", y="aqi")
    plt.title("Karachi AQI Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("AQI (OpenWeather Scale)")
    plt.tight_layout()
    output_file = output_dir / "aqi_over_time.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved AQI trend plot to: {output_file}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate correlation matrix between weather and pollutant variables."""
    cols_for_corr = [
        "aqi",
        "co",
        "no",
        "no2",
        "o3",
        "so2",
        "pm2_5",
        "pm10",
        "nh3",
        "temp_c",
        "feels_like_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_mps",
        "clouds_pct",
    ]

    available_cols = [col for col in cols_for_corr if col in df.columns]
    corr_df = df[available_cols].corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Matrix: Weather vs Pollutants (Karachi)")
    plt.tight_layout()
    output_file = output_dir / "correlation_matrix.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved correlation matrix plot to: {output_file}")


def main() -> None:
    """Run EDA workflow for Karachi AQI raw dataset."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_PATH)
    df = handle_missing_values(df)
    print_summary_statistics(df)

    plot_aqi_over_time(df, OUTPUT_DIR)
    plot_correlation_matrix(df, OUTPUT_DIR)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
