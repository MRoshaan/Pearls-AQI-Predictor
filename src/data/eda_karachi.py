"""EDA workflow for Karachi historical air quality dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("data/raw/karachi_historical_aqi.csv")
FIGURES_DIR = Path("reports/figures")


def load_data(file_path: Path) -> pd.DataFrame:
    """Load historical AQI data and set datetime index."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found at {file_path}. Run historical ingestion first."
        )

    df = pd.read_csv(file_path)

    time_col = "time" if "time" in df.columns else "timestamp"
    if time_col not in df.columns:
        raise ValueError("Expected a 'time' or 'timestamp' column in input CSV.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col)
    df = df.set_index(time_col)

    return df


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values using time-series friendly strategy."""
    missing_before = df.isna().sum()
    print("\nMissing values before cleaning:")
    print(missing_before[missing_before > 0] if (missing_before > 0).any() else "None")

    df_clean = df.ffill().bfill()

    missing_after = df_clean.isna().sum()
    print("\nMissing values after cleaning:")
    print(missing_after[missing_after > 0] if (missing_after > 0).any() else "None")

    return df_clean


def print_statistical_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics using pandas and numpy."""
    print("\nDescriptive statistics (.describe):")
    print(df.describe(include="all"))

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        print("\nNumpy means by pollutant:")
        mean_values = np.nanmean(numeric_df.to_numpy(), axis=0)
        print(pd.Series(mean_values, index=numeric_df.columns))


def plot_pm_trends(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot PM2.5 and PM10 trends across full timeframe."""
    required_cols = ["pm2_5", "pm10"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for PM trend plot: {missing_cols}")

    plt.figure(figsize=(14, 6))
    sns.lineplot(data=df[required_cols])
    plt.title("Karachi PM2.5 and PM10 Trend (Hourly)")
    plt.xlabel("Time")
    plt.ylabel("Concentration")
    plt.legend(["PM2.5", "PM10"])
    plt.tight_layout()

    output_file = output_dir / "karachi_pm25_pm10_trend.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved PM trend plot to {output_file}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot correlation heatmap for pollutant columns."""
    pollutant_cols = [
        "pm2_5",
        "pm10",
        "carbon_monoxide",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "ozone",
    ]
    available_cols = [col for col in pollutant_cols if col in df.columns]

    if len(available_cols) < 2:
        raise ValueError("Not enough pollutant columns available for correlation plot.")

    corr_df = df[available_cols].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Karachi Pollutant Correlation Heatmap")
    plt.tight_layout()

    output_file = output_dir / "karachi_pollutant_correlation_heatmap.png"
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved correlation heatmap to {output_file}")


def main() -> None:
    """Run EDA steps and generate saved figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(INPUT_PATH)
    df = clean_missing_values(df)
    print_statistical_summary(df)

    plot_pm_trends(df, FIGURES_DIR)
    plot_correlation_heatmap(df, FIGURES_DIR)


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
