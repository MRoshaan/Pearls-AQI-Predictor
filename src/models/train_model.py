"""Train AQI forecasting models using features from Hopsworks.

This script pulls engineered features from the Hopsworks Feature Store,
trains baseline multi-output regressors for +24h/+48h/+72h PM2.5 forecasts,
evaluates model performance, saves local artifacts, and uploads the winning
model to the Hopsworks Model Registry.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACTS_ROOT = Path("artifacts/models")
TARGET_COLUMNS = [
    "target_pm2_5_t_plus_24h",
    "target_pm2_5_t_plus_48h",
    "target_pm2_5_t_plus_72h",
]


def load_environment() -> dict[str, Any]:
    """Load required configuration from environment variables."""
    load_dotenv()

    config = {
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", "pearls_aqi_predictor"),
        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "run.hopsworks.ai"),
        "hopsworks_port": int(os.getenv("HOPSWORKS_PORT", "443")),
        "feature_group": os.getenv("HOPSWORKS_FEATURE_GROUP", "karachi_aqi_features"),
        "feature_group_version": int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1")),
        "model_name": os.getenv("HOPSWORKS_MODEL_NAME", "karachi_aqi_forecaster"),
        "train_test_split_ratio": float(os.getenv("TRAIN_SPLIT_RATIO", "0.8")),
        "random_state": int(os.getenv("RANDOM_STATE", "42")),
        "upload_to_registry": os.getenv("UPLOAD_TO_HOPSWORKS", "true").lower()
        == "true",
    }

    if not config["hopsworks_api_key"]:
        raise ValueError("Missing HOPSWORKS_API_KEY in .env.")

    return config


def connect_hopsworks(
    project_name: str,
    api_key: str,
    host: str,
    port: int,
) -> Any:
    """Connect to Hopsworks project."""
    try:
        import hopsworks
    except ImportError as exc:
        raise ImportError(
            "hopsworks package is not installed. Install dependencies first."
        ) from exc

    return hopsworks.login(
        project=project_name,
        host=host,
        port=port,
        api_key_value=api_key,
    )


def fetch_feature_group_dataframe(
    project: Any,
    feature_group_name: str,
    feature_group_version: int,
) -> pd.DataFrame:
    """Fetch historical features/targets from Hopsworks Feature Store."""
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(
        name=feature_group_name,
        version=feature_group_version,
    )

    df = feature_group.read()
    if df.empty:
        raise ValueError("Feature group returned no rows.")

    return df


def prepare_training_data(
    df: pd.DataFrame,
    target_cols: list[str],
    split_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare chronological train/test split for time-series forecasting."""
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns in feature data: {missing_targets}")

    if "timestamp" not in df.columns:
        raise ValueError("Feature dataframe must contain 'timestamp' column.")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(
        drop=True
    )

    drop_cols = ["id", "timestamp", *target_cols]
    feature_cols = [col for col in data.columns if col not in drop_cols]

    X = data[feature_cols].copy()
    y = data[target_cols].copy()

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        X = X.drop(columns=non_numeric_cols)

    valid_rows = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    split_index = int(len(X) * split_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Invalid train/test split. Check data size and split ratio.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def evaluate_predictions(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    target_cols: list[str],
) -> dict[str, Any]:
    """Compute RMSE, MAE, and R2 for each horizon and overall averages."""
    metrics: dict[str, Any] = {"per_target": {}}

    rmse_values: list[float] = []
    mae_values: list[float] = []
    r2_values: list[float] = []

    for idx, target in enumerate(target_cols):
        true_values = y_true.iloc[:, idx]
        pred_values = y_pred[:, idx]

        rmse = float(np.sqrt(mean_squared_error(true_values, pred_values)))
        mae = float(mean_absolute_error(true_values, pred_values))
        r2 = float(r2_score(true_values, pred_values))

        metrics["per_target"][target] = {"rmse": rmse, "mae": mae, "r2": r2}
        rmse_values.append(rmse)
        mae_values.append(mae)
        r2_values.append(r2)

    metrics["overall"] = {
        "rmse": float(np.mean(rmse_values)),
        "mae": float(np.mean(mae_values)),
        "r2": float(np.mean(r2_values)),
    }
    return metrics


def train_baseline_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    random_state: int,
) -> dict[str, dict[str, Any]]:
    """Train baseline models and return models with metrics."""
    models: dict[str, Any] = {
        "ridge": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", MultiOutputRegressor(Ridge(alpha=1.0))),
            ]
        ),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }

    results: dict[str, dict[str, Any]] = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions, TARGET_COLUMNS)
        results[model_name] = {"model": model, "metrics": metrics}

    return results


def maybe_train_deep_learning_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    random_state: int,
) -> dict[str, Any] | None:
    """Optionally train a TensorFlow MLP model if TensorFlow is installed."""
    if os.getenv("ENABLE_TENSORFLOW", "false").lower() != "true":
        return None

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed. Skipping deep learning model.")
        return None

    tf.keras.utils.set_random_seed(random_state)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(y_train.shape[1]),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_train.values,
        y_train.values,
        validation_split=0.1,
        epochs=int(os.getenv("TF_EPOCHS", "25")),
        batch_size=int(os.getenv("TF_BATCH_SIZE", "256")),
        verbose=0,
    )

    predictions = model.predict(X_test.values, verbose=0)
    metrics = evaluate_predictions(y_test, predictions, TARGET_COLUMNS)
    return {"model": model, "metrics": metrics, "framework": "tensorflow"}


def select_best_model(results: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Select best model using lowest overall RMSE."""
    best_name = min(
        results,
        key=lambda name: results[name]["metrics"]["overall"]["rmse"],
    )
    return best_name, results[best_name]


def save_local_artifacts(
    model_name: str,
    model_bundle: dict[str, Any],
    feature_columns: list[str],
) -> Path:
    """Persist model and metadata locally for traceability."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = ARTIFACTS_ROOT / f"{model_name}_{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_bundle.get("framework") == "tensorflow":
        model_bundle["model"].save(model_dir / "tf_model")
    else:
        joblib.dump(model_bundle["model"], model_dir / "model.joblib")

    metadata = {
        "model_name": model_name,
        "framework": model_bundle.get("framework", "scikit-learn"),
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "metrics": model_bundle["metrics"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with (model_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return model_dir


def upload_to_hopsworks_registry(
    project: Any,
    model_name: str,
    model_dir: Path,
    metrics: dict[str, Any],
) -> None:
    """Upload winning model artifacts to Hopsworks Model Registry."""
    model_registry = project.get_model_registry()
    flattened_metrics = {
        "rmse": float(metrics["overall"]["rmse"]),
        "mae": float(metrics["overall"]["mae"]),
        "r2": float(metrics["overall"]["r2"]),
    }

    if hasattr(model_registry, "python"):
        model = model_registry.python.create_model(
            name=model_name,
            description="Karachi AQI +24h/+48h/+72h PM2.5 forecaster",
            metrics=flattened_metrics,
        )
    elif hasattr(model_registry, "sklearn"):
        model = model_registry.sklearn.create_model(
            name=model_name,
            description="Karachi AQI +24h/+48h/+72h PM2.5 forecaster",
            metrics=flattened_metrics,
        )
    else:
        raise RuntimeError("Unsupported Hopsworks Model Registry client API.")

    model.save(str(model_dir))
    version = getattr(model, "version", "unknown")
    print(f"Uploaded model '{model_name}' to Hopsworks Model Registry (v{version}).")


def main() -> None:
    """Run complete model training and registry workflow."""
    config = load_environment()
    project = connect_hopsworks(
        project_name=config["hopsworks_project"],
        api_key=config["hopsworks_api_key"],
        host=config["hopsworks_host"],
        port=config["hopsworks_port"],
    )

    features_df = fetch_feature_group_dataframe(
        project=project,
        feature_group_name=config["feature_group"],
        feature_group_version=config["feature_group_version"],
    )

    X_train, X_test, y_train, y_test = prepare_training_data(
        df=features_df,
        target_cols=TARGET_COLUMNS,
        split_ratio=config["train_test_split_ratio"],
    )

    baseline_results = train_baseline_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=config["random_state"],
    )

    dl_result = maybe_train_deep_learning_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=config["random_state"],
    )
    if dl_result is not None:
        baseline_results["tensorflow_mlp"] = dl_result

    best_model_name, best_bundle = select_best_model(baseline_results)

    print("Model evaluation summary:")
    for name, bundle in baseline_results.items():
        overall = bundle["metrics"]["overall"]
        print(
            f"- {name}: RMSE={overall['rmse']:.4f}, "
            f"MAE={overall['mae']:.4f}, R2={overall['r2']:.4f}"
        )

    print(f"Selected best model: {best_model_name}")
    model_dir = save_local_artifacts(best_model_name, best_bundle, list(X_train.columns))
    print(f"Saved local model artifacts to {model_dir}")

    if config["upload_to_registry"]:
        upload_to_hopsworks_registry(
            project=project,
            model_name=config["model_name"],
            model_dir=model_dir,
            metrics=best_bundle["metrics"],
        )


if __name__ == "__main__":
    main()
