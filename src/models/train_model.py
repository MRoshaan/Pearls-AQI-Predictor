"""Train AQI forecasting models using features from Hopsworks.

This script pulls engineered features from the Hopsworks Feature Store,
trains horizon-specific regressors for +24h/+48h/+72h PM2.5 forecasts,
evaluates model performance, saves local artifacts, and uploads the winning
model bundle to the Hopsworks Model Registry.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ARTIFACTS_ROOT = Path("artifacts/models")
LOCAL_FEATURES_PATH = Path("data/processed/karachi_features.csv")
TARGET_COLUMNS = [
    "target_pm2_5_t_plus_24h",
    "target_pm2_5_t_plus_48h",
    "target_pm2_5_t_plus_72h",
]
RIDGE_ALPHAS = [0.1, 1.0, 3.0, 10.0, 30.0]
LASSO_ALPHAS = [0.0005, 0.001, 0.003, 0.01, 0.03]
ELASTICNET_CONFIGS = [
    {"alpha": 0.001, "l1_ratio": 0.2},
    {"alpha": 0.003, "l1_ratio": 0.5},
    {"alpha": 0.01, "l1_ratio": 0.7},
]
VOTING_HGBR_CONFIGS = [
    {"max_iter": 180, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 20},
    {"max_iter": 260, "learning_rate": 0.05, "max_depth": 8, "min_samples_leaf": 20},
    {"max_iter": 220, "learning_rate": 0.08, "max_depth": 6, "min_samples_leaf": 30},
]


def ensure_windows_hopsworks_tmp(host: str) -> None:
    """Create temp directories expected by Hopsworks client on Windows."""
    if os.name != "nt":
        return

    tmp_root = Path("/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / host).mkdir(parents=True, exist_ok=True)


def _as_int_env(raw_value: str | None, default: int) -> int:
    """Parse integer env value with safe fallback."""
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if stripped == "":
        return default

    return int(stripped)


def _as_float_env(raw_value: str | None, default: float) -> float:
    """Parse float env value with safe fallback."""
    if raw_value is None:
        return default

    stripped = raw_value.strip()
    if stripped == "":
        return default

    return float(stripped)


def load_environment() -> dict[str, Any]:
    """Load required configuration from environment variables."""
    load_dotenv()

    config = {
        "use_hopsworks": os.getenv("USE_HOPSWORKS", "false").lower() == "true",
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", "pearls_aqi_predictor"),
        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        "hopsworks_port": _as_int_env(os.getenv("HOPSWORKS_PORT"), 443),
        "feature_group": os.getenv("HOPSWORKS_FEATURE_GROUP", "karachi_aqi_features"),
        "feature_group_version": _as_int_env(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION"), 3),
        "model_name": os.getenv("HOPSWORKS_MODEL_NAME", "karachi_aqi_forecaster"),
        "train_test_split_ratio": _as_float_env(os.getenv("TRAIN_SPLIT_RATIO"), 0.8),
        "random_state": _as_int_env(os.getenv("RANDOM_STATE"), 42),
        "upload_to_registry": os.getenv("UPLOAD_TO_HOPSWORKS", "false").lower() == "true",
        "max_feature_age_hours": _as_float_env(os.getenv("MAX_FEATURE_AGE_HOURS"), 48.0),
        "enforce_fresh_features": os.getenv("ENFORCE_FRESH_FEATURES", "false").lower() == "true",
    }

    if config["use_hopsworks"] and not config["hopsworks_api_key"]:
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

    ensure_windows_hopsworks_tmp(host)

    try:
        return hopsworks.login(
            project=project_name,
            host=host,
            port=port,
            api_key_value=api_key,
        )
    except Exception as exc:
        print(
            "Host-specific login failed. Retrying with default Hopsworks endpoint: "
            f"{exc}"
        )
        return hopsworks.login(project=project_name, api_key_value=api_key)


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

    if "timestamp" not in df.columns:
        raise ValueError("Feature group dataframe missing timestamp column.")

    return df


def fetch_local_features_dataframe(path: Path = LOCAL_FEATURES_PATH) -> pd.DataFrame:
    """Fetch engineered features from local CSV for offline/local-first training."""
    if not path.exists():
        raise FileNotFoundError(f"Local features file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Local features file is empty.")

    if "timestamp" not in df.columns:
        raise ValueError("Local features dataframe missing timestamp column.")

    return df


def validate_feature_freshness(df: pd.DataFrame, config: dict[str, Any]) -> None:
    """Optionally block training on stale feature data."""
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    latest = ts.max()
    if pd.isna(latest):
        raise ValueError("Cannot determine latest timestamp for freshness validation.")

    age_hours = (datetime.now(timezone.utc) - latest.to_pydatetime()).total_seconds() / 3600
    if config.get("enforce_fresh_features", False) and age_hours > float(config["max_feature_age_hours"]):
        raise ValueError(
            "Training blocked due to stale feature data: "
            f"age={age_hours:.2f}h, limit={float(config['max_feature_age_hours']):.2f}h"
        )

    print(
        "Feature freshness check: "
        f"latest={latest.isoformat()}, age_hours={age_hours:.2f}, "
        f"enforced={config.get('enforce_fresh_features', False)}"
    )


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
    data = data.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

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

    if "precipitation" in X.columns:
        precip_std = float(X["precipitation"].std(skipna=True))
        if np.isnan(precip_std) or precip_std <= 1e-3:
            drop_cols = [
                col
                for col in X.columns
                if col == "precipitation"
                or col.startswith("precipitation_lag_")
                or col.startswith("weather_forecast_precipitation_")
            ]
            if drop_cols:
                print(
                    "Dropping low-signal precipitation feature family during training: "
                    f"std={precip_std:.6f}, cols={len(drop_cols)}"
                )
                X = X.drop(columns=drop_cols)

    split_index = int(len(X) * split_ratio)
    if split_index <= 0 or split_index >= len(X):
        raise ValueError("Invalid train/test split. Check data size and split ratio.")

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def split_train_validation(
    X_data: pd.DataFrame,
    y_data: pd.DataFrame,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train/validation split from training data."""
    split_idx = int(len(X_data) * (1 - validation_ratio))
    if split_idx <= 0 or split_idx >= len(X_data):
        raise ValueError("Invalid train/validation split while tuning models.")

    X_fit = X_data.iloc[:split_idx]
    X_val = X_data.iloc[split_idx:]
    y_fit = y_data.iloc[:split_idx]
    y_val = y_data.iloc[split_idx:]
    return X_fit, X_val, y_fit, y_val


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


def _build_ridge(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _build_lasso(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=alpha, max_iter=10000)),
        ]
    )


def _build_elasticnet(alpha: float, l1_ratio: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000),
            ),
        ]
    )


def _build_hist_gradient_boosting(config: dict[str, Any], random_state: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        random_state=random_state,
    )


def _build_voting_regressor(config: dict[str, Any], random_state: int) -> VotingRegressor:
    """Build robust ensemble combining boosted trees and random forest."""
    hgbr = _build_hist_gradient_boosting(config, random_state)
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=20,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return VotingRegressor(
        estimators=[
            ("hgbr", hgbr),
            ("rf", rf),
        ]
    )


def _fit_predict_single_horizon(
    model: Any,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    model.fit(X_fit, y_fit)
    pred = model.predict(X_eval)
    return np.asarray(pred, dtype=float)


def _rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def tune_and_train_horizon_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    """Tune model family per horizon and retrain on full train set."""
    fast_train = os.getenv("FAST_TRAIN", "false").lower() == "true"
    X_fit, X_val, y_fit, y_val = split_train_validation(X_train, y_train.to_frame())
    y_fit_series = y_fit.iloc[:, 0]
    y_val_series = y_val.iloc[:, 0]

    hgbr_candidates = VOTING_HGBR_CONFIGS[:1] if fast_train else VOTING_HGBR_CONFIGS
    candidates: list[tuple[str, Any]] = [
        (
            (
                f"voting_hgbr_iter={cfg['max_iter']}_lr={cfg['learning_rate']}_"
                f"d={cfg['max_depth']}_leaf={cfg['min_samples_leaf']}"
            ),
            _build_voting_regressor(cfg, random_state),
        )
        for cfg in hgbr_candidates
    ]

    best_label = ""
    best_model: Any | None = None
    best_val_rmse = float("inf")

    for label, model in candidates:
        val_pred = _fit_predict_single_horizon(model, X_fit, y_fit_series, X_val)
        val_rmse = _rmse(y_val_series, val_pred)
        if val_rmse < best_val_rmse:
            best_label = label
            best_model = model
            best_val_rmse = val_rmse

    if best_model is None:
        raise RuntimeError("No horizon model selected during tuning.")

    best_model.fit(X_train, y_train)
    test_pred = np.asarray(best_model.predict(X_test), dtype=float)

    return {
        "model": best_model,
        "validation": {"best_config": best_label, "rmse": best_val_rmse},
        "test_predictions": test_pred,
    }


def train_horizon_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    random_state: int,
) -> dict[str, Any]:
    """Train and tune horizon-specific models, then aggregate metrics."""
    horizon_models: dict[str, Any] = {}
    stacked_predictions: list[np.ndarray] = []

    for target_col in TARGET_COLUMNS:
        y_target = cast(pd.Series, y_train[target_col])
        trained = tune_and_train_horizon_model(
            X_train=X_train,
            y_train=y_target,
            X_test=X_test,
            random_state=random_state,
        )
        horizon_models[target_col] = {
            "model": trained["model"],
            "validation": trained["validation"],
        }
        stacked_predictions.append(trained["test_predictions"])

    y_pred_matrix = np.column_stack(stacked_predictions)
    metrics = evaluate_predictions(y_test, y_pred_matrix, TARGET_COLUMNS)

    return {
        "model": horizon_models,
        "metrics": metrics,
        "framework": "scikit-learn",
        "model_type": "horizon_specific",
    }


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
    return {
        "model": model,
        "metrics": metrics,
        "framework": "tensorflow",
        "model_type": "multi_output",
    }


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

    model_type = model_bundle.get("model_type", "multi_output")
    if model_bundle.get("framework") == "tensorflow":
        model_bundle["model"].save(model_dir / "tf_model")
    elif model_type == "horizon_specific":
        horizon_models = model_bundle["model"]
        for target, horizon_bundle in horizon_models.items():
            horizon_file = model_dir / f"{target}.joblib"
            joblib.dump(horizon_bundle["model"], horizon_file)
    else:
        joblib.dump(model_bundle["model"], model_dir / "model.joblib")

    metadata = {
        "model_name": model_name,
        "framework": model_bundle.get("framework", "scikit-learn"),
        "model_type": model_type,
        "feature_columns": feature_columns,
        "target_columns": TARGET_COLUMNS,
        "metrics": model_bundle["metrics"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if model_type == "horizon_specific":
        metadata["horizon_models"] = {
            target: {
                "artifact": f"{target}.joblib",
                "validation": bundle.get("validation", {}),
            }
            for target, bundle in model_bundle["model"].items()
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
    project = None
    if config["use_hopsworks"]:
        try:
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
            print("Loaded training data from Hopsworks Feature Store.")
        except Exception as exc:
            print(f"Hopsworks read failed, falling back to local features: {exc}")
            features_df = fetch_local_features_dataframe()
            print(f"Loaded training data from local CSV: {LOCAL_FEATURES_PATH}")
    else:
        features_df = fetch_local_features_dataframe()
        print(f"Loaded training data from local CSV: {LOCAL_FEATURES_PATH}")
    validate_feature_freshness(features_df, config)

    X_train, X_test, y_train, y_test = prepare_training_data(
        df=features_df,
        target_cols=TARGET_COLUMNS,
        split_ratio=config["train_test_split_ratio"],
    )

    results: dict[str, dict[str, Any]] = {
        "horizon_specific_ensemble": train_horizon_models(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            random_state=config["random_state"],
        )
    }

    dl_result = maybe_train_deep_learning_model(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=config["random_state"],
    )
    if dl_result is not None:
        results["tensorflow_mlp"] = dl_result

    best_model_name, best_bundle = select_best_model(results)

    print("Model evaluation summary:")
    for name, bundle in results.items():
        overall = bundle["metrics"]["overall"]
        print(
            f"- {name}: RMSE={overall['rmse']:.4f}, "
            f"MAE={overall['mae']:.4f}, R2={overall['r2']:.4f}"
        )

        if bundle.get("model_type") == "horizon_specific":
            for target, target_bundle in bundle["model"].items():
                val = target_bundle.get("validation", {})
                if val:
                    print(
                        f"  * {target}: val_rmse={val['rmse']:.4f} "
                        f"({val['best_config']})"
                    )

    print(f"Selected best model: {best_model_name}")
    model_dir = save_local_artifacts(best_model_name, best_bundle, list(X_train.columns))
    print(f"Saved local model artifacts to {model_dir}")

    if config["upload_to_registry"] and project is not None:
        upload_to_hopsworks_registry(
            project=project,
            model_name=config["model_name"],
            model_dir=model_dir,
            metrics=best_bundle["metrics"],
        )


if __name__ == "__main__":
    main()
