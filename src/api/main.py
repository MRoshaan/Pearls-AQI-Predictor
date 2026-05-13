"""FastAPI serving layer for AQI forecast inference."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


LOCAL_ARTIFACTS_DIR = Path("artifacts/models")


@dataclass
class RuntimeState:
    """In-memory serving state to avoid repeated cold loads."""

    model: Any | None = None
    feature_columns: list[str] | None = None
    model_source: str | None = None


state = RuntimeState()


class PredictionResponse(BaseModel):
    """Prediction payload returned by the API."""

    model_config = {"protected_namespaces": ()}

    model_source: str
    city: str
    generated_from_timestamp: str
    pm2_5_forecast: dict[str, float]
    hazardous_alert: bool


def ensure_windows_hopsworks_tmp(host: str) -> None:
    """Create temp directories expected by Hopsworks client on Windows."""
    if os.name != "nt":
        return

    tmp_root = Path("/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / host).mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load runtime configuration from environment."""
    load_dotenv()
    return {
        "hopsworks_api_key": os.getenv("HOPSWORKS_API_KEY"),
        "hopsworks_project": os.getenv("HOPSWORKS_PROJECT", "pearls_aqi_predictor"),
        "hopsworks_host": os.getenv("HOPSWORKS_HOST", "eu-west.cloud.hopsworks.ai"),
        "hopsworks_port": int(os.getenv("HOPSWORKS_PORT", "443")),
        "feature_group": os.getenv("HOPSWORKS_FEATURE_GROUP", "karachi_aqi_features"),
        "feature_group_version": int(os.getenv("HOPSWORKS_FEATURE_GROUP_VERSION", "1")),
        "model_name": os.getenv("HOPSWORKS_MODEL_NAME", "karachi_aqi_forecaster"),
        "model_version": os.getenv("HOPSWORKS_MODEL_VERSION"),
        "default_city": os.getenv("DEFAULT_CITY", "Karachi"),
    }


def connect_hopsworks(config: dict[str, Any]) -> Any:
    """Connect to Hopsworks and return project handle."""
    try:
        import hopsworks
    except ImportError as exc:
        raise RuntimeError("hopsworks package is required for online inference.") from exc

    api_key = config["hopsworks_api_key"]
    if not api_key:
        raise RuntimeError("Missing HOPSWORKS_API_KEY in environment.")

    ensure_windows_hopsworks_tmp(config["hopsworks_host"])

    try:
        return hopsworks.login(
            project=config["hopsworks_project"],
            host=config["hopsworks_host"],
            port=config["hopsworks_port"],
            api_key_value=api_key,
        )
    except Exception:
        return hopsworks.login(
            project=config["hopsworks_project"],
            api_key_value=api_key,
        )


def _download_model_from_registry(project: Any, config: dict[str, Any]) -> tuple[Any, list[str], str]:
    """Try downloading latest model artifact from Hopsworks Model Registry."""
    model_registry = project.get_model_registry()
    model_name = config["model_name"]
    model_version = config["model_version"]

    model_obj: Any | None = None
    if model_version:
        model_obj = model_registry.get_model(name=model_name, version=int(model_version))
    else:
        try:
            candidates = model_registry.get_models(name=model_name)
            if candidates:
                model_obj = max(candidates, key=lambda item: getattr(item, "version", 0))
        except Exception:
            model_obj = model_registry.get_model(name=model_name)

    if model_obj is None:
        raise RuntimeError("No model found in registry.")

    model_dir = Path(model_obj.download())
    model_file = model_dir / "model.joblib"
    metadata_file = model_dir / "metadata.json"

    if not model_file.exists():
        fallback_files = sorted(model_dir.glob("**/model.joblib"))
        if not fallback_files:
            raise RuntimeError("Downloaded registry artifact has no model.joblib file.")
        model_file = fallback_files[-1]

    metadata: dict[str, Any] = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    model = joblib.load(model_file)
    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise RuntimeError("metadata.json missing feature_columns in registry artifact.")

    source = f"hopsworks_registry:{model_name}"
    if getattr(model_obj, "version", None) is not None:
        source += f":v{model_obj.version}"

    return model, feature_columns, source


def _load_latest_local_model() -> tuple[Any, list[str], str]:
    """Fallback: load latest locally saved model artifact."""
    if not LOCAL_ARTIFACTS_DIR.exists():
        raise RuntimeError("No local artifacts directory found.")

    model_dirs = [path for path in LOCAL_ARTIFACTS_DIR.glob("*") if path.is_dir()]
    if not model_dirs:
        raise RuntimeError("No local model directories found.")

    latest_dir = max(model_dirs, key=lambda item: item.stat().st_mtime)
    model_file = latest_dir / "model.joblib"
    metadata_file = latest_dir / "metadata.json"

    if not model_file.exists() or not metadata_file.exists():
        raise RuntimeError(f"Local model artifact incomplete at {latest_dir}")

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    feature_columns_raw = metadata.get("feature_columns")
    if isinstance(feature_columns_raw, dict):
        feature_columns = list(feature_columns_raw.values())
    elif isinstance(feature_columns_raw, list):
        feature_columns = feature_columns_raw
    else:
        raise RuntimeError("metadata.json missing feature_columns in local artifact.")

    return joblib.load(model_file), feature_columns, f"local_artifact:{latest_dir.name}"


def load_serving_model() -> tuple[Any, list[str], str]:
    """Load model and feature schema from registry, fallback to local artifact."""
    config = load_config()

    try:
        project = connect_hopsworks(config)
        return _download_model_from_registry(project, config)
    except Exception:
        return _load_latest_local_model()


def fetch_latest_feature_row(project: Any, config: dict[str, Any]) -> pd.DataFrame:
    """Fetch latest row from feature group and return as dataframe."""
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(
        name=config["feature_group"],
        version=config["feature_group_version"],
    )
    df = feature_group.read()
    if df.empty:
        raise RuntimeError("Feature group returned zero rows.")

    if "timestamp" not in df.columns:
        raise RuntimeError("Feature group missing timestamp column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if df.empty:
        raise RuntimeError("Feature group has no valid timestamp rows.")

    return df.tail(1)


def get_or_load_model() -> tuple[Any, list[str], str]:
    """Return cached model, loading once per process."""
    if state.model is None or state.feature_columns is None or state.model_source is None:
        state.model, state.feature_columns, state.model_source = load_serving_model()

    return state.model, state.feature_columns, state.model_source


def make_prediction() -> PredictionResponse:
    """Compute latest +24h/+48h/+72h PM2.5 forecast."""
    config = load_config()
    project = connect_hopsworks(config)
    latest = fetch_latest_feature_row(project, config)

    model, feature_columns, model_source = get_or_load_model()

    missing_cols = [col for col in feature_columns if col not in latest.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required features for prediction: {missing_cols}")

    X = latest[feature_columns].copy()
    y_pred = model.predict(X)
    if len(y_pred.shape) != 2 or y_pred.shape[1] < 3:
        raise RuntimeError("Model output shape is invalid for 3-horizon prediction.")

    forecast = {
        "+24h": float(y_pred[0][0]),
        "+48h": float(y_pred[0][1]),
        "+72h": float(y_pred[0][2]),
    }
    hazardous = any(value >= 250.5 for value in forecast.values())

    return PredictionResponse(
        model_source=model_source,
        city=config["default_city"],
        generated_from_timestamp=str(latest.iloc[0]["timestamp"]),
        pm2_5_forecast=forecast,
        hazardous_alert=hazardous,
    )


app = FastAPI(title="Karachi AQI Forecast API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}


@app.get("/predict/latest", response_model=PredictionResponse)
def predict_latest() -> PredictionResponse:
    """Run one-shot forecast from latest feature row."""
    try:
        return make_prediction()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
