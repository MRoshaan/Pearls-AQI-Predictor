"""FastAPI serving layer for AQI forecast inference and explainability."""

from __future__ import annotations

import json
import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"
LOCAL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/models"
LOCAL_FEATURES_PATH = PROJECT_ROOT / "data/processed/karachi_features.csv"


@dataclass
class RuntimeState:
    """In-memory serving state to avoid repeated cold loads."""

    model: Any | None = None
    feature_columns: list[str] | None = None
    model_source: str | None = None
    model_type: str | None = None
    target_columns: list[str] | None = None
    feature_df: pd.DataFrame | None = None
    feature_source: str | None = None
    feature_fetched_at: float | None = None


state = RuntimeState()


class PredictionResponse(BaseModel):
    """Prediction payload returned by the API."""

    model_config = ConfigDict(protected_namespaces=())

    model_source: str
    feature_source: str
    city: str
    generated_from_timestamp: str
    prediction_unit: str
    pm2_5_forecast: dict[str, float]
    hazardous_alert: bool


class FeatureExplanation(BaseModel):
    """One feature contribution score from LIME."""

    feature: str
    explanation_score: float
    abs_explanation_score: float


class ExplainResponse(BaseModel):
    """Explainability payload for latest prediction."""

    model_config = ConfigDict(protected_namespaces=())

    model_source: str
    feature_source: str
    city: str
    generated_from_timestamp: str
    horizon: str
    top_features: list[FeatureExplanation]


def ensure_windows_hopsworks_tmp(host: str) -> None:
    """Create temp directories expected by Hopsworks client on Windows."""
    if os.name != "nt":
        return

    tmp_root = Path("/tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)
    (tmp_root / host).mkdir(parents=True, exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load runtime configuration from environment."""
    load_dotenv(dotenv_path=ENV_PATH)
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
        "hazardous_pm25_threshold": float(os.getenv("HAZARDOUS_PM25_THRESHOLD", "250.5")),
        "feature_cache_ttl_seconds": int(os.getenv("FEATURE_CACHE_TTL_SECONDS", "120")),
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


def _download_model_from_registry(
    project: Any,
    config: dict[str, Any],
) -> tuple[Any, list[str], str, str, list[str]]:
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

    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise RuntimeError("metadata.json missing feature_columns in registry artifact.")

    model_type = metadata.get("model_type", "multi_output")
    target_columns = metadata.get(
        "target_columns",
        [
            "target_pm2_5_t_plus_24h",
            "target_pm2_5_t_plus_48h",
            "target_pm2_5_t_plus_72h",
        ],
    )

    if model_type == "horizon_specific":
        horizon_models: dict[str, Any] = {}
        for target in target_columns:
            target_file = model_dir / f"{target}.joblib"
            if not target_file.exists():
                raise RuntimeError(f"Missing horizon model artifact in registry: {target_file.name}")
            horizon_models[target] = joblib.load(target_file)
        model: Any = horizon_models
    else:
        model = joblib.load(model_file)

    source = f"hopsworks_registry:{model_name}"
    if getattr(model_obj, "version", None) is not None:
        source += f":v{model_obj.version}"

    return model, feature_columns, source, model_type, target_columns


def _load_latest_local_model() -> tuple[Any, list[str], str, str, list[str]]:
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
    if not isinstance(feature_columns_raw, list):
        raise RuntimeError("metadata.json missing feature_columns in local artifact.")

    model_type = metadata.get("model_type", "multi_output")
    target_columns = metadata.get(
        "target_columns",
        [
            "target_pm2_5_t_plus_24h",
            "target_pm2_5_t_plus_48h",
            "target_pm2_5_t_plus_72h",
        ],
    )

    if model_type == "horizon_specific":
        horizon_models: dict[str, Any] = {}
        for target in target_columns:
            target_file = latest_dir / f"{target}.joblib"
            if not target_file.exists():
                raise RuntimeError(f"Missing horizon model artifact in local artifact: {target_file.name}")
            horizon_models[target] = joblib.load(target_file)
        model: Any = horizon_models
    else:
        model = joblib.load(model_file)

    return model, feature_columns_raw, f"local_artifact:{latest_dir.name}", model_type, target_columns


def load_serving_model() -> tuple[Any, list[str], str, str, list[str]]:
    """Load model and feature schema from registry, fallback to local artifact."""
    config = load_config()

    try:
        project = connect_hopsworks(config)
        return _download_model_from_registry(project, config)
    except Exception:
        return _load_latest_local_model()


def fetch_feature_data(project: Any, config: dict[str, Any]) -> pd.DataFrame:
    """Fetch feature group dataframe sorted by timestamp."""
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
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Feature group has no valid timestamp rows.")

    return df


def load_local_feature_data() -> pd.DataFrame:
    """Load locally engineered features as resilience fallback."""
    if not LOCAL_FEATURES_PATH.exists():
        raise RuntimeError(f"Local features file not found: {LOCAL_FEATURES_PATH}")

    df = pd.read_csv(LOCAL_FEATURES_PATH)
    if "timestamp" not in df.columns:
        raise RuntimeError("Local features file missing timestamp column.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Local features file has no valid timestamp rows.")

    return df


def fetch_feature_data_with_fallback(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Try Hopsworks feature data first, fallback to local features CSV."""
    try:
        project = connect_hopsworks(config)
        return fetch_feature_data(project, config), "hopsworks_feature_store"
    except Exception:
        return load_local_feature_data(), "local_features_csv"


def get_cached_feature_data(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Return cached feature dataframe if fresh, otherwise fetch and cache."""
    now = time.time()
    ttl = max(0, int(config.get("feature_cache_ttl_seconds", 120)))

    if (
        state.feature_df is not None
        and state.feature_source is not None
        and state.feature_fetched_at is not None
        and (now - state.feature_fetched_at) <= ttl
    ):
        return state.feature_df, state.feature_source

    feature_df, feature_source = fetch_feature_data_with_fallback(config)
    state.feature_df = feature_df
    state.feature_source = feature_source
    state.feature_fetched_at = now
    return feature_df, feature_source


def get_or_load_model() -> tuple[Any, list[str], str]:
    """Return cached model, loading once per process."""
    if (
        state.model is None
        or state.feature_columns is None
        or state.model_source is None
        or state.model_type is None
        or state.target_columns is None
    ):
        (
            state.model,
            state.feature_columns,
            state.model_source,
            state.model_type,
            state.target_columns,
        ) = load_serving_model()

    return state.model, state.feature_columns, state.model_source


def get_or_load_model_details() -> tuple[Any, list[str], str, str, list[str]]:
    """Return cached model details including model type and targets."""
    get_or_load_model()
    if state.model_type is None or state.target_columns is None:
        raise RuntimeError("Serving model metadata not initialized.")

    return (
        state.model,
        state.feature_columns,
        state.model_source,
        state.model_type,
        state.target_columns,
    )


def _latest_row_and_matrix(
    full_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Prepare latest inference row and full clean matrix for explainability."""
    missing_cols = [col for col in feature_columns if col not in full_df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing required features: {missing_cols}")

    matrix = full_df[feature_columns].copy()
    matrix = cast(pd.DataFrame, matrix.replace([np.inf, -np.inf], np.nan).dropna())
    if matrix.empty:
        raise RuntimeError("No valid feature rows after null filtering.")

    valid_df = cast(pd.DataFrame, full_df.loc[matrix.index].copy())
    latest_row = cast(pd.Series, valid_df.iloc[-1])

    matrix = cast(pd.DataFrame, matrix.reset_index(drop=True))
    return latest_row, matrix


def make_prediction() -> PredictionResponse:
    """Compute latest +24h/+48h/+72h PM2.5 forecast."""
    config = load_config()
    model, feature_columns, model_source, model_type, target_columns = get_or_load_model_details()

    feature_df, feature_source = get_cached_feature_data(config)
    latest_row, matrix = _latest_row_and_matrix(feature_df, feature_columns)

    X_latest = matrix.tail(1)
    if model_type == "horizon_specific":
        pred_map: dict[str, float] = {}
        for target_col in target_columns:
            if target_col not in model:
                raise RuntimeError(f"Missing horizon model for target: {target_col}")
            pred_map[target_col] = float(model[target_col].predict(X_latest)[0])

        forecast = {
            "+24h": pred_map.get("target_pm2_5_t_plus_24h", np.nan),
            "+48h": pred_map.get("target_pm2_5_t_plus_48h", np.nan),
            "+72h": pred_map.get("target_pm2_5_t_plus_72h", np.nan),
        }
        if any(np.isnan(value) for value in forecast.values()):
            raise RuntimeError("Horizon model predictions missing one or more forecast horizons.")
    else:
        y_pred = model.predict(X_latest)
        if len(y_pred.shape) != 2 or y_pred.shape[1] < 3:
            raise RuntimeError("Model output shape is invalid for 3-horizon prediction.")

        forecast = {
            "+24h": float(y_pred[0][0]),
            "+48h": float(y_pred[0][1]),
            "+72h": float(y_pred[0][2]),
        }
    hazardous = any(value >= config["hazardous_pm25_threshold"] for value in forecast.values())

    return PredictionResponse(
        model_source=model_source,
        feature_source=feature_source,
        city=config["default_city"],
        generated_from_timestamp=str(latest_row["timestamp"]),
        prediction_unit="ug/m^3",
        pm2_5_forecast=forecast,
        hazardous_alert=hazardous,
    )


def make_explanation(max_features: int = 10) -> ExplainResponse:
    """Return LIME top feature contributions for latest +24h prediction."""
    try:
        lime_tabular = importlib.import_module("lime.lime_tabular")
    except ImportError as exc:
        raise RuntimeError("LIME is not installed. Add lime to requirements.") from exc

    config = load_config()
    model, feature_columns, model_source, model_type, _ = get_or_load_model_details()

    feature_df, feature_source = get_cached_feature_data(config)
    latest_row, matrix = _latest_row_and_matrix(feature_df, feature_columns)

    sample_size = min(1000, len(matrix))
    background = matrix.tail(sample_size)
    X_latest = matrix.tail(1)

    explanation_model = model
    if model_type == "horizon_specific":
        explanation_model = model.get("target_pm2_5_t_plus_24h")
        if explanation_model is None:
            raise RuntimeError("Missing +24h horizon model for explainability.")

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=background.to_numpy(dtype=float),
        feature_names=feature_columns,
        mode="regression",
        random_state=42,
    )

    def predict_24h(values: np.ndarray) -> np.ndarray:
        values_df = pd.DataFrame(values, columns=feature_columns)
        preds = explanation_model.predict(values_df)
        preds_arr = np.asarray(preds)
        if preds_arr.ndim == 1:
            return preds_arr
        if preds_arr.ndim == 2 and preds_arr.shape[1] >= 1:
            return preds_arr[:, 0]
        raise RuntimeError("Model output shape is invalid for explanation.")

    explanation = explainer.explain_instance(
        data_row=X_latest.iloc[0].to_numpy(dtype=float),
        predict_fn=predict_24h,
        num_features=max_features,
    )

    weight_map = dict(explanation.as_list())
    rows = []
    for expression, weight in weight_map.items():
        matched_feature = next(
            (name for name in feature_columns if name in expression),
            expression,
        )
        rows.append(
            {
                "feature": matched_feature,
                "explanation_score": float(weight),
                "abs_explanation_score": float(abs(weight)),
            }
        )

    explanation_df = pd.DataFrame(rows)
    explanation_df = explanation_df.drop_duplicates(subset=["feature"])
    explanation_df = explanation_df.sort_values(
        "abs_explanation_score", ascending=False
    ).head(max_features)

    top_features = [
        FeatureExplanation(
            feature=str(row["feature"]),
            explanation_score=float(row["explanation_score"]),
            abs_explanation_score=float(row["abs_explanation_score"]),
        )
        for _, row in explanation_df.iterrows()
    ]

    return ExplainResponse(
        model_source=model_source,
        feature_source=feature_source,
        city=config["default_city"],
        generated_from_timestamp=str(latest_row["timestamp"]),
        horizon="+24h",
        top_features=top_features,
    )


app = FastAPI(title="Karachi AQI Forecast API", version="0.2.0")


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


@app.get("/predict/latest/explain", response_model=ExplainResponse)
def predict_latest_explain(
    max_features: int = Query(default=10, ge=3, le=25),
) -> ExplainResponse:
    """Return LIME explanation for latest +24h prediction."""
    try:
        return make_explanation(max_features=max_features)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Explainability failed: {exc}") from exc
