"""FastAPI service for SKAB industrial sensor anomaly detection.

Run from the project root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> anomaly score, anomaly flag and top contributing sensors

Configuration (environment variables):
    SKAB_ANOMALY_THRESHOLD  Reconstruction-error threshold (default: 0.1)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.features import SENSOR_COLUMNS, window_to_features
from src.preprocess import load_artifacts, transform_new_data
from src.train_autoencoder import load_autoencoder

__all__ = [
    "SENSOR_COLUMNS",
    "ModelBundle",
    "PredictRequest",
    "PredictResponse",
    "SensorReading",
    "app",
    "load_models",
    "run_inference",
]

logger = logging.getLogger("skab-api")
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WINDOW_SIZE: int = 30
N_SENSORS: int = len(SENSOR_COLUMNS)
N_STATS: int = 3  # mean, std, slope (sensor-major layout)
N_FEATURES: int = N_SENSORS * N_STATS  # 24

DEFAULT_THRESHOLD: float = 0.1
THRESHOLD_ENV_VAR: str = "SKAB_ANOMALY_THRESHOLD"
TOP_K_SENSORS: int = 3


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class SensorReading(BaseModel):
    """A single timestep of the 8 sensor measurements."""

    model_config = ConfigDict(populate_by_name=True)

    accelerometer1_rms: float = Field(alias="Accelerometer1RMS", allow_inf_nan=False)
    accelerometer2_rms: float = Field(alias="Accelerometer2RMS", allow_inf_nan=False)
    current: float = Field(alias="Current", allow_inf_nan=False)
    pressure: float = Field(alias="Pressure", allow_inf_nan=False)
    temperature: float = Field(alias="Temperature", allow_inf_nan=False)
    thermocouple: float = Field(alias="Thermocouple", allow_inf_nan=False)
    voltage: float = Field(alias="Voltage", allow_inf_nan=False)
    volume_flow_rate_rms: float = Field(alias="Volume Flow RateRMS", allow_inf_nan=False)

    def to_row(self) -> list[float]:
        """Return the reading as a list ordered like ``SENSOR_COLUMNS``."""
        data = self.model_dump(by_alias=True)
        return [float(data[col]) for col in SENSOR_COLUMNS]


class PredictRequest(BaseModel):
    """A chronological window of sensor readings (oldest first).

    At least ``WINDOW_SIZE`` readings are required. If more are sent, only the
    most recent ``WINDOW_SIZE`` readings are used.
    """

    readings: list[SensorReading] = Field(min_length=1)


class PredictResponse(BaseModel):
    """Prediction result for one window."""

    anomaly_score: float
    is_anomaly: bool
    top_sensors: list[str]
    model_used: Literal["autoencoder"] = "autoencoder"


# --------------------------------------------------------------------------- #
# Model bundle
# --------------------------------------------------------------------------- #
@dataclass
class ModelBundle:
    """Everything needed for inference, loaded once at startup."""

    artifacts: Any  # PreprocessArtifacts
    autoencoder: torch.nn.Module
    feature_mean: np.ndarray
    feature_std: np.ndarray
    threshold: float
    window_size: int = WINDOW_SIZE
    top_k: int = TOP_K_SENSORS


def _read_threshold() -> float:
    """Read the anomaly threshold from the environment, falling back to the default."""
    raw = os.getenv(THRESHOLD_ENV_VAR)
    if raw is None:
        return DEFAULT_THRESHOLD
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{THRESHOLD_ENV_VAR} must be a number, got {raw!r}.") from exc


def load_models() -> ModelBundle:
    """Load the preprocessing artifacts and the trained autoencoder."""
    artifacts = load_artifacts()
    model, feature_mean, feature_std = load_autoencoder()
    model.eval()

    mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(feature_std, dtype=np.float32).reshape(1, -1)
    std = np.where(std < 1e-8, 1.0, std).astype(np.float32)  # avoid divide-by-zero

    threshold = _read_threshold()
    logger.info("Models loaded (window_size=%d, threshold=%.6f)", WINDOW_SIZE, threshold)
    return ModelBundle(
        artifacts=artifacts,
        autoencoder=model,
        feature_mean=mean,
        feature_std=std,
        threshold=threshold,
    )


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def run_inference(bundle: ModelBundle, readings: list[SensorReading]) -> PredictResponse:
    """Run preprocessing, feature extraction and autoencoder scoring.

    Pipeline: ``transform_new_data`` -> ``window_to_features`` -> feature
    standardisation (``feature_mean`` / ``feature_std``) -> autoencoder ->
    mean squared reconstruction error.

    Args:
        bundle: Loaded artifacts, model and settings.
        readings: Chronological sensor readings (oldest first).

    Returns:
        The prediction for the most recent window.

    Raises:
        ValueError: If there are too few readings or a shape is unexpected.
    """
    if len(readings) < bundle.window_size:
        raise ValueError(
            f"At least {bundle.window_size} readings are required, got {len(readings)}."
        )

    rows = [r.to_row() for r in readings[-bundle.window_size:]]
    frame = pd.DataFrame(rows, columns=SENSOR_COLUMNS)

    scaled = transform_new_data(frame, bundle.artifacts)
    if len(scaled) < bundle.window_size:
        raise ValueError(
            f"Only {len(scaled)} rows remained after cleaning; "
            f"{bundle.window_size} are required."
        )

    window = scaled[SENSOR_COLUMNS].tail(bundle.window_size).to_numpy(dtype=np.float64)
    features = np.asarray(window_to_features(window), dtype=np.float32).reshape(1, -1)
    if features.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}.")

    x = torch.from_numpy((features - bundle.feature_mean) / bundle.feature_std)
    with torch.no_grad():
        reconstruction = bundle.autoencoder(x)
    per_feature_error = ((reconstruction - x) ** 2).squeeze(0).cpu().numpy()

    score = float(per_feature_error.mean())

    # Sensor-major layout: [s0_mean, s0_std, s0_slope, s1_mean, ...]
    sensor_errors = per_feature_error.reshape(N_SENSORS, N_STATS).sum(axis=1)
    top_idx = np.argsort(sensor_errors)[::-1][: bundle.top_k]

    return PredictResponse(
        anomaly_score=score,
        is_anomaly=bool(score > bundle.threshold),
        top_sensors=[SENSOR_COLUMNS[i] for i in top_idx],
    )


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load all models once when the application starts."""
    app.state.bundle = load_models()
    yield
    app.state.bundle = None


app = FastAPI(
    title="SKAB Anomaly Detection API",
    description="Unsupervised anomaly detection on industrial sensor data.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 response and log the underlying error."""
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Score the most recent window of sensor readings."""
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        return run_inference(bundle, payload.readings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc