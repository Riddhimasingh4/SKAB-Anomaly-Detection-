"""Preprocessing: load raw SKAB CSVs, clean, clip outliers, scale, split, and save.
 
Pipeline (PRD 8.1):
    1. Load all SKAB CSV files from ``data/raw/`` (recursively).
    2. Handle missing values (forward-fill, then drop remaining NaNs).
    3. Clip outliers using the IQR rule (bounds fitted on normal data only).
    4. Fit a ``StandardScaler`` on normal data only (prevents leakage).
    5. Chronological train / validation / test split (per source file).
    6. Save the processed data to ``data/processed/`` and the fitted
       artefacts (scaler, clip bounds, sensor list) to ``models/``.
 
Windowing and feature extraction are handled in ``src/features.py``.
 
Usage:
    python -m src.preprocess


"""
 
from __future__ import annotations
 
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
 
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
 
logger = logging.getLogger(__name__)
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
 
PROCESSED_FILE = "processed.csv"
ARTEFACTS_FILE = "preprocess_artifacts.joblib"
 
SENSOR_COLUMNS: list[str] = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
LABEL_COLUMNS: list[str] = ["anomaly", "changepoint"]
TIME_COLUMN = "datetime"
 
IQR_FACTOR = 1.5
# Chronological split fractions applied to each source file.
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2  # remaining 0.2 -> test
# Folder name that holds SKAB's anomaly-free (normal) recording.
NORMAL_FOLDER = "anomaly-free"
 
 
@dataclass
class PreprocessArtifacts:
    """Fitted objects needed to preprocess new data at inference time."""
 
    sensor_columns: list[str]
    clip_lower: pd.Series
    clip_upper: pd.Series
    scaler: StandardScaler
 
 
# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
 
 
# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load every SKAB CSV found under ``raw_dir`` into one DataFrame.
 
    SKAB files are semicolon-separated. Each row gets a ``source`` column
    (path relative to ``raw_dir``) and an ``is_normal_source`` flag that is
    True for files in the ``anomaly-free`` folder.
 
    Raises:
        FileNotFoundError: if no CSV files are found.
        ValueError: if a file is missing required sensor columns.
    """
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. See README for how to download SKAB."
        )
 
    frames: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, sep=";", parse_dates=[TIME_COLUMN])
        missing = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing sensor columns: {missing}")
 
        rel = path.relative_to(raw_dir)
        df["source"] = rel.as_posix()
        df["is_normal_source"] = NORMAL_FOLDER in rel.parts
        for col in LABEL_COLUMNS:
            if col not in df.columns:  # anomaly-free file has no labels
                df[col] = 0.0
        frames.append(df)
        logger.info("Loaded %s (%d rows)", rel, len(df))
 
    data = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d files, %d rows total", len(files), len(data))
    return data
 
 
# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean(df: pd.DataFrame, sensor_columns: list[str] = SENSOR_COLUMNS) -> pd.DataFrame:
    """Sort by time, forward-fill missing sensor values, then drop leftover NaNs.
 
    Forward-fill is applied per source file so values never leak across files.
    """
    df = df.copy()
    sort_cols = ["source", TIME_COLUMN] if "source" in df.columns else [TIME_COLUMN]
    df = df.sort_values(sort_cols).reset_index(drop=True)
 
    if "source" in df.columns:
        df[sensor_columns] = df.groupby("source")[sensor_columns].ffill()
    else:
        df[sensor_columns] = df[sensor_columns].ffill()
 
    before = len(df)
    df = df.dropna(subset=sensor_columns).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info("Dropped %d rows with unresolved missing values", dropped)
    return df
 
 
# --------------------------------------------------------------------------- #
# Outlier clipping
# --------------------------------------------------------------------------- #
def fit_iqr_bounds(
    normal_df: pd.DataFrame,
    sensor_columns: list[str] = SENSOR_COLUMNS,
    factor: float = IQR_FACTOR,
) -> tuple[pd.Series, pd.Series]:
    """Compute per-sensor IQR clipping bounds from normal data."""
    q1 = normal_df[sensor_columns].quantile(0.25)
    q3 = normal_df[sensor_columns].quantile(0.75)
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr
 
 
def clip_outliers(
    df: pd.DataFrame,
    lower: pd.Series,
    upper: pd.Series,
    sensor_columns: list[str] = SENSOR_COLUMNS,
) -> pd.DataFrame:
    """Clip sensor values to the given per-sensor bounds."""
    df = df.copy()
    df[sensor_columns] = df[sensor_columns].clip(lower=lower, upper=upper, axis=1)
    return df
 
 
# --------------------------------------------------------------------------- #
# Normal-data selection
# --------------------------------------------------------------------------- #
def select_normal(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows considered normal, used for fitting clip bounds and scaler.
 
    Prefers the dedicated ``anomaly-free`` recording. If it is absent, falls
    back to rows labelled ``anomaly == 0`` (with a warning).
    """
    normal = df[df["is_normal_source"]]
    if normal.empty:
        logger.warning(
            "No anomaly-free file found; falling back to rows with anomaly == 0."
        )
        normal = df[df["anomaly"] == 0]
    if normal.empty:
        raise ValueError("No normal data available to fit the scaler.")
    return normal
 
 
# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def fit_scaler(
    normal_df: pd.DataFrame, sensor_columns: list[str] = SENSOR_COLUMNS
) -> StandardScaler:
    """Fit a StandardScaler on normal data only."""
    scaler = StandardScaler()
    scaler.fit(normal_df[sensor_columns])
    return scaler
 
 
def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler,
    sensor_columns: list[str] = SENSOR_COLUMNS,
) -> pd.DataFrame:
    """Return a copy of ``df`` with sensor columns standardized."""
    df = df.copy()
    df[sensor_columns] = scaler.transform(df[sensor_columns])
    return df
 
 
# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def chronological_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> pd.DataFrame:
    """Add a ``split`` column (train / val / test) using time order per file.
 
    Splitting inside each source file keeps windows from straddling
    experiments and guarantees test data is always later in time than
    train/validation data.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1.")
 
    df = df.copy()
    df["split"] = "test"
    for _, idx in df.groupby("source").groups.items():
        idx = np.asarray(idx)  # already time-sorted by clean()
        n = len(idx)
        n_train = int(n * train_frac)
        n_val = int(n * (train_frac + val_frac)) - n_train
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train : n_train + n_val], "split"] = "val"
    return df
 
 
# --------------------------------------------------------------------------- #
# Inference-time helper (reused by the API / dashboard later)
# --------------------------------------------------------------------------- #
def transform_new_data(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    """Apply the fitted cleaning, clipping and scaling to unseen sensor data."""
    cols = artifacts.sensor_columns
    df = df.copy()
    df[cols] = df[cols].ffill()
    df = df.dropna(subset=cols).reset_index(drop=True)
    df = clip_outliers(df, artifacts.clip_lower, artifacts.clip_upper, cols)
    return apply_scaler(df, artifacts.scaler, cols)
 
 
def load_artifacts(models_dir: Path = MODELS_DIR) -> PreprocessArtifacts:
    """Load the saved preprocessing artefacts from ``models/``."""
    return joblib.load(models_dir / ARTEFACTS_FILE)
 
 
# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_preprocessing(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and persist outputs.
 
    Returns:
        The processed DataFrame (scaled sensors, labels, ``source`` and
        ``split`` columns).
    """
    config = load_config()
    np.random.seed(config.get("random_seed", 42))
 
    data = load_raw(raw_dir)
    data = clean(data)
 
    normal = select_normal(data)
    lower, upper = fit_iqr_bounds(normal)
    data = clip_outliers(data, lower, upper)
 
    # Re-select normal rows after clipping, then fit the scaler on them only.
    normal = select_normal(data)
    scaler = fit_scaler(normal)
    data = apply_scaler(data, scaler)
 
    data = chronological_split(data)
 
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
 
    out_cols = (
        [TIME_COLUMN, "source", "is_normal_source", "split"]
        + SENSOR_COLUMNS
        + LABEL_COLUMNS
    )
    out_path = processed_dir / PROCESSED_FILE
    data[out_cols].to_csv(out_path, index=False)
 
    artifacts = PreprocessArtifacts(
        sensor_columns=SENSOR_COLUMNS,
        clip_lower=lower,
        clip_upper=upper,
        scaler=scaler,
    )
    joblib.dump(artifacts, models_dir / ARTEFACTS_FILE)
 
    logger.info("Saved processed data -> %s", out_path)
    logger.info("Saved artefacts -> %s", models_dir / ARTEFACTS_FILE)
    logger.info("Split sizes: %s", data["split"].value_counts().to_dict())
    return data
 
 
def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run_preprocessing()
 
 
if __name__ == "__main__":
    main()
