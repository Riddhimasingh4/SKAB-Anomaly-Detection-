"""Feature engineering: sliding windows + per-sensor statistical features.

For every window (default 30 timesteps, stride 5) and each of the 8 sensors,
three features are extracted: mean, standard deviation, and slope
(least-squares linear trend). Result: 8 sensors x 3 stats = 24 features.

Feature layout is sensor-major: [s0_mean, s0_std, s0_slope, s1_mean, ...].
This means ``errors.reshape(-1, 8, 3)`` groups features by sensor, which
``attribution.py`` relies on.

Usage:
    python -m src.features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "processed.csv"

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
STATS: list[str] = ["mean", "std", "slope"]
LABEL_COLUMN = "anomaly"
TIME_COLUMN = "datetime"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_names(sensor_columns: list[str] = SENSOR_COLUMNS) -> list[str]:
    """Return feature names in output order (sensor-major), e.g. 'Current__std'."""
    return [f"{s}__{stat}" for s in sensor_columns for stat in STATS]


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def make_windows(
    values: np.ndarray,
    labels: np.ndarray | None = None,
    window_size: int = 30,
    stride: int = 5,
    label_mode: str = "any",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Slice a time series into fixed-size sliding windows.

    Args:
        values: Array of shape (T, n_sensors), time-ordered.
        labels: Optional point-wise labels of shape (T,) (0 = normal, 1 = anomaly).
        window_size: Number of timesteps per window.
        stride: Step between consecutive window starts.
        label_mode: How to turn point labels into a window label:
            "any"      -> 1 if any point in the window is anomalous,
            "majority" -> 1 if more than half of the points are anomalous.

    Returns:
        windows: (N, window_size, n_sensors)
        window_labels: (N,) int array, or None if ``labels`` is None
        starts: (N,) start index of each window in the input series

    If the series is shorter than ``window_size``, N = 0 (empty arrays).
    """
    if window_size < 2:
        raise ValueError("window_size must be >= 2 (slope needs at least 2 points).")
    if stride < 1:
        raise ValueError("stride must be >= 1.")
    if label_mode not in ("any", "majority"):
        raise ValueError("label_mode must be 'any' or 'majority'.")

    values = np.asarray(values, dtype=np.float64)
    n_steps, n_sensors = values.shape
    starts = np.arange(0, n_steps - window_size + 1, stride)

    if len(starts) == 0:
        empty_labels = None if labels is None else np.empty((0,), dtype=int)
        return np.empty((0, window_size, n_sensors)), empty_labels, starts

    idx = starts[:, None] + np.arange(window_size)[None, :]  # (N, W)
    windows = values[idx]  # (N, W, n_sensors)

    window_labels = None
    if labels is not None:
        point_labels = np.asarray(labels, dtype=float)[idx]  # (N, W)
        if label_mode == "any":
            window_labels = (point_labels.max(axis=1) > 0).astype(int)
        else:
            window_labels = (point_labels.mean(axis=1) > 0.5).astype(int)

    return windows, window_labels, starts


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def extract_window_features(windows: np.ndarray) -> np.ndarray:
    """Compute mean, std and slope per sensor for each window (vectorized).

    Args:
        windows: Array of shape (N, window_size, n_sensors).

    Returns:
        Array of shape (N, n_sensors * 3), sensor-major layout.
        Slope is the least-squares trend per timestep; std uses ddof=0.
    """
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError(f"Expected 3D array (N, W, S), got shape {windows.shape}.")
    n_windows, window_size, n_sensors = windows.shape

    mean = windows.mean(axis=1)  # (N, S)
    std = windows.std(axis=1)  # (N, S)

    t = np.arange(window_size, dtype=np.float64)
    t_centered = t - t.mean()
    denom = np.sum(t_centered**2)
    slope = np.einsum("t,nts->ns", t_centered, windows - mean[:, None, :]) / denom

    # Stack to (N, S, 3) then flatten -> [s0_mean, s0_std, s0_slope, s1_mean, ...]
    feats = np.stack([mean, std, slope], axis=2)
    return feats.reshape(n_windows, n_sensors * len(STATS))


def window_to_features(window: np.ndarray) -> np.ndarray:
    """Convenience wrapper for a single window (W, S) -> features (1, S*3).

    Intended for inference (API / dashboard).
    """
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2:
        raise ValueError(f"Expected 2D array (W, S), got shape {window.shape}.")
    return extract_window_features(window[None, :, :])


# --------------------------------------------------------------------------- #
# DataFrame-level pipeline
# --------------------------------------------------------------------------- #
def build_features(
    df: pd.DataFrame,
    window_size: int = 30,
    stride: int = 5,
    sensor_columns: list[str] = SENSOR_COLUMNS,
    label_mode: str = "any",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build window features and labels from a preprocessed DataFrame.

    Windows are created separately for every (source, split) segment, so a
    window never spans two files or two splits. If the frame has no
    ``source`` / ``split`` columns, it is treated as one continuous series.

    Returns:
        X: (N, 24) feature matrix
        y: (N,) window labels (0 = normal, 1 = anomaly)
        meta: DataFrame with ``source``, ``split``, ``start_time`` per window
    """
    df = df.copy()
    if "source" not in df.columns:
        df["source"] = "all"
    if "split" not in df.columns:
        df["split"] = "all"
    if LABEL_COLUMN not in df.columns:
        df[LABEL_COLUMN] = 0.0

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    meta_parts: list[pd.DataFrame] = []

    for (source, split), seg in df.groupby(["source", "split"], sort=False):
        if TIME_COLUMN in seg.columns:
            seg = seg.sort_values(TIME_COLUMN)
        windows, labels, starts = make_windows(
            seg[sensor_columns].to_numpy(),
            seg[LABEL_COLUMN].to_numpy(),
            window_size=window_size,
            stride=stride,
            label_mode=label_mode,
        )
        if len(windows) == 0:
            logger.warning("Segment (%s, %s) shorter than window; skipped.", source, split)
            continue

        x_parts.append(extract_window_features(windows))
        y_parts.append(labels)
        start_time = (
            seg[TIME_COLUMN].to_numpy()[starts] if TIME_COLUMN in seg.columns else starts
        )
        meta_parts.append(
            pd.DataFrame({"source": source, "split": split, "start_time": start_time})
        )

    n_features = len(sensor_columns) * len(STATS)
    if not x_parts:
        return np.empty((0, n_features)), np.empty((0,), dtype=int), pd.DataFrame(
            columns=["source", "split", "start_time"]
        )

    X = np.vstack(x_parts)
    y = np.concatenate(y_parts)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta


def build_dataset(
    df: pd.DataFrame, config: dict[str, Any] | None = None
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    """Build features for every split using ``configs/config.yaml`` settings.

    Returns:
        Dict mapping split name ("train", "val", "test") to (X, y, meta).
    """
    config = config or load_config()
    X, y, meta = build_features(
        df,
        window_size=config["window_size"],
        stride=config["stride"],
    )
    out: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for split in meta["split"].unique():
        mask = (meta["split"] == split).to_numpy()
        out[str(split)] = (X[mask], y[mask], meta[mask].reset_index(drop=True))
    return out


def load_processed(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    """Load ``data/processed/processed.csv`` produced by ``preprocess.py``."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.preprocess` first.")
    return pd.read_csv(path, parse_dates=[TIME_COLUMN])


# --------------------------------------------------------------------------- #
# Self-test / demo
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """Verify feature math on synthetic data (no real files needed)."""
    rng = np.random.default_rng(0)
    n_steps, n_sensors, w, s = 100, len(SENSOR_COLUMNS), 30, 5

    # Known linear ramp in sensor 0: slope must be exactly 0.5 per step.
    values = rng.normal(size=(n_steps, n_sensors))
    values[:, 0] = 0.5 * np.arange(n_steps)
    labels = np.zeros(n_steps)
    labels[60:70] = 1

    windows, win_labels, starts = make_windows(values, labels, w, s)
    feats = extract_window_features(windows)

    expected_n = (n_steps - w) // s + 1
    assert windows.shape == (expected_n, w, n_sensors), windows.shape
    assert feats.shape == (expected_n, 24), feats.shape
    assert len(get_feature_names()) == 24
    assert np.allclose(feats[:, 2], 0.5), "slope of ramp should be 0.5"
    ref = np.polyfit(np.arange(w), windows[3, :, 4], 1)[0]
    assert np.isclose(feats[3, 4 * 3 + 2], ref), "slope mismatch vs np.polyfit"
    assert win_labels is not None and win_labels.sum() > 0
    assert window_to_features(windows[0]).shape == (1, 24)
    print("Self-test passed: windows", windows.shape, "| features", feats.shape)


def main() -> None:
    """Run the self-test, then summarise real processed data if available."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    config = load_config()
    print(f"Config: window_size={config['window_size']}, stride={config['stride']}")
    _self_test()

    try:
        df = load_processed()
    except FileNotFoundError as e:
        print(f"Skipping real-data demo: {e}")
        return

    dataset = build_dataset(df, config)
    print("\nFeatures per split:")
    for split, (X, y, _) in dataset.items():
        print(f"  {split:5s} X={X.shape}  anomalous windows={int(y.sum())}/{len(y)}")


if __name__ == "__main__":
    main()