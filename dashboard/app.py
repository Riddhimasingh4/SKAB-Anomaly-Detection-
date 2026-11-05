"""Streamlit dashboard for SKAB industrial sensor anomaly detection.

Run from the project root:
    streamlit run dashboard/app.py

The dashboard reuses the API's inference pipeline (``run_inference`` in
``api/main.py``), which chains ``load_artifacts`` / ``transform_new_data``,
``window_to_features`` and ``load_autoencoder``. Results are therefore
identical to those returned by ``POST /predict``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Make the project root importable when launched via `streamlit run`.
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import (  # noqa: E402
    SENSOR_COLUMNS,
    ModelBundle,
    PredictResponse,
    SensorReading,
    load_models,
    run_inference,
)

RESULT_KEY = "detection_result"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models...")
def get_bundle() -> ModelBundle:
    """Load the preprocessing artifacts and autoencoder once per server process."""
    return load_models()


def load_csv(uploaded: UploadedFile) -> pd.DataFrame:
    """Read an uploaded SKAB-format CSV (auto-detects ``;`` or ``,`` separators).

    If a ``datetime`` column exists, it is parsed and used as the index. Label
    columns such as ``anomaly`` and ``changepoint`` are ignored by detection.
    """
    df = pd.read_csv(uploaded, sep=None, engine="python")
    df.columns = [str(c).strip() for c in df.columns]

    datetime_col = next((c for c in df.columns if c.lower() == "datetime"), None)
    if datetime_col is not None:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=[datetime_col]).set_index(datetime_col)
    return df


def missing_columns(df: pd.DataFrame) -> list[str]:
    """Return the required sensor columns that are absent from the data."""
    return [c for c in SENSOR_COLUMNS if c not in df.columns]


def detect(bundle: ModelBundle, df: pd.DataFrame) -> PredictResponse:
    """Score the most recent window of the uploaded data.

    Args:
        bundle: Loaded preprocessing artifacts and model.
        df: Uploaded data containing all sensor columns.

    Returns:
        The prediction for the latest ``window_size`` valid rows.

    Raises:
        ValueError: If there are too few valid rows or the values are invalid.
    """
    sensors = df[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    clean = sensors.ffill().dropna()  # mirrors the forward-fill + dropna step
    if len(clean) < bundle.window_size:
        raise ValueError(
            f"Not enough valid rows: need at least {bundle.window_size}, "
            f"found {len(clean)}."
        )

    tail = clean.tail(bundle.window_size)
    try:
        readings = [
            SensorReading.model_validate(record)
            for record in tail.to_dict(orient="records")
        ]
    except ValidationError as exc:
        raise ValueError(f"Invalid sensor values in the uploaded data: {exc}") from exc

    return run_inference(bundle, readings)


def top_sensors_frame(top_sensors: Any) -> pd.DataFrame:
    """Convert the ``top_sensors`` field of a prediction into a DataFrame."""
    if not top_sensors:
        return pd.DataFrame({"sensor": []})
    items = list(top_sensors)
    if isinstance(items[0], dict):
        return pd.DataFrame(items)
    if hasattr(items[0], "model_dump"):
        return pd.DataFrame([item.model_dump() for item in items])
    return pd.DataFrame({"sensor": [str(item) for item in items]})


def show_result(result: PredictResponse, df: pd.DataFrame) -> None:
    """Render the anomaly score, verdict, top sensors and the sensor chart."""
    col_score, col_flag = st.columns(2)
    col_score.metric("Anomaly Score", f"{result.anomaly_score:.4f}")
    col_flag.metric("Is Anomaly", "Yes" if result.is_anomaly else "No")

    if result.is_anomaly:
        st.error("The latest window looks anomalous.")
    else:
        st.success("The latest window looks normal.")

    st.subheader("Top contributing sensors")
    st.dataframe(top_sensors_frame(result.top_sensors), use_container_width=True)

    st.subheader("Sensor readings")
    numeric = df[SENSOR_COLUMNS].apply(pd.to_numeric, errors="coerce")
    selected = st.multiselect(
        "Sensors to plot", SENSOR_COLUMNS, default=SENSOR_COLUMNS[:3]
    )
    if selected:
        st.line_chart(numeric[selected])


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    """Build the Streamlit page."""
    st.set_page_config(page_title="SKAB Anomaly Detection", layout="wide")
    st.title("SKAB Anomaly Detection Dashboard")
    st.write(
        "Upload a SKAB-format CSV with the 8 sensor columns. The most recent "
        "window of rows is scored by the autoencoder."
    )

    try:
        bundle = get_bundle()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not load the models: {exc}")
        st.stop()

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file to get started.")
        return

    try:
        df = load_csv(uploaded)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the CSV file: {exc}")
        return

    missing = missing_columns(df)
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        return

    st.caption(f"{len(df)} rows loaded. Window size: {bundle.window_size}.")

    if st.button("Run Detection", type="primary"):
        try:
            st.session_state[RESULT_KEY] = detect(bundle, df)
        except ValueError as exc:
            st.session_state.pop(RESULT_KEY, None)
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.session_state.pop(RESULT_KEY, None)
            st.error(f"Detection failed: {exc}")

    result = st.session_state.get(RESULT_KEY)
    if result is not None:
        show_result(result, df)


main()