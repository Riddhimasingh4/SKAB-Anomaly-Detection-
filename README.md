# skab-anomaly-mvp

An MVP system for **industrial sensor anomaly detection** built on the [SKAB (Skoltech Anomaly Benchmark)](https://github.com/waico/SKAB) dataset. It trains an Isolation Forest baseline and a PyTorch autoencoder on multivariate sensor windows, evaluates them, serves predictions through a FastAPI service, and visualizes results in a Streamlit dashboard.

---

## MVP Scope

### Included
- Preprocessing of raw SKAB CSV files (cleaning, scaling)
- Sliding-window feature generation (configurable window size and stride)
- Isolation Forest baseline model
- PyTorch autoencoder (reconstruction-error based detection)
- Percentile-based anomaly thresholding
- Evaluation (precision, recall, F1, ROC-AUC)
- Basic per-sensor anomaly attribution
- FastAPI `/predict` endpoint
- Streamlit dashboard
- Docker / docker-compose setup

### Deferred (not in MVP)
- Real-time streaming ingestion (Kafka, MQTT, etc.)
- Online / incremental model retraining
- Advanced models (LSTM/Transformer autoencoders, TCN)
- Hyperparameter search and experiment tracking (MLflow, W&B)
- Authentication, rate limiting, and production hardening
- Alerting / notification integrations
- CI/CD and cloud deployment

> **Status:** This repository currently contains the project skeleton only. Modules contain placeholders and will be implemented incrementally.

---

## Project Structure

```
skab-anomaly-mvp/
├── data/
│   ├── raw/                 # SKAB CSV files (you provide)
│   └── processed/           # Preprocessed data
├── notebooks/               # Exploration notebooks
├── src/
│   ├── preprocess.py        # Load, clean, scale
│   ├── features.py          # Sliding windows / features
│   ├── train_iforest.py     # Isolation Forest training
│   ├── train_autoencoder.py # Autoencoder training
│   ├── evaluate.py          # Model evaluation
│   └── attribution.py       # Sensor-level attribution
├── api/main.py              # FastAPI service
├── dashboard/app.py         # Streamlit dashboard
├── models/                  # Saved model artifacts
├── configs/config.yaml      # Hyperparameters and settings
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 1. Download the SKAB Dataset

1. Clone or download the SKAB repository:
   ```bash
   git clone https://github.com/waico/SKAB.git /tmp/SKAB
   ```
2. Copy the CSV files from its `data/` folder into `data/raw/`, keeping the subfolder layout (e.g. `valve1/`, `valve2/`, `other/`, `anomaly-free/`):
   ```bash
   cp -r /tmp/SKAB/data/* data/raw/
   ```
3. Verify:
   ```bash
   ls data/raw
   ```

The CSV files are semicolon-separated (`;`) and contain a `datetime` column, sensor channels, an `anomaly` label, and a `changepoint` label.

---

## 2. Install Dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Run Preprocessing, Training, and Evaluation

Run all commands from the project root.

```bash
# Preprocess raw data -> data/processed/
python -m src.preprocess

# Train models -> models/
python -m src.train_iforest
python -m src.train_autoencoder

# Evaluate
python -m src.evaluate
```

Hyperparameters (window size, stride, epochs, etc.) are set in `configs/config.yaml`.

---

## 4. Start the FastAPI Service

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- API root: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### Example `/predict` request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "window": [
      [0.21, 0.35, 1.02, 0.98, 71.2, 24.1, 32.5, 0.01],
      [0.22, 0.36, 1.01, 0.97, 71.4, 24.1, 32.5, 0.01]
    ],
    "model": "autoencoder"
  }'
```

Each row of `window` is one timestep with the 8 SKAB sensor readings (Accelerometer1RMS, Accelerometer2RMS, Current, Pressure, Temperature, Thermocouple, Voltage, Volume Flow RateRMS). The exact request/response schema will be finalized when the endpoint is implemented.

---

## 5. Start the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501.

---

## 6. Run Everything with Docker

```bash
# Build and start the API and dashboard
docker compose up --build

# Run in the background
docker compose up -d --build

# Stop
docker compose down
```

| Service   | URL                    |
|-----------|------------------------|
| API       | http://localhost:8000  |
| Dashboard | http://localhost:8501  |

The `data/`, `models/`, and `configs/` folders are mounted as volumes, so train models locally (or inside the container) and they will be available to the services.

Run a pipeline step inside Docker:

```bash
docker compose run --rm api python -m src.preprocess
```

---

## Configuration

Defaults in `configs/config.yaml`:

| Key                  | Default        | Description                                  |
|----------------------|----------------|----------------------------------------------|
| `window_size`        | 30             | Timesteps per window                         |
| `stride`             | 5              | Step between windows                         |
| `anomaly_percentile` | 99.0           | Percentile of normal scores used as threshold|
| `random_seed`        | 42             | Reproducibility seed                         |
| `ae_hidden_dims`     | [64, 32, 16]   | Autoencoder hidden layer sizes               |
| `ae_epochs`          | 50             | Autoencoder training epochs                  |
| `ae_batch_size`      | 64             | Autoencoder batch size                       |
| `ae_lr`              | 0.001          | Autoencoder learning rate                    |
| `contamination`      | 0.05           | Isolation Forest expected anomaly fraction   |

---

## License

SKAB is released under its own license; see the [SKAB repository](https://github.com/waico/SKAB). Add your project license here.
