from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from src.features import build_dataset, get_feature_names, load_config, load_processed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "iforest.joblib"
VAL_SCORES_PATH = MODELS_DIR / "iforest_val_scores.csv"

N_ESTIMATORS = 200


def score_windows(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores for feature matrix ``X`` (higher = more anomalous)."""
    return -model.score_samples(X)


def train_iforest(
    X_train: np.ndarray,
    contamination: float,
    random_state: int,
    n_estimators: int = N_ESTIMATORS,
) -> IsolationForest:
    """Fit an IsolationForest on the given (normal) training features."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train)
    return model


def main() -> None:
    """Train the Isolation Forest, save it, and save validation scores."""
    config = load_config()
    contamination = float(config["contamination"])
    seed = int(config["random_seed"])

    print("=" * 60)
    print("Isolation Forest training")
    print("=" * 60)
    print(
        f"Config: window_size={config['window_size']}, stride={config['stride']}, "
        f"contamination={contamination}, random_state={seed}, "
        f"n_estimators={N_ESTIMATORS}"
    )

    # 1-2. Load processed data and build windowed features
    print("\n[1/4] Loading processed data and building features...")
    df = load_processed()
    dataset = build_dataset(df, config)
    for split in ("train", "val"):
        if split not in dataset:
            raise RuntimeError(f"Split '{split}' not found. Re-run src.preprocess.")
    for split, (X, y, _) in dataset.items():
        print(f"  {split:5s} windows={len(X):5d}  features={X.shape[1]}  "
              f"anomalous windows={int(y.sum())}")

    # 3. Select normal training windows only
    X_train_all, y_train_all, _ = dataset["train"]
    X_train = X_train_all[y_train_all == 0]
    print(f"\n[2/4] Training on normal windows only: "
          f"{len(X_train)} of {len(X_train_all)} train windows "
          f"(excluded {len(X_train_all) - len(X_train)} anomalous)")
    if len(X_train) == 0:
        raise RuntimeError("No normal training windows available.")

    start = time.perf_counter()
    model = train_iforest(X_train, contamination, seed)
    elapsed = time.perf_counter() - start
    print(f"  Fitted IsolationForest in {elapsed:.2f}s")

    # 4. Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n[3/4] Saved model -> {MODEL_PATH}")

    # 5. Validation scores
    X_val, y_val, meta_val = dataset["val"]
    val_scores = score_windows(model, X_val)
    scores_df = meta_val.copy()
    scores_df["score"] = val_scores
    scores_df["label"] = y_val
    scores_df.to_csv(VAL_SCORES_PATH, index=False)
    print(f"[4/4] Saved validation scores -> {VAL_SCORES_PATH}")

    # Basic info
    print("\nValidation score summary (higher = more anomalous):")
    print(f"  windows : {len(val_scores)}")
    print(f"  min/mean/max : {val_scores.min():.4f} / {val_scores.mean():.4f} / "
          f"{val_scores.max():.4f}")
    if len(np.unique(y_val)) == 2:
        print(f"  mean score normal   : {val_scores[y_val == 0].mean():.4f}")
        print(f"  mean score anomalous: {val_scores[y_val == 1].mean():.4f}")
        print(f"  quick sanity ROC-AUC: {roc_auc_score(y_val, val_scores):.3f} "
              f"(full evaluation happens in src/evaluate.py)")
    print(f"\nFeature layout: {len(get_feature_names())} features "
          f"(8 sensors x mean/std/slope)")
    print("Done.")


if __name__ == "__main__":
    main()