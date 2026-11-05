"""Train a dense PyTorch autoencoder on NORMAL windows and save it to models/.

Architecture (from config ``ae_hidden_dims``), with 24 input features:
    24 -> 64 -> 32 -> 16 -> 32 -> 64 -> 24

Steps:
    1. Load data/processed/processed.csv and build 24-feature windows.
    2. Standardize the features using statistics from NORMAL train windows.
    3. Train with MSE loss on normal train windows only. Early stopping
       (patience = 7) monitors the loss on NORMAL validation windows.
    4. Save the best model to models/autoencoder.pt (weights + feature
       statistics + architecture, everything needed for inference).
    5. Save per-window validation reconstruction errors to
       models/autoencoder_val_scores.csv.

Anomaly score convention: HIGHER = MORE ANOMALOUS
(score = mean squared reconstruction error over the 24 features).

Usage:
    python -m src.train_autoencoder
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from sklearn.metrics import roc_auc_score

from src.features import build_dataset, get_feature_names, load_config, load_processed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "autoencoder.pt"
VAL_SCORES_PATH = MODELS_DIR / "autoencoder_val_scores.csv"

PATIENCE = 7
MIN_DELTA = 1e-6  # minimum validation-loss improvement that counts
EPS = 1e-8


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Autoencoder(nn.Module):
    """Symmetric dense autoencoder: input -> hidden dims -> latent -> mirror -> input."""

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)

        enc_sizes = [input_dim] + self.hidden_dims
        enc_layers: list[nn.Module] = []
        for i in range(len(enc_sizes) - 1):
            enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i + 1]))
            if i < len(enc_sizes) - 2:  # no activation on the latent layer
                enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_sizes = self.hidden_dims[::-1] + [input_dim]
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_sizes) - 1):
            dec_layers.append(nn.Linear(dec_sizes[i], dec_sizes[i + 1]))
            if i < len(dec_sizes) - 2:  # linear output layer
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``x``."""
        return self.decoder(self.encoder(x))

    def architecture_string(self) -> str:
        """Human-readable layer sizes, e.g. '24 -> 64 -> 32 -> 16 -> 32 -> 64 -> 24'."""
        sizes = [self.input_dim] + self.hidden_dims + self.hidden_dims[-2::-1] + [self.input_dim]
        return " -> ".join(str(s) for s in sizes)


# --------------------------------------------------------------------------- #
# Feature standardization (fitted on normal training windows only)
# --------------------------------------------------------------------------- #
def fit_feature_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-feature mean and std (std floored to avoid division by zero)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < EPS, 1.0, std)
    return mean, std


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize features with the given statistics."""
    return (X - mean) / std


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def reconstruction_errors(
    model: Autoencoder,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    per_feature: bool = False,
    device: str = "cpu",
) -> np.ndarray:
    """Squared reconstruction error of raw (unstandardized) window features.

    Args:
        per_feature: if True return (N, n_features) squared errors (used for
            sensor attribution); otherwise return the (N,) mean per window.
    """
    model.eval()
    x = torch.as_tensor(standardize(X, mean, std), dtype=torch.float32, device=device)
    err = (model(x) - x) ** 2
    err = err.cpu().numpy()
    return err if per_feature else err.mean(axis=1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    hidden_dims: list[int],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    patience: int = PATIENCE,
    device: str = "cpu",
) -> tuple[Autoencoder, dict[str, Any]]:
    """Train with MSE loss and early stopping on validation loss.

    ``X_train`` and ``X_val`` must already be standardized.

    Returns:
        The model restored to its best epoch, and a history dict with
        ``train_loss``, ``val_loss``, ``best_epoch`` and ``stopped_early``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Autoencoder(X_train.shape[1], hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    x_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    n = len(x_train)

    generator = torch.Generator().manual_seed(seed)
    history: dict[str, Any] = {
        "train_loss": [], "val_loss": [], "best_epoch": 0, "stopped_early": False,
    }
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=generator)
        running = 0.0
        for start in range(0, n, batch_size):
            batch = x_train[perm[start : start + batch_size].to(device)]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(batch)
        train_loss = running / n

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(x_val), x_val).item()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - MIN_DELTA
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        marker = "  * best" if improved else f"  (no improvement {epochs_without_improvement}/{patience})"
        print(f"  Epoch {epoch:3d}/{epochs} | train loss {train_loss:.5f} | "
              f"val loss {val_loss:.5f}{marker}")

        if epochs_without_improvement >= patience:
            history["stopped_early"] = True
            print(f"  Early stopping at epoch {epoch} (patience = {patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, history


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def save_autoencoder(
    model: Autoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    config: dict[str, Any],
    path: Path = MODEL_PATH,
) -> None:
    """Save weights, architecture and feature statistics in one checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": model.input_dim,
            "hidden_dims": model.hidden_dims,
            "feature_mean": torch.as_tensor(mean, dtype=torch.float64),
            "feature_std": torch.as_tensor(std, dtype=torch.float64),
            "feature_names": get_feature_names(),
            "window_size": int(config["window_size"]),
            "stride": int(config["stride"]),
        },
        path,
    )


def load_autoencoder(
    path: Path = MODEL_PATH, device: str = "cpu"
) -> tuple[Autoencoder, np.ndarray, np.ndarray]:
    """Load a checkpoint saved by ``save_autoencoder``.

    Returns:
        (model in eval mode, feature_mean, feature_std)
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = Autoencoder(ckpt["input_dim"], ckpt["hidden_dims"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["feature_mean"].cpu().numpy(), ckpt["feature_std"].cpu().numpy()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Train the autoencoder, save the best model, and save validation errors."""
    config = load_config()
    hidden_dims = [int(h) for h in config["ae_hidden_dims"]]
    epochs = int(config["ae_epochs"])
    batch_size = int(config["ae_batch_size"])
    lr = float(config["ae_lr"])
    seed = int(config["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Autoencoder training")
    print("=" * 60)
    print(f"Config: hidden_dims={hidden_dims}, epochs={epochs}, "
          f"batch_size={batch_size}, lr={lr}, seed={seed}, patience={PATIENCE}")
    print(f"Device: {device}")

    # 1. Load data and build windowed features
    print("\n[1/5] Loading processed data and building features...")
    df = load_processed()
    dataset = build_dataset(df, config)
    for split in ("train", "val"):
        if split not in dataset:
            raise RuntimeError(f"Split '{split}' not found. Re-run src.preprocess.")
    for split, (X, y, _) in dataset.items():
        print(f"  {split:5s} windows={len(X):5d}  features={X.shape[1]}  "
              f"anomalous windows={int(y.sum())}")

    # 2. Normal windows only for training and for early-stopping validation
    X_train_all, y_train_all, _ = dataset["train"]
    X_val_all, y_val_all, meta_val = dataset["val"]
    X_train = X_train_all[y_train_all == 0]
    X_val_normal = X_val_all[y_val_all == 0]
    print(f"\n[2/5] Normal windows: train {len(X_train)} of {len(X_train_all)}, "
          f"val {len(X_val_normal)} of {len(X_val_all)} (used for early stopping)")
    if len(X_train) == 0 or len(X_val_normal) == 0:
        raise RuntimeError("Not enough normal windows to train / validate.")

    # Feature standardization fitted on normal train windows only
    mean, std = fit_feature_stats(X_train)
    X_train_s = standardize(X_train, mean, std)
    X_val_normal_s = standardize(X_val_normal, mean, std)

    # 3. Train
    model = Autoencoder(X_train.shape[1], hidden_dims)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/5] Training  architecture: {model.architecture_string()}  "
          f"({n_params:,} parameters)")
    start = time.perf_counter()
    model, history = train_autoencoder(
        X_train_s, X_val_normal_s, hidden_dims, epochs, batch_size, lr, seed,
        patience=PATIENCE, device=device,
    )
    elapsed = time.perf_counter() - start
    print(f"  Finished in {elapsed:.1f}s | best epoch {history['best_epoch']} | "
          f"best val loss {min(history['val_loss']):.5f}")

    # 4. Save best model
    save_autoencoder(model, mean, std, config)
    print(f"\n[4/5] Saved best model -> {MODEL_PATH}")

    # 5. Validation reconstruction errors (all val windows, normal + anomalous)
    val_scores = reconstruction_errors(model, X_val_all, mean, std, device=device)
    scores_df = meta_val.copy()
    scores_df["score"] = val_scores
    scores_df["label"] = y_val_all
    scores_df.to_csv(VAL_SCORES_PATH, index=False)
    print(f"[5/5] Saved validation reconstruction errors -> {VAL_SCORES_PATH}")

    print("\nValidation reconstruction error summary (higher = more anomalous):")
    print(f"  windows : {len(val_scores)}")
    print(f"  min/mean/max : {val_scores.min():.4f} / {val_scores.mean():.4f} / "
          f"{val_scores.max():.4f}")
    if len(np.unique(y_val_all)) == 2:
        print(f"  mean error normal   : {val_scores[y_val_all == 0].mean():.4f}")
        print(f"  mean error anomalous: {val_scores[y_val_all == 1].mean():.4f}")
        print(f"  quick sanity ROC-AUC: {roc_auc_score(y_val_all, val_scores):.3f} "
              f"(full evaluation happens in src/evaluate.py)")
    print("Done.")


if __name__ == "__main__":
    main()