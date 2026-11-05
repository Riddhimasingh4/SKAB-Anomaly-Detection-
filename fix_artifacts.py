"""One-off fix: re-save preprocess artifacts so the class is pickled as src.preprocess."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib

from src.preprocess import PreprocessArtifacts

ARTIFACTS_PATH = Path(__file__).resolve().parent / "models" / "preprocess_artifacts.joblib"


def main() -> None:
    """Load the old artifacts using a __main__ alias, then dump them again."""
    sys.modules["__main__"].PreprocessArtifacts = PreprocessArtifacts  # type: ignore[attr-defined]
    artifacts = joblib.load(ARTIFACTS_PATH)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"Re-saved {ARTIFACTS_PATH} as {type(artifacts).__module__}.{type(artifacts).__name__}")


if __name__ == "__main__":
    main()