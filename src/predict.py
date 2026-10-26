"""Load the saved model and generate churn predictions."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import METADATA_PATH, MODEL_PATH, THRESHOLD


@dataclass(frozen=True)
class LoadedModel:
    """Bundle containing a pipeline and its prediction metadata."""

    pipeline: Pipeline
    threshold: float
    model_version: str


def load_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> LoadedModel:
    """Load the persisted pipeline and its serving metadata."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata file not found: {metadata_path}")

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Saved model must be a scikit-learn Pipeline.")
    if not hasattr(pipeline, "predict_proba"):
        raise TypeError("Saved pipeline must support predict_proba.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = float(metadata.get("threshold", THRESHOLD))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Model threshold must be between 0 and 1: {threshold}")
    model_version = str(metadata.get("training_date", "unknown"))
    return LoadedModel(
        pipeline=pipeline,
        threshold=threshold,
        model_version=model_version,
    )


def predict_churn(
    model: Pipeline,
    customer: pd.DataFrame,
    threshold: float = THRESHOLD,
) -> dict[str, float | bool]:
    """Predict churn probability and class for one customer."""
    if len(customer) != 1:
        raise ValueError("Prediction expects exactly one customer row.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    probability = float(model.predict_proba(customer)[0, 1])
    return {
        "churn_probability": probability,
        "churn_prediction": probability >= threshold,
        "threshold": threshold,
    }
