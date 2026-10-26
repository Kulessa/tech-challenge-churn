"""Train and persist the selected churn model."""

import json
import platform
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import (
    DATA_PATH,
    FEATURES,
    METADATA_PATH,
    MODEL_NAME,
    MODEL_PARAMS,
    MODEL_PATH,
    RANDOM_STATE,
    THRESHOLD,
)
from src.data import clean_data, load_raw_data
from src.preprocessing import build_preprocessor


def build_model_pipeline() -> Pipeline:
    """Build the selected model and preprocessing pipeline."""
    classifier = LogisticRegression(
        C=MODEL_PARAMS["C"],
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict[str, Any]:
    """Fit the selected pipeline on available data and save its metadata."""
    raw_data = load_raw_data(data_path)
    features, target = clean_data(raw_data)
    pipeline = build_model_pipeline()
    pipeline.fit(features, target)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    previous_metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        loaded_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(loaded_metadata, dict):
            raise ValueError("Model metadata must contain a JSON object.")
        previous_metadata = loaded_metadata

    metadata: dict[str, Any] = {
        **previous_metadata,
        "model_name": MODEL_NAME,
        "best_parameters": {"classifier__C": MODEL_PARAMS["C"]},
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "positive_class": "Churn = Yes",
        "features": FEATURES,
        "training_date": date.today().isoformat(),
        "train_size": int(len(features)),
        "test_size": None,
        "evaluation_status": previous_metadata.get(
            "evaluation_status",
            "training_only; independent_test_set_unavailable",
        ),
        "cross_validation": previous_metadata.get("cross_validation"),
        "test_metrics": None,
        "model_size_bytes": int(model_path.stat().st_size),
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata


if __name__ == "__main__":
    trained_metadata = train_and_save()
    print(
        f"Modelo salvo em {MODEL_PATH} "
        f"({trained_metadata['model_size_bytes']} bytes)."
    )
    print(f"Metadados salvos em {METADATA_PATH}.")
    print("Métricas de teste não foram calculadas; o teste anterior não é independente.")
