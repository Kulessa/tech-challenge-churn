"""Tests for the churn prediction API."""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.api.main import create_app
from src.preprocessing import build_preprocessor
from tests.test_preprocessing import make_customer_frame


def create_test_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fitted synthetic model and matching metadata."""
    frame = make_customer_frame()
    target = pd.Series([0, 1, 0, 1])
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(frame, target)

    model_path = tmp_path / "synthetic_model.joblib"
    metadata_path = tmp_path / "synthetic_metadata.json"
    joblib.dump(model, model_path)
    metadata_path.write_text(
        json.dumps({"threshold": 0.5, "training_date": "test-version"}),
        encoding="utf-8",
    )
    return model_path, metadata_path


def test_health_reports_model_loaded(tmp_path: Path) -> None:
    """Return healthy status when model artifacts load at startup."""
    model_path, metadata_path = create_test_artifacts(tmp_path)
    app = create_app(model_path, metadata_path)

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_expected_fields(tmp_path: Path) -> None:
    """Return a valid churn prediction for a synthetic customer."""
    model_path, metadata_path = create_test_artifacts(tmp_path)
    app = create_app(model_path, metadata_path)
    customer = make_customer_frame().iloc[0].to_dict()
    customer["TotalCharges"] = 30.0

    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 200
    assert set(response.json()) == {
        "churn_probability",
        "churn_prediction",
        "threshold",
        "model_version",
    }
    assert response.json()["model_version"] == "test-version"


def test_predict_rejects_invalid_customer_input(tmp_path: Path) -> None:
    """Return a validation error for invalid customer fields."""
    model_path, metadata_path = create_test_artifacts(tmp_path)
    app = create_app(model_path, metadata_path)
    customer = make_customer_frame().iloc[0].to_dict()
    customer["tenure"] = -1

    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_health_reports_model_missing(tmp_path: Path) -> None:
    """Return healthy API status and report a missing model artifact."""
    app = create_app(tmp_path / "missing.joblib", tmp_path / "missing.json")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False}


def test_predict_returns_service_unavailable_without_model(tmp_path: Path) -> None:
    """Return 503 when prediction is requested without a model."""
    app = create_app(tmp_path / "missing.joblib", tmp_path / "missing.json")
    customer = make_customer_frame().iloc[0].to_dict()

    with TestClient(app) as client:
        response = client.post("/predict", json=customer)

    assert response.status_code == 503
