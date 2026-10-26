"""Tests for churn prediction helpers."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.predict import predict_churn
from src.preprocessing import build_preprocessor
from tests.test_preprocessing import make_customer_frame


def make_fitted_pipeline() -> Pipeline:
    """Fit a lightweight model on synthetic customer data."""
    features = make_customer_frame()
    target = pd.Series([0, 1, 0, 1])
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    pipeline.fit(features, target)
    return pipeline


def test_prediction_probability_and_class_match_threshold() -> None:
    """Return a bounded score and a class consistent with its threshold."""
    customer = make_customer_frame().iloc[[0]]
    model = make_fitted_pipeline()

    result = predict_churn(model, customer, threshold=0.5)

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_prediction"] is (
        result["churn_probability"] >= result["threshold"]
    )


def test_prediction_rejects_multiple_rows() -> None:
    """Reject requests containing more than one customer."""
    customers = make_customer_frame().iloc[:2]

    with pytest.raises(ValueError, match="exactly one"):
        predict_churn(make_fitted_pipeline(), customers)


def test_prediction_rejects_invalid_threshold() -> None:
    """Reject thresholds outside the probability range."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        predict_churn(make_fitted_pipeline(), make_customer_frame().iloc[[0]], 1.5)
