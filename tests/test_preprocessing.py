"""Tests for churn data preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.data import split_data
from src.preprocessing import build_preprocessor, coerce_total_charges


def make_customer_frame() -> pd.DataFrame:
    """Create a small synthetic dataset with all production features."""
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 12, 24, 48],
            "PhoneService": ["Yes"] * 4,
            "MultipleLines": ["No", "Yes", "No", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
            "OnlineSecurity": ["No", "Yes", "No internet service", "No"],
            "OnlineBackup": ["Yes", "No", "No internet service", "Yes"],
            "DeviceProtection": ["No", "Yes", "No internet service", "No"],
            "TechSupport": ["No", "Yes", "No internet service", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "Yes"],
            "StreamingMovies": ["No", "Yes", "No internet service", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Credit card (automatic)",
                "Bank transfer (automatic)",
            ],
            "MonthlyCharges": [30.0, 70.0, 20.0, 90.0],
            "TotalCharges": [30.0, 840.0, np.nan, 4320.0],
        }
    )


def test_coerce_total_charges_converts_blank_to_missing() -> None:
    """Convert blank charge strings to NaN without mutating the source."""
    source = pd.DataFrame({"TotalCharges": ["10.5", " ", "invalid"]})

    cleaned = coerce_total_charges(source)

    assert cleaned["TotalCharges"].isna().tolist() == [False, True, True]
    assert source["TotalCharges"].tolist() == ["10.5", " ", "invalid"]


def test_preprocessor_transforms_synthetic_data_without_missing_values() -> None:
    """Impute numeric missing values and return a fully numeric matrix."""
    frame = make_customer_frame()

    transformed = build_preprocessor().fit_transform(frame)

    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] == 44
    assert np.isfinite(transformed).all()


def test_preprocessor_ignores_unknown_categories() -> None:
    """Transform unseen categories without raising an error."""
    training = make_customer_frame()
    preprocessor = build_preprocessor().fit(training)
    new_customer = training.iloc[[0]].copy()
    new_customer.loc[:, "Contract"] = "Unseen contract"

    transformed = preprocessor.transform(new_customer)

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_split_data_is_stratified_and_reproducible() -> None:
    """Return the same stratified split for a fixed random state."""
    features = pd.DataFrame({"value": range(100)})
    target = pd.Series([0] * 75 + [1] * 25)

    first_split = split_data(features, target)
    second_split = split_data(features, target)

    assert first_split[0].index.tolist() == second_split[0].index.tolist()
    assert len(first_split[1]) == 20
    assert first_split[3].mean() == pytest.approx(0.25)
