"""Leakage-safe preprocessing for churn model pipelines."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def coerce_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with TotalCharges converted to numeric values."""
    if "TotalCharges" not in data.columns:
        raise ValueError("Required column TotalCharges is missing.")
    cleaned_data = data.copy()
    cleaned_data["TotalCharges"] = pd.to_numeric(
        cleaned_data["TotalCharges"],
        errors="coerce",
    )
    return cleaned_data


def build_preprocessor() -> ColumnTransformer:
    """Build numeric and categorical preprocessing transformations."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
