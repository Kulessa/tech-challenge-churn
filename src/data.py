"""Dataset loading, cleaning, and splitting utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH,
    FEATURES,
    RANDOM_STATE,
    TARGET,
    TARGET_MAPPING,
    TEST_SIZE,
)
from src.preprocessing import coerce_total_charges


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw churn dataset from a CSV file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean predictors and encode the churn target."""
    required_columns = {"customerID", TARGET, *FEATURES}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned_data = coerce_total_charges(data)
    target = cleaned_data[TARGET].map(TARGET_MAPPING)
    if target.isna().any():
        invalid_values = sorted(cleaned_data.loc[target.isna(), TARGET].unique())
        raise ValueError(f"Unexpected target values: {invalid_values}")

    features = cleaned_data.loc[:, FEATURES].copy()
    target = target.astype("int64").rename(TARGET)
    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )
