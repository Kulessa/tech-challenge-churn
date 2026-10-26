"""Classification and subgroup evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> dict[str, float]:
    """Calculate binary classification metrics for positive churn."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_probability)),
    }


def subgroup_metrics(
    groups: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Calculate churn rates and classification metrics for each subgroup."""
    if len(groups) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("groups, y_true, and y_pred must have the same length.")

    results: list[dict[str, object]] = []
    for column in groups.columns:
        subgroup_data = groups[column].reset_index(drop=True)
        target = y_true.reset_index(drop=True)
        predictions = pd.Series(y_pred).reset_index(drop=True)
        for value in subgroup_data.dropna().unique():
            mask = subgroup_data.eq(value)
            subgroup_target = target.loc[mask]
            subgroup_predictions = predictions.loc[mask]
            true_positive = int(
                ((subgroup_target == 1) & (subgroup_predictions == 1)).sum()
            )
            false_positive = int(
                ((subgroup_target == 0) & (subgroup_predictions == 1)).sum()
            )
            false_negative = int(
                ((subgroup_target == 1) & (subgroup_predictions == 0)).sum()
            )
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            results.append(
                {
                    "feature": column,
                    "group": value,
                    "count": int(mask.sum()),
                    "actual_churn_rate": float(subgroup_target.mean()),
                    "predicted_churn_rate": float(subgroup_predictions.mean()),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
    return pd.DataFrame(results)
