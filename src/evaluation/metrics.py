from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import mutual_info_classif


def classification_metrics(y_true: pd.Series, prob_up: pd.Series) -> dict[str, float]:
    pred = (prob_up >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, prob_up) if y_true.nunique() == 2 else np.nan
    metrics["log_loss"] = log_loss(y_true, np.column_stack([1 - prob_up, prob_up]), labels=[0, 1])
    return {key: float(value) for key, value in metrics.items()}


def statistical_tests(df: pd.DataFrame) -> dict[str, float]:
    clean = df[["prob_up", "future_return_15m", "target"]].dropna()
    if len(clean) < 3:
        return {"pearson_corr": np.nan, "spearman_corr": np.nan, "mutual_information": np.nan, "information_coefficient": np.nan}
    pearson = pearsonr(clean["prob_up"], clean["future_return_15m"])[0]
    spearman = spearmanr(clean["prob_up"], clean["future_return_15m"])[0]
    mi = mutual_info_classif(clean[["prob_up"]], clean["target"], discrete_features=False, random_state=42)[0]
    return {
        "pearson_corr": float(pearson),
        "spearman_corr": float(spearman),
        "mutual_information": float(mi),
        "information_coefficient": float(spearman),
    }


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return float("nan")
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
    if len(quantiles) < 3:
        return 0.0
    expected_counts = np.histogram(expected, bins=quantiles)[0] / len(expected)
    actual_counts = np.histogram(actual, bins=quantiles)[0] / len(actual)
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)).sum())
