from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import classification_metrics


def regime_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for col in ["volatility_regime", "session_asia", "session_europe", "session_us"]:
        if col not in predictions:
            continue
        for value, group in predictions.groupby(col, observed=False):
            if len(group) < 20 or group["target"].nunique() < 2:
                continue
            output[f"{col}={value}"] = classification_metrics(group["target"], group["prob_up"])
    return output
