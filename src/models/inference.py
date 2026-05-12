from __future__ import annotations

import joblib
import pandas as pd


def load_model(path: str):
    return joblib.load(path)


def predict_probabilities(model, features: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(features)[:, 1], index=features.index, name="prob_up")
