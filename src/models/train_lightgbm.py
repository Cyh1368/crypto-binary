from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from src.models.walk_forward import make_walk_forward_splits
from src.utils.io import ensure_dir


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return float(negatives / positives) if positives else 1.0


def train_walk_forward(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict,
    split_config: dict,
    early_stopping_rounds: int,
) -> tuple[list[LGBMClassifier], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable_features = [col for col in feature_cols if not dataset[col].isna().all()]
    if not usable_features:
        raise ValueError("No usable numeric features found after excluding all-null columns")
    data = dataset.dropna(subset=["target"]).copy()
    splits = make_walk_forward_splits(len(data), **split_config)
    if not splits:
        raise ValueError("Not enough rows for walk-forward validation")

    models: list[LGBMClassifier] = []
    predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []

    for fold, split in enumerate(splits):
        train = data.iloc[split.train_start:split.train_end]
        val = data.iloc[split.val_start:split.val_end]
        test = data.iloc[split.test_start:split.test_end]
        params = dict(model_params)
        params["scale_pos_weight"] = _scale_pos_weight(train["target"])
        model = LGBMClassifier(**params)
        model.fit(
            train[usable_features],
            train["target"],
            eval_set=[(val[usable_features], val["target"])],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        prob_up = model.predict_proba(test[usable_features])[:, 1]
        pred = test[["target", "future_return_15m", "volatility_regime", "session_asia", "session_europe", "session_us"]].copy()
        pred["prob_up"] = prob_up
        pred["fold"] = fold
        predictions.append(pred)

        importances.append(
            pd.DataFrame(
                {
                    "feature": usable_features,
                    "gain": model.booster_.feature_importance(importance_type="gain"),
                    "split": model.booster_.feature_importance(importance_type="split"),
                    "fold": fold,
                }
            )
        )
        sample = test[usable_features].head(min(1000, len(test)))
        explainer = shap.TreeExplainer(model)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed.*",
                category=UserWarning,
            )
            shap_values = explainer.shap_values(sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[-1]
        shap_frames.append(
            pd.DataFrame(
                {
                    "feature": usable_features,
                    "mean_abs_shap": np.abs(shap_values).mean(axis=0),
                    "fold": fold,
                }
            )
        )
        models.append(model)

    predictions_df = pd.concat(predictions).sort_index()
    importance_df = pd.concat(importances).groupby("feature", as_index=False)[["gain", "split"]].mean()
    shap_df = pd.concat(shap_frames).groupby("feature", as_index=False)["mean_abs_shap"].mean()
    return models, predictions_df, importance_df, shap_df


def save_training_artifacts(models: list[LGBMClassifier], feature_cols: list[str], model_dir: str | Path) -> None:
    model_dir = ensure_dir(model_dir)
    joblib.dump({"models": models, "feature_cols": feature_cols}, model_dir / "lightgbm_model.pkl")
    pd.Series(feature_cols, name="feature").to_csv(model_dir / "feature_list.csv", index=False)
