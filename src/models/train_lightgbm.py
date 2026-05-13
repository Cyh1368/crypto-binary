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


PREDICTION_COLUMNS = [
    "target",
    "future_return_15m",
    "volatility_regime",
    "session_asia",
    "session_europe",
    "session_us",
]


def _scale_pos_weight(y: pd.Series) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return float(negatives / positives) if positives else 1.0


def _balanced_binary_sample(frame: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
    counts = frame[target_col].value_counts()
    if len(counts) != 2:
        raise ValueError(f"Cannot balance split with class counts: {counts.to_dict()}")
    target_size = int(counts.min())
    parts = []
    for class_value in sorted(counts.index):
        class_frame = frame[frame[target_col] == class_value]
        if len(class_frame) > target_size:
            positions = np.linspace(0, len(class_frame) - 1, target_size, dtype=int)
            class_frame = class_frame.iloc[positions]
        parts.append(class_frame)
    return pd.concat(parts).sort_index()


def _split_balance_row(fold: int, split_name: str, frame: pd.DataFrame) -> dict[str, int | float | str]:
    positives = int(frame["target"].sum())
    negatives = int(len(frame) - positives)
    return {
        "fold": fold,
        "split": split_name,
        "rows": int(len(frame)),
        "up": positives,
        "down": negatives,
        "up_ratio": float(positives / len(frame)) if len(frame) else np.nan,
        "down_ratio": float(negatives / len(frame)) if len(frame) else np.nan,
        "start": frame.index.min(),
        "end": frame.index.max(),
    }


def train_walk_forward(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    model_params: dict,
    split_config: dict,
    early_stopping_rounds: int,
    balance_splits: bool = False,
) -> tuple[list[LGBMClassifier], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usable_features = [col for col in feature_cols if not dataset[col].isna().all()]
    if not usable_features:
        raise ValueError("No usable numeric features found after excluding all-null columns")
    data = dataset.dropna(subset=["target"]).copy()
    splits = make_walk_forward_splits(len(data), **split_config)
    if not splits:
        raise ValueError("Not enough rows for walk-forward validation")

    models: list[LGBMClassifier] = []
    predictions: list[pd.DataFrame] = []
    validation_predictions: list[pd.DataFrame] = []
    importances: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []
    balance_rows: list[dict[str, int | float | str]] = []

    for fold, split in enumerate(splits):
        train = data.iloc[split.train_start:split.train_end]
        val = data.iloc[split.val_start:split.val_end]
        test = data.iloc[split.test_start:split.test_end]
        if balance_splits:
            train = _balanced_binary_sample(train)
            val = _balanced_binary_sample(val)
            test = _balanced_binary_sample(test)
        balance_rows.extend(
            [
                _split_balance_row(fold, "train", train),
                _split_balance_row(fold, "validation", val),
                _split_balance_row(fold, "test", test),
            ]
        )
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
        val_prob_up = model.predict_proba(val[usable_features])[:, 1]
        val_pred = val[PREDICTION_COLUMNS].copy()
        val_pred["prob_up"] = val_prob_up
        val_pred["fold"] = fold
        validation_predictions.append(val_pred)

        prob_up = model.predict_proba(test[usable_features])[:, 1]
        pred = test[PREDICTION_COLUMNS].copy()
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
    validation_predictions_df = pd.concat(validation_predictions).sort_index()
    importance_df = pd.concat(importances).groupby("feature", as_index=False, sort=False)[["gain", "split"]].mean()
    shap_df = pd.concat(shap_frames).groupby("feature", as_index=False, sort=False)["mean_abs_shap"].mean()
    balance_report = pd.DataFrame(balance_rows)
    return models, predictions_df, validation_predictions_df, importance_df, shap_df, balance_report


def save_training_artifacts(models: list[LGBMClassifier], feature_cols: list[str], model_dir: str | Path) -> None:
    model_dir = ensure_dir(model_dir)
    joblib.dump({"models": models, "feature_cols": feature_cols}, model_dir / "lightgbm_model.pkl")
    pd.Series(feature_cols, name="feature").to_csv(model_dir / "feature_list.csv", index=False)
