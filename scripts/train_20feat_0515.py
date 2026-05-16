from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, log_loss, precision_recall_curve, roc_curve

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import classification_metrics, statistical_tests
from src.features.feature_pipeline import build_features
from src.utils.io import ensure_dir, load_yaml, save_json, save_parquet


MODEL_NAME = "20feat-0515"
FEATURE_COLS = [
    "volume",
    "high_low_range",
    "parkinson_volatility_1",
    "parkinson_volatility_3",
    "parkinson_volatility_5",
    "parkinson_volatility_15",
    "parkinson_volatility_30",
    "realized_volatility_1",
    "realized_vol_percentile",
    "rolling_entropy",
    "hurst_exponent",
    "rolling_return_3",
    "rolling_return_5",
    "rolling_return_15",
    "rolling_return_30",
    "vwap_distance",
    "close_open_range",
    "log_return",
    "session_asia",
    "session_europe",
]


def _load_raw(symbol: str) -> pd.DataFrame:
    path = ROOT / "data/raw" / f"{symbol}_real.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing cached raw data: {path}")
    return pd.read_parquet(path).sort_index()


def _build_feature_prefix(raw: pd.DataFrame, raw_end: int, feature_cfg: dict[str, Any]) -> pd.DataFrame:
    prefix = raw.iloc[:raw_end].copy()
    dataset, _ = build_features(
        prefix,
        price_windows=feature_cfg["price_windows"],
        orderbook_levels=feature_cfg["orderbook_levels"],
        target_horizon_bars=int(feature_cfg["target_horizon_bars"]),
    )
    missing = [col for col in FEATURE_COLS if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing requested features: {missing}")
    return dataset.dropna(subset=["target", "future_return_15m"]).copy()


def _fold_specs(n_labeled_rows: int) -> list[dict[str, int]]:
    train_step = 2_940
    val_bars = 294
    fold_specs: list[dict[str, int]] = []
    for fold in range(1, 17):
        train_end = fold * train_step
        val_end = train_end + val_bars
        if val_end > n_labeled_rows:
            break
        fold_specs.append({"fold": fold, "train_end": train_end, "val_end": val_end})

    final_train_end = min(49_680, n_labeled_rows - 1)
    if final_train_end > 0 and final_train_end < n_labeled_rows:
        fold_specs.append({"fold": 17, "train_end": final_train_end, "val_end": n_labeled_rows})

    if len(fold_specs) != 17:
        raise ValueError(f"Expected 17 usable folds, got {len(fold_specs)} for {n_labeled_rows} labeled rows")
    return fold_specs


def _balanced_binary_sample(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    counts = frame["target"].value_counts()
    if len(counts) != 2:
        raise ValueError(f"Cannot balance split with class counts: {counts.to_dict()}")
    target_size = int(counts.min())
    parts = [
        frame[frame["target"] == class_value].sample(n=target_size, random_state=seed, replace=False)
        for class_value in sorted(counts.index)
    ]
    return pd.concat(parts).sort_index()


def _balance_row(fold: int | str, split: str, frame: pd.DataFrame) -> dict[str, Any]:
    up = int(frame["target"].sum())
    down = int(len(frame) - up)
    return {
        "fold": fold,
        "split": split,
        "rows": int(len(frame)),
        "up": up,
        "down": down,
        "up_ratio": float(up / len(frame)) if len(frame) else np.nan,
        "start": str(frame.index.min()) if len(frame) else None,
        "end": str(frame.index.max()) if len(frame) else None,
    }


def _fit_model(
    train: pd.DataFrame,
    val: pd.DataFrame | None,
    model_params: dict[str, Any],
    early_stopping_rounds: int,
) -> LGBMClassifier:
    model = LGBMClassifier(**model_params)
    if val is None or val.empty:
        model.fit(train[FEATURE_COLS], train["target"])
        return model
    model.fit(
        train[FEATURE_COLS],
        train["target"],
        eval_set=[(val[FEATURE_COLS], val["target"])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def _sample_params(trial: optuna.Trial, base_params: dict[str, Any]) -> dict[str, Any]:
    params = dict(base_params)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 300, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 800),
            "subsample": trial.suggest_float("subsample", 0.50, 0.98),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
        }
    )
    params["objective"] = "binary"
    params["random_state"] = 42
    params["n_jobs"] = -1
    params["force_col_wise"] = True
    params["verbosity"] = -1
    return params


def _prepare_fold_frames(
    raw: pd.DataFrame,
    feature_cfg: dict[str, Any],
    fold_specs: list[dict[str, int]],
    seed: int,
) -> list[dict[str, Any]]:
    fold_frames: list[dict[str, Any]] = []
    for spec in fold_specs:
        fold = spec["fold"]
        print(f"Preparing fold {fold}/17 feature frame for Optuna", flush=True)
        dataset = _build_feature_prefix(raw, min(spec["val_end"] + 1, len(raw)), feature_cfg)
        train = dataset.iloc[: spec["train_end"]].copy()
        val = dataset.iloc[spec["train_end"]: spec["val_end"]].copy()
        balanced_train = _balanced_binary_sample(train, seed + fold)
        fold_frames.append({"spec": spec, "train": train, "balanced_train": balanced_train, "val": val})
    return fold_frames


def _save_optuna_plots(study: optuna.Study, output_dir: Path) -> None:
    figures_dir = ensure_dir(output_dir / "figures" / "optuna")
    metrics_dir = ensure_dir(output_dir / "metrics")
    completed = [trial for trial in study.trials if trial.value is not None]
    if completed:
        numbers = [trial.number for trial in completed]
        values = [trial.value for trial in completed]
        best_values = []
        best = float("inf")
        for value in values:
            best = min(best, value)
            best_values.append(best)
        plt.figure(figsize=(9, 4))
        plt.plot(numbers, values, ".", alpha=0.45, label="Trial")
        plt.plot(numbers, best_values, linewidth=2, label="Best so far")
        plt.xlabel("Trial")
        plt.ylabel("Mean validation log loss")
        plt.title("Optuna Optimization History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "optimization_history.png", dpi=160)
        plt.close()

    try:
        importances = optuna.importance.get_param_importances(study)
    except Exception:
        importances = {}
    if importances:
        frame = pd.DataFrame({"parameter": importances.keys(), "importance": importances.values()})
        frame.to_csv(metrics_dir / "optuna_param_importance.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.barh(frame["parameter"], frame["importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title("Optuna Parameter Importance")
        plt.tight_layout()
        plt.savefig(figures_dir / "param_importance.png", dpi=160)
        plt.close()


def _optimize_hyperparameters(
    fold_frames: list[dict[str, Any]],
    base_params: dict[str, Any],
    early_stopping_rounds: int,
    n_trials: int,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, base_params)
        losses: list[float] = []
        for position, frame in enumerate(fold_frames):
            train = frame["balanced_train"]
            val = frame["val"]
            model = _fit_model(train, val, params, early_stopping_rounds)
            prob_up = model.predict_proba(val[FEATURE_COLS])[:, 1]
            losses.append(float(log_loss(val["target"], prob_up, labels=[0, 1])))
            if position in {2, 5, 8, 11, 14, len(fold_frames) - 1}:
                partial = float(np.mean(losses))
                trial.report(partial, step=position)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        mean_loss = float(np.mean(losses))
        trial.set_user_attr("mean_validation_log_loss", mean_loss)
        trial.set_user_attr("fold_log_losses", losses)
        return mean_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=2),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    metrics_dir = ensure_dir(output_dir / "metrics")
    joblib.dump(study, metrics_dir / "optuna_study.pkl")
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
        metrics_dir / "optuna_trials.csv",
        index=False,
    )
    save_json(
        {
            "direction": "minimize",
            "objective": "mean_walk_forward_validation_log_loss",
            "n_trials": n_trials,
            "best_value": float(study.best_value),
            "best_params": study.best_params,
        },
        metrics_dir / "optuna_best.json",
    )
    _save_optuna_plots(study, output_dir)

    best_params = dict(base_params)
    best_params.update(study.best_params)
    best_params["objective"] = "binary"
    best_params["random_state"] = seed
    best_params["n_jobs"] = -1
    best_params["force_col_wise"] = True
    best_params["verbosity"] = -1
    return best_params


def _prediction_frame(fold: int, model: LGBMClassifier, val: pd.DataFrame) -> pd.DataFrame:
    out_cols = [
        "target",
        "future_return_15m",
        "session_asia",
        "session_europe",
    ]
    pred = val[out_cols].copy()
    pred["prob_up"] = model.predict_proba(val[FEATURE_COLS])[:, 1]
    pred["pred"] = (pred["prob_up"] >= 0.5).astype(int)
    pred["fold"] = fold
    return pred


def _metrics_row(name: str, pred: pd.DataFrame, evaluation_scope: str, in_sample: bool) -> dict[str, Any]:
    metrics = classification_metrics(pred["target"], pred["prob_up"])
    metrics.update(statistical_tests(pred))
    return {
        "model": name,
        "evaluation_scope": evaluation_scope,
        "in_sample": in_sample,
        "rows": int(len(pred)),
        "up": int(pred["target"].sum()),
        "down": int(len(pred) - pred["target"].sum()),
        "up_ratio": float(pred["target"].mean()),
        "primary_metric": "log_loss",
        **metrics,
    }


def _save_roc_curve(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 6))
    for name, pred in predictions.items():
        fpr, tpr, _ = roc_curve(pred["target"], pred["prob_up"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_precision_recall_curve(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 6))
    for name, pred in predictions.items():
        precision, recall, _ = precision_recall_curve(pred["target"], pred["prob_up"])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{name} AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_calibration(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 6))
    for name, pred in predictions.items():
        frac_pos, mean_pred = calibration_curve(pred["target"], pred["prob_up"], n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=name)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Realized UP frequency")
    plt.title("Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_probability_histogram(predictions: dict[str, pd.DataFrame], path: Path) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(8, 5))
    for name, pred in predictions.items():
        plt.hist(pred["prob_up"], bins=30, alpha=0.45, label=name, density=True)
    plt.xlabel("Predicted probability UP")
    plt.ylabel("Density")
    plt.title("Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_log_loss_bar(metrics: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    labels = metrics["model"] + "\n" + metrics["evaluation_scope"]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, metrics["log_loss"])
    plt.ylabel("Log loss")
    plt.title("Primary Evaluation Metric")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _save_confusion(pred: pd.DataFrame, title: str, path: Path) -> None:
    ensure_dir(path.parent)
    cm = confusion_matrix(pred["target"], (pred["prob_up"] >= 0.5).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Down", "Up"])
    ax.set_yticks([0, 1], labels=["Down", "Up"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_feature_importance(model: LGBMClassifier, title: str, path: Path) -> None:
    ensure_dir(path.parent)
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(importance["feature"], importance["gain"])
    plt.xlabel("LightGBM gain")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _evaluate_focused_models(
    full_dataset: pd.DataFrame,
    fold17_val: pd.DataFrame,
    fold17_model: LGBMClassifier,
    recent_models: list[LGBMClassifier],
    aggregate_model: LGBMClassifier,
    output_dir: Path,
) -> None:
    figures_dir = ensure_dir(output_dir / "figures" / "focused")
    metrics_dir = ensure_dir(output_dir / "metrics")
    predictions_dir = ensure_dir(output_dir / "predictions" / "focused")

    fold17_pred = _prediction_frame(17, fold17_model, fold17_val)
    ensemble_pred = fold17_val[["target", "future_return_15m", "session_asia", "session_europe"]].copy()
    ensemble_probs = [model.predict_proba(fold17_val[FEATURE_COLS])[:, 1] for model in recent_models]
    ensemble_pred["prob_up"] = np.mean(ensemble_probs, axis=0)
    ensemble_pred["pred"] = (ensemble_pred["prob_up"] >= 0.5).astype(int)
    ensemble_pred["fold"] = "13-17_equal_weight"

    aggregate_full_pred = full_dataset[["target", "future_return_15m", "session_asia", "session_europe"]].copy()
    aggregate_full_pred["prob_up"] = aggregate_model.predict_proba(full_dataset[FEATURE_COLS])[:, 1]
    aggregate_full_pred["pred"] = (aggregate_full_pred["prob_up"] >= 0.5).astype(int)
    aggregate_full_pred["fold"] = "aggregate_full_in_sample"

    aggregate_fold17_pred = fold17_val[["target", "future_return_15m", "session_asia", "session_europe"]].copy()
    aggregate_fold17_pred["prob_up"] = aggregate_model.predict_proba(fold17_val[FEATURE_COLS])[:, 1]
    aggregate_fold17_pred["pred"] = (aggregate_fold17_pred["prob_up"] >= 0.5).astype(int)
    aggregate_fold17_pred["fold"] = "aggregate_on_fold17_in_sample"

    save_parquet(fold17_pred, predictions_dir / "fold17_predictions.parquet")
    save_parquet(ensemble_pred, predictions_dir / "ensemble_13_17_equal_weight_predictions.parquet")
    save_parquet(aggregate_full_pred, predictions_dir / "aggregate_full_in_sample_predictions.parquet")
    save_parquet(aggregate_fold17_pred, predictions_dir / "aggregate_on_fold17_in_sample_predictions.parquet")

    comparison = {
        "fold17": fold17_pred,
        "ensemble_13_17_equal": ensemble_pred,
        "aggregate_on_fold17_in_sample": aggregate_fold17_pred,
    }
    _save_roc_curve(comparison, figures_dir / "roc_fold17_comparison.png")
    _save_precision_recall_curve(comparison, figures_dir / "precision_recall_fold17_comparison.png")
    _save_calibration(comparison, figures_dir / "calibration_fold17_comparison.png")
    _save_probability_histogram(comparison, figures_dir / "probability_histogram_fold17_comparison.png")
    _save_roc_curve({"aggregate_full_in_sample": aggregate_full_pred}, figures_dir / "roc_aggregate_full_in_sample.png")
    _save_precision_recall_curve(
        {"aggregate_full_in_sample": aggregate_full_pred},
        figures_dir / "precision_recall_aggregate_full_in_sample.png",
    )
    _save_calibration({"aggregate_full_in_sample": aggregate_full_pred}, figures_dir / "calibration_aggregate_full_in_sample.png")
    _save_probability_histogram(
        {"aggregate_full_in_sample": aggregate_full_pred},
        figures_dir / "probability_histogram_aggregate_full_in_sample.png",
    )

    _save_confusion(fold17_pred, "Fold 17", figures_dir / "confusion_fold17.png")
    _save_confusion(ensemble_pred, "Equal-Weight Folds 13-17", figures_dir / "confusion_ensemble_13_17_equal.png")
    _save_confusion(aggregate_fold17_pred, "Aggregate on Fold 17 In-Sample", figures_dir / "confusion_aggregate_on_fold17_in_sample.png")
    _save_confusion(aggregate_full_pred, "Aggregate Full In-Sample", figures_dir / "confusion_aggregate_full_in_sample.png")
    _save_feature_importance(fold17_model, "Fold 17 Feature Importance", figures_dir / "feature_importance_fold17.png")
    _save_feature_importance(
        aggregate_model,
        "Aggregate Feature Importance",
        figures_dir / "feature_importance_aggregate.png",
    )

    rows = [
        _metrics_row("fold17", fold17_pred, "fold17_validation_oos", False),
        _metrics_row("ensemble_13_17_equal_weight", ensemble_pred, "fold17_validation_oos", False),
        _metrics_row("aggregate", aggregate_fold17_pred, "fold17_validation_in_sample", True),
        _metrics_row("aggregate", aggregate_full_pred, "all_labeled_rows_in_sample", True),
    ]
    focused_metrics = pd.DataFrame(rows).sort_values(["in_sample", "log_loss"], ascending=[True, True])
    focused_metrics.to_csv(metrics_dir / "focused_model_metrics.csv", index=False)
    _save_log_loss_bar(focused_metrics, figures_dir / "log_loss_focused_models.png")


def run(output_subdir: str = MODEL_NAME, seed: int = 42, n_trials: int = 100) -> None:
    data_cfg = load_yaml(ROOT / "config/data.yaml")
    model_cfg = load_yaml(ROOT / "config/model.yaml")
    feature_cfg = load_yaml(ROOT / "config/feature_config.yaml")
    raw = _load_raw(data_cfg["symbol"])

    full_dataset = _build_feature_prefix(raw, len(raw), feature_cfg)
    n_labeled_rows = len(full_dataset)
    fold_specs = _fold_specs(n_labeled_rows)

    output_dir = ensure_dir(ROOT / "outputs" / output_subdir)
    models_dir = ensure_dir(output_dir / "models")
    metrics_dir = ensure_dir(output_dir / "metrics")
    predictions_dir = ensure_dir(output_dir / "predictions")

    base_params = dict(model_cfg["lightgbm"])
    early_stopping_rounds = int(model_cfg["early_stopping_rounds"])
    fold_frames = _prepare_fold_frames(raw, feature_cfg, fold_specs, seed)
    best_params = _optimize_hyperparameters(
        fold_frames=fold_frames,
        base_params=base_params,
        early_stopping_rounds=early_stopping_rounds,
        n_trials=n_trials,
        seed=seed,
        output_dir=output_dir,
    )
    fold_predictions: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    best_iterations: list[int] = []

    for frame in fold_frames:
        spec = frame["spec"]
        fold = spec["fold"]
        print(f"Retraining fold {fold}/17 with Optuna best params", flush=True)
        train = frame["train"]
        balanced_train = frame["balanced_train"]
        val = frame["val"]
        if len(val) != spec["val_end"] - spec["train_end"]:
            raise ValueError(f"Fold {fold} validation length mismatch: {len(val)}")

        balance_rows.extend(
            [
                _balance_row(fold, "train_raw", train),
                _balance_row(fold, "train_balanced", balanced_train),
                _balance_row(fold, "validation", val),
            ]
        )

        model = _fit_model(balanced_train, val, best_params, early_stopping_rounds)
        joblib.dump(
            {"model": model, "feature_cols": FEATURE_COLS, "fold": fold, "fold_spec": spec},
            models_dir / f"fold_{fold:02d}.pkl",
        )
        if getattr(model, "best_iteration_", None):
            best_iterations.append(int(model.best_iteration_))

        pred = _prediction_frame(fold, model, val)
        fold_predictions.append(pred)
        metrics = classification_metrics(pred["target"], pred["prob_up"])
        metrics.update(statistical_tests(pred))
        fold_metric_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train)),
                "train_balanced_rows": int(len(balanced_train)),
                "validation_rows": int(len(val)),
                "train_start": str(train.index.min()),
                "train_end": str(train.index.max()),
                "validation_start": str(val.index.min()),
                "validation_end": str(val.index.max()),
                "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                **metrics,
            }
        )

    predictions = pd.concat(fold_predictions).sort_index()
    save_parquet(predictions, predictions_dir / "validation_predictions.parquet")
    save_parquet(predictions[["prob_up"]], predictions_dir / "probabilities.parquet")
    pd.DataFrame(fold_metric_rows).to_csv(metrics_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(balance_rows).to_csv(metrics_dir / "split_class_balance.csv", index=False)

    aggregate_params = dict(best_params)
    if best_iterations:
        aggregate_params["n_estimators"] = max(1, int(np.median(best_iterations)))
    aggregate_train = _balanced_binary_sample(full_dataset, seed)
    print("Training aggregate deployment model", flush=True)
    aggregate_model = _fit_model(aggregate_train, None, aggregate_params, early_stopping_rounds)
    joblib.dump(
        {
            "model": aggregate_model,
            "feature_cols": FEATURE_COLS,
            "model_name": MODEL_NAME,
            "trained_rows_raw": int(len(full_dataset)),
            "trained_rows_balanced": int(len(aggregate_train)),
            "params": aggregate_params,
        },
        models_dir / "aggregate_model.pkl",
    )
    joblib.dump(
        {"models": [aggregate_model], "feature_cols": FEATURE_COLS, "model_name": MODEL_NAME, "kind": "aggregate"},
        models_dir / "lightgbm_model.pkl",
    )
    recent_models = [joblib.load(models_dir / f"fold_{fold:02d}.pkl")["model"] for fold in range(13, 18)]
    joblib.dump(
        {"models": recent_models, "feature_cols": FEATURE_COLS, "model_name": MODEL_NAME, "kind": "equal_weight_folds_13_17"},
        models_dir / "recent_5_fold_ensemble.pkl",
    )
    pd.Series(FEATURE_COLS, name="feature").to_csv(models_dir / "feature_list.csv", index=False)

    fold17_model = joblib.load(models_dir / "fold_17.pkl")["model"]
    fold17_spec = fold_specs[-1]
    fold17_dataset = _build_feature_prefix(raw, min(fold17_spec["val_end"] + 1, len(raw)), feature_cfg)
    fold17_val = fold17_dataset.iloc[fold17_spec["train_end"]:fold17_spec["val_end"]].copy()
    _evaluate_focused_models(
        full_dataset=full_dataset,
        fold17_val=fold17_val,
        fold17_model=fold17_model,
        recent_models=recent_models,
        aggregate_model=aggregate_model,
        output_dir=output_dir,
    )
    print("Saved focused model metrics and plots", flush=True)

    overall = classification_metrics(predictions["target"], predictions["prob_up"])
    overall.update(statistical_tests(predictions))
    save_json(overall, metrics_dir / "validation_classification_metrics.json")
    save_json(
        {
            "model_name": MODEL_NAME,
            "symbol": data_cfg["symbol"],
            "raw_rows": int(len(raw)),
            "labeled_rows": int(n_labeled_rows),
            "requested_rows": 50_000,
            "note": "A 1-bar-ahead target leaves the final raw bar unlabeled, so training used all honest labeled rows.",
            "fold_specs": fold_specs,
            "feature_cols": FEATURE_COLS,
            "optuna_trials": n_trials,
            "optuna_objective": "mean_walk_forward_validation_log_loss",
            "aggregate_balanced_rows": int(len(aggregate_train)),
            "aggregate_params": aggregate_params,
            "recent_inference_ensemble": "models/recent_5_fold_ensemble.pkl",
            "primary_evaluation_metric": "log_loss",
            "lightgbm_early_stopping_metric": "binary_logloss",
        },
        metrics_dir / "metadata.json",
    )

    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write(
            "# 20feat-0515\n\n"
            "Expanding-window walk-forward LightGBM model using the requested 20 features.\n\n"
            "Artifacts:\n"
            "- `models/fold_01.pkl` through `models/fold_17.pkl`: per-fold models.\n"
            "- `models/aggregate_model.pkl`: deployment model retrained on all labeled rows after 50/50 undersampling.\n"
            "- `models/lightgbm_model.pkl`: existing-loader-compatible alias for the aggregate deployment model.\n"
            "- `models/recent_5_fold_ensemble.pkl`: convenience ensemble containing folds 13-17 only.\n"
            "- `metrics/optuna_best.json`: best 100-trial Optuna hyperparameters by mean walk-forward validation log loss.\n"
            "- `metrics/optuna_trials.csv`: full Optuna trial table.\n"
            "- `predictions/validation_predictions.parquet`: out-of-sample fold predictions.\n"
            "- `metrics/fold_metrics.csv`: per-fold log-loss-led metrics.\n"
            "- `metrics/focused_model_metrics.csv`: log-loss-led stats for fold 17, aggregate, and folds 13-17 ensemble.\n"
            "- `figures/optuna/`: Optuna optimization history and parameter importance plots.\n"
            "- `figures/focused/`: ROC, precision-recall, calibration, probability, confusion, and importance plots.\n"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the 20feat-0515 expanding-window model.")
    parser.add_argument("--output-subdir", default=MODEL_NAME)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(output_subdir=args.output_subdir, seed=args.seed, n_trials=args.n_trials)
