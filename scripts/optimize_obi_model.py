from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import lightgbm as lgb
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, roc_auc_score

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import load_or_download_raw
from src.evaluation.backtest import build_strategy_returns, financial_metrics
from src.evaluation.diagnostics import regime_metrics
from src.evaluation.metrics import classification_metrics, population_stability_index, statistical_tests
from src.evaluation.plots import (
    save_calibration_curve,
    save_confusion_matrix,
    save_feature_importance,
    save_pnl_curve,
    save_prediction_actual_heatmap,
    save_regime_performance,
    save_shap_importance,
)
from src.features.feature_pipeline import build_features
from src.models.train_lightgbm import (
    PREDICTION_COLUMNS,
    _balanced_binary_sample,
    save_training_artifacts,
    train_walk_forward,
)
from src.models.walk_forward import make_walk_forward_splits
from src.utils.io import ensure_dir, load_yaml, save_json, save_parquet
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _sample_params(trial: optuna.Trial, base_params: dict) -> dict:
    params = dict(base_params)
    params.update(
        {
            "n_estimators": trial.suggest_int("n_estimators", 400, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 1000),
            "subsample": trial.suggest_float("subsample", 0.50, 0.95),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.35, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 10.0, log=True),
        }
    )
    params["random_state"] = 42
    params["n_jobs"] = -1
    params["force_col_wise"] = True
    params["verbosity"] = -1
    params["objective"] = "binary"
    return params


def _objective_factory(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    base_params: dict,
    split_config: dict,
    early_stopping_rounds: int,
):
    data = dataset.dropna(subset=["target"]).copy()
    splits = make_walk_forward_splits(len(data), **split_config)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, base_params)
        train_probs: list[pd.Series] = []
        train_targets: list[pd.Series] = []
        val_probs: list[pd.Series] = []
        val_targets: list[pd.Series] = []

        for fold, split in enumerate(splits):
            train = _balanced_binary_sample(data.iloc[split.train_start:split.train_end])
            val = _balanced_binary_sample(data.iloc[split.val_start:split.val_end])
            model = LGBMClassifier(**params)
            model.fit(
                train[feature_cols],
                train["target"],
                eval_set=[(val[feature_cols], val["target"])],
                eval_metric="binary_logloss",
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
            train_probs.append(pd.Series(model.predict_proba(train[feature_cols])[:, 1], index=train.index))
            train_targets.append(train["target"])
            val_probs.append(pd.Series(model.predict_proba(val[feature_cols])[:, 1], index=val.index))
            val_targets.append(val["target"])

            if fold in {4, 9, 14, len(splits) - 1}:
                partial_val = pd.concat(val_probs)
                partial_target = pd.concat(val_targets)
                trial.report(roc_auc_score(partial_target, partial_val), step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        train_prob = pd.concat(train_probs)
        train_target = pd.concat(train_targets)
        val_prob = pd.concat(val_probs)
        val_target = pd.concat(val_targets)
        val_auc = roc_auc_score(val_target, val_prob)
        train_auc = roc_auc_score(train_target, train_prob)
        val_log_loss = log_loss(val_target, val_prob, labels=[0, 1])
        gap_penalty = max(0.0, train_auc - val_auc) * 0.20
        trial.set_user_attr("train_auc", float(train_auc))
        trial.set_user_attr("validation_auc", float(val_auc))
        trial.set_user_attr("validation_log_loss", float(val_log_loss))
        trial.set_user_attr("auc_gap_penalty", float(gap_penalty))
        return float(val_auc - gap_penalty)

    return objective


def _save_train_predictions(
    dataset: pd.DataFrame,
    models: list[LGBMClassifier],
    feature_cols: list[str],
    split_config: dict,
) -> pd.DataFrame:
    data = dataset.dropna(subset=["target"]).copy()
    splits = make_walk_forward_splits(len(data), **split_config)
    frames = []
    for fold, (model, split) in enumerate(zip(models, splits, strict=True)):
        train = _balanced_binary_sample(data.iloc[split.train_start:split.train_end])
        pred = train[PREDICTION_COLUMNS].copy()
        pred["prob_up"] = model.predict_proba(train[feature_cols])[:, 1]
        pred["fold"] = fold
        frames.append(pred)
    return pd.concat(frames).sort_index()


def _save_split_artifacts(
    outputs: Path,
    split_name: str,
    predictions: pd.DataFrame,
    model_cfg: dict,
    data_cfg: dict,
) -> dict[str, float]:
    metrics = classification_metrics(predictions["target"], predictions["prob_up"])
    metrics.update(statistical_tests(predictions))
    save_json(metrics, outputs / f"metrics/{split_name}_classification_metrics.json")

    by_fold = []
    for fold, group in predictions.groupby("fold"):
        fold_metrics = classification_metrics(group["target"], group["prob_up"])
        by_fold.append({"fold": int(fold), "rows": int(len(group)), **fold_metrics})
    pd.DataFrame(by_fold).to_csv(outputs / f"metrics/{split_name}_by_fold.csv", index=False)

    backtest = build_strategy_returns(
        predictions,
        long_threshold=float(model_cfg["thresholds"]["long"]),
        short_threshold=float(model_cfg["thresholds"]["short"]),
        transaction_cost_bps=float(data_cfg["costs"]["transaction_cost_bps"]),
        slippage_bps=float(data_cfg["costs"]["slippage_bps"]),
    )
    save_parquet(backtest, outputs / f"predictions/{split_name}_backtest.parquet")
    save_json(financial_metrics(backtest), outputs / f"metrics/{split_name}_financial_metrics.json")

    prefix = "" if split_name == "test" else f"{split_name}_"
    save_confusion_matrix(predictions["target"], predictions["prob_up"], outputs / f"figures/{prefix}confusion_matrix.png")
    save_calibration_curve(predictions["target"], predictions["prob_up"], outputs / f"figures/{prefix}calibration_curve.png")
    save_prediction_actual_heatmap(predictions, outputs / f"figures/{prefix}prediction_actual_heatmap.png")
    save_pnl_curve(backtest, outputs / f"figures/{prefix}pnl_curve.png")
    return metrics


def _save_optuna_plots(study: optuna.Study, outputs: Path) -> None:
    import matplotlib.pyplot as plt
    from optuna.importance import get_param_importances

    ensure_dir(outputs / "figures")
    trials = [trial for trial in study.trials if trial.value is not None]
    if trials:
        numbers = [trial.number for trial in trials]
        values = [trial.value for trial in trials]
        best_values = []
        best = float("-inf")
        for value in values:
            best = max(best, value)
            best_values.append(best)
        plt.figure(figsize=(9, 4))
        plt.plot(numbers, values, ".", alpha=0.45, label="Trial")
        plt.plot(numbers, best_values, linewidth=2, label="Best so far")
        plt.xlabel("Trial")
        plt.ylabel("Objective")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outputs / "figures/optuna_optimization_history.png", dpi=160)
        plt.close()

    importances = get_param_importances(study)
    if importances:
        frame = pd.DataFrame({"parameter": importances.keys(), "importance": importances.values()})
        frame.to_csv(outputs / "metrics/optuna_param_importance.csv", index=False)
        plt.figure(figsize=(8, 5))
        plt.barh(frame["parameter"], frame["importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(outputs / "figures/optuna_param_importance.png", dpi=160)
        plt.close()


def run(n_trials: int, output_subdir: str) -> None:
    data_cfg = load_yaml(ROOT / "config/data.yaml")
    model_cfg = load_yaml(ROOT / "config/model.yaml")
    feature_cfg = load_yaml(ROOT / "config/feature_config.yaml")
    outputs = ROOT / "outputs" / output_subdir

    raw = load_or_download_raw(data_cfg, force_download=False)
    dataset, feature_cols = build_features(
        raw,
        price_windows=feature_cfg["price_windows"],
        orderbook_levels=feature_cfg["orderbook_levels"],
        target_horizon_bars=int(feature_cfg["target_horizon_bars"]),
    )
    save_parquet(dataset, ROOT / data_cfg["datasets_dir"] / f"{data_cfg['symbol']}_features.parquet")
    LOGGER.info("Built dataset with %s rows and %s features", len(dataset), len(feature_cols))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=50, n_warmup_steps=2),
    )
    objective = _objective_factory(
        dataset=dataset,
        feature_cols=feature_cols,
        base_params=model_cfg["lightgbm"],
        split_config=model_cfg["walk_forward"],
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True)

    ensure_dir(outputs / "metrics")
    joblib.dump(study, outputs / "metrics/optuna_study.pkl")
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
        outputs / "metrics/optuna_trials.csv",
        index=False,
    )
    save_json({"best_value": study.best_value, "best_params": study.best_params}, outputs / "metrics/optuna_best.json")
    _save_optuna_plots(study, outputs)

    best_params = dict(model_cfg["lightgbm"])
    best_params.update(study.best_params)
    best_params["random_state"] = 42
    best_params["n_jobs"] = -1
    best_params["force_col_wise"] = True
    best_params["verbosity"] = -1
    best_params["objective"] = "binary"

    models, test_predictions, validation_predictions, importance, shap_importance, balance_report = train_walk_forward(
        dataset,
        feature_cols,
        model_params=best_params,
        split_config=model_cfg["walk_forward"],
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        balance_splits=True,
    )

    trained_feature_set = set(importance["feature"])
    trained_features = [col for col in feature_cols if col in trained_feature_set]
    save_training_artifacts(models, trained_features, outputs / "models")
    save_parquet(test_predictions, outputs / "predictions/test_predictions.parquet")
    save_parquet(validation_predictions, outputs / "predictions/validation_predictions.parquet")
    save_parquet(test_predictions[["prob_up"]], outputs / "predictions/probabilities.parquet")
    importance.to_csv(outputs / "metrics/feature_importance.csv", index=False)
    shap_importance.to_csv(outputs / "metrics/shap_importance.csv", index=False)
    balance_report.to_csv(outputs / "metrics/split_class_balance.csv", index=False)

    train_predictions = _save_train_predictions(
        dataset,
        models,
        trained_features,
        model_cfg["walk_forward"],
    )
    save_parquet(train_predictions, outputs / "predictions/train_predictions.parquet")

    split_rows = []
    for split_name, predictions in {
        "train": train_predictions,
        "validation": validation_predictions,
        "test": test_predictions,
    }.items():
        metrics = _save_split_artifacts(outputs, split_name, predictions, model_cfg, data_cfg)
        split_rows.append({"split": split_name, "rows": int(len(predictions)), **metrics})

    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(outputs / "metrics/split_metrics_summary.csv", index=False)
    save_json(split_summary.to_dict(orient="records"), outputs / "metrics/split_metrics_summary.json")

    test_backtest = pd.read_parquet(outputs / "predictions/test_backtest.parquet")
    save_parquet(test_backtest, outputs / "predictions/backtest.parquet")
    save_json(financial_metrics(test_backtest), outputs / "metrics/financial_metrics.json")

    reg = regime_metrics(test_predictions)
    save_json(reg, outputs / "metrics/regime_metrics.json")
    first_fold = dataset.iloc[: int(model_cfg["walk_forward"]["train_bars"])]
    test_aligned = dataset.loc[test_predictions.index]
    psi = {
        col: population_stability_index(first_fold[col], test_aligned[col])
        for col in feature_cols[:100]
    }
    save_json(psi, outputs / "metrics/feature_psi.json")

    save_feature_importance(importance, outputs / "figures/feature_importance.png")
    save_shap_importance(shap_importance, outputs / "figures/shap_importance.png")
    save_regime_performance(reg, outputs / "figures/regime_performance.png")
    LOGGER.info("Optuna best value: %.6f", study.best_value)
    LOGGER.info("Outputs written under %s", outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and train the OBI LightGBM model with Optuna.")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--output-subdir", default="obi-optuna-500")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_trials=args.n_trials, output_subdir=args.output_subdir)
