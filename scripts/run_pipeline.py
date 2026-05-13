from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import pandas as pd

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.download_binance import build_bids_asks_from_depth, download_binance_vision_depth
from src.data.download_kraken import fetch_kraken_futures_data
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
from src.models.train_lightgbm import save_training_artifacts, train_walk_forward
from src.utils.io import ensure_dir, load_yaml, save_json, save_parquet
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

def load_or_download_raw(data_cfg: dict, force_download: bool = False) -> pd.DataFrame:
    symbol = data_cfg["symbol"]
    raw_path = ROOT / data_cfg["raw_dir"] / f"{symbol}_real.parquet"
    if raw_path.exists() and not force_download:
        LOGGER.info("Loading existing real raw data from %s", raw_path)
        return pd.read_parquet(raw_path)

    kraken_symbol = data_cfg["exchange_symbols"]["kraken_futures"][symbol]
    binance_symbol = data_cfg["exchange_symbols"]["binance_vision"][symbol]
    ohlcv = fetch_kraken_futures_data(kraken_symbol, limit=int(data_cfg["history_bars"]))
    depth = download_binance_vision_depth(ohlcv.index[0], ohlcv.index[-1], binance_symbol)
    if depth.empty:
        raise RuntimeError("No real Binance Vision depth data downloaded; refusing to continue with proxy order books")
    save_parquet(depth, ROOT / data_cfg["raw_dir"] / f"{symbol}_binance_book_depth.parquet")
    raw = build_bids_asks_from_depth(ohlcv, depth)
    save_parquet(raw, raw_path)
    return raw


def run(force_download: bool = False, balance_splits: bool = False, output_subdir: str | None = None) -> None:
    data_cfg = load_yaml(ROOT / "config/data.yaml")
    model_cfg = load_yaml(ROOT / "config/model.yaml")
    feature_cfg = load_yaml(ROOT / "config/feature_config.yaml")

    raw = load_or_download_raw(data_cfg, force_download=force_download)
    dataset, feature_cols = build_features(
        raw,
        price_windows=feature_cfg["price_windows"],
        orderbook_levels=feature_cfg["orderbook_levels"],
        target_horizon_bars=int(feature_cfg["target_horizon_bars"]),
    )
    dataset_path = ROOT / data_cfg["datasets_dir"] / f"{data_cfg['symbol']}_features.parquet"
    save_parquet(dataset, dataset_path)
    LOGGER.info("Saved feature dataset with %s rows and %s features to %s", len(dataset), len(feature_cols), dataset_path)

    models, predictions, validation_predictions, importance, shap_importance, balance_report = train_walk_forward(
        dataset,
        feature_cols,
        model_params=model_cfg["lightgbm"],
        split_config=model_cfg["walk_forward"],
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        balance_splits=balance_splits,
    )

    outputs = ROOT / "outputs"
    if output_subdir:
        outputs = outputs / output_subdir
    trained_feature_set = set(importance["feature"])
    trained_features = [col for col in feature_cols if col in trained_feature_set]
    save_training_artifacts(models, trained_features, outputs / "models")
    save_parquet(predictions, outputs / "predictions/test_predictions.parquet")
    save_parquet(validation_predictions, outputs / "predictions/validation_predictions.parquet")
    save_parquet(predictions[["prob_up"]], outputs / "predictions/probabilities.parquet")
    importance.to_csv(ensure_dir(outputs / "metrics") / "feature_importance.csv", index=False)
    shap_importance.to_csv(outputs / "metrics/shap_importance.csv", index=False)
    balance_report.to_csv(outputs / "metrics/split_class_balance.csv", index=False)

    backtest = build_strategy_returns(
        predictions,
        long_threshold=float(model_cfg["thresholds"]["long"]),
        short_threshold=float(model_cfg["thresholds"]["short"]),
        transaction_cost_bps=float(data_cfg["costs"]["transaction_cost_bps"]),
        slippage_bps=float(data_cfg["costs"]["slippage_bps"]),
    )
    save_parquet(backtest, outputs / "predictions/backtest.parquet")

    cls = classification_metrics(predictions["target"], predictions["prob_up"])
    cls.update(statistical_tests(predictions))
    save_json(cls, outputs / "metrics/classification_metrics.json")
    validation_cls = classification_metrics(validation_predictions["target"], validation_predictions["prob_up"])
    validation_cls.update(statistical_tests(validation_predictions))
    save_json(validation_cls, outputs / "metrics/validation_classification_metrics.json")
    save_json(financial_metrics(backtest), outputs / "metrics/financial_metrics.json")
    reg = regime_metrics(predictions)
    save_json(reg, outputs / "metrics/regime_metrics.json")

    first_fold = dataset.iloc[: int(model_cfg["walk_forward"]["train_bars"])]
    test_aligned = dataset.loc[predictions.index]
    psi = {
        col: population_stability_index(first_fold[col], test_aligned[col])
        for col in feature_cols[:100]
    }
    save_json(psi, outputs / "metrics/feature_psi.json")

    save_confusion_matrix(predictions["target"], predictions["prob_up"], outputs / "figures/confusion_matrix.png")
    save_prediction_actual_heatmap(predictions, outputs / "figures/prediction_actual_heatmap.png")
    save_confusion_matrix(
        validation_predictions["target"],
        validation_predictions["prob_up"],
        outputs / "figures/validation_confusion_matrix.png",
    )
    save_prediction_actual_heatmap(validation_predictions, outputs / "figures/validation_prediction_actual_heatmap.png")
    save_calibration_curve(
        validation_predictions["target"],
        validation_predictions["prob_up"],
        outputs / "figures/validation_calibration_curve.png",
    )
    save_feature_importance(importance, outputs / "figures/feature_importance.png")
    save_shap_importance(shap_importance, outputs / "figures/shap_importance.png")
    save_calibration_curve(predictions["target"], predictions["prob_up"], outputs / "figures/calibration_curve.png")
    save_pnl_curve(backtest, outputs / "figures/pnl_curve.png")
    save_regime_performance(reg, outputs / "figures/regime_performance.png")
    LOGGER.info("Pipeline complete. Outputs written under %s", outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC 15-minute direction prediction pipeline.")
    parser.add_argument("--force-download", action="store_true", help="Ignore cached parquet and download real data again.")
    parser.add_argument(
        "--balance-splits",
        action="store_true",
        help="Downsample each train/validation/test split to exactly 50%% up and 50%% down after chronological splitting.",
    )
    parser.add_argument(
        "--output-subdir",
        help="Write artifacts under outputs/<output-subdir> instead of overwriting the default outputs directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(force_download=args.force_download, balance_splits=args.balance_splits, output_subdir=args.output_subdir)
