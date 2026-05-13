from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import joblib
import ccxt
import pandas as pd
import requests

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_pipeline import load_or_download_raw
from src.data.download_binance import build_bids_asks_from_depth, download_binance_vision_depth
from src.evaluation.diagnostics import regime_metrics
from src.evaluation.metrics import classification_metrics, population_stability_index, statistical_tests
from src.evaluation.plots import (
    save_calibration_curve,
    save_confusion_matrix,
    save_feature_importance,
    save_prediction_actual_heatmap,
    save_regime_performance,
    save_shap_importance,
)
from src.features.feature_pipeline import build_features
from src.models.train_lightgbm import save_training_artifacts, train_walk_forward
from src.utils.io import ensure_dir, load_yaml, save_json, save_parquet
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

ASSETS = ["ETH", "SOL", "XRP", "HYPE", "DOGE", "BNB"]
ASSET_SYMBOLS = {asset: f"{asset}_USDT" for asset in ASSETS}
KRAKEN_FUTURES = {
    "ETH_USDT": "ETH/USD:USD",
    "SOL_USDT": "SOL/USD:USD",
    "XRP_USDT": "XRP/USD:USD",
    "HYPE_USDT": "HYPE/USD:USD",
    "DOGE_USDT": "DOGE/USD:USD",
    "BNB_USDT": "BNB/USD:USD",
}
BINANCE_VISION = {symbol: symbol.replace("_", "") for symbol in ASSET_SYMBOLS.values()}


def _download_kline_day(day: pd.Timestamp, binance_symbol: str, timeout: int = 20) -> pd.DataFrame | None:
    day_str = day.strftime("%Y-%m-%d")
    url = (
        "https://data.binance.vision/data/futures/um/daily/klines/"
        f"{binance_symbol}/15m/{binance_symbol}-15m-{day_str}.zip"
    )
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            frame = pd.read_csv(zf.open(csv_name))
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        return frame[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception as exc:
        LOGGER.warning("Failed Binance Vision kline download for %s %s: %s", binance_symbol, day_str, exc)
        return None


def _download_binance_vision_klines(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    binance_symbol: str,
    max_workers: int = 12,
) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D")
    LOGGER.info("Downloading Binance Vision 15m klines for %s over %s days", binance_symbol, len(dates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(lambda day: _download_kline_day(day, binance_symbol), dates))
    frames = [frame for frame in rows if frame is not None and not frame.empty]
    time.sleep(0.1)
    if not frames:
        raise RuntimeError(f"No Binance Vision klines returned for {binance_symbol}")
    klines = pd.concat(frames, ignore_index=True)
    for col in ["open", "high", "low", "close", "volume"]:
        klines[col] = pd.to_numeric(klines[col], errors="coerce")
    return klines.drop_duplicates("timestamp").set_index("timestamp").sort_index()


def _add_kraken_funding(frame: pd.DataFrame, kraken_symbol: str) -> pd.DataFrame:
    exchange = ccxt.krakenfutures({"enableRateLimit": True})
    since_funding = int(frame.index[0].timestamp() * 1000)
    end_ms = int(frame.index[-1].timestamp() * 1000)
    funding_rows: list[dict[str, Any]] = []
    LOGGER.info("Fetching Kraken Futures funding history for %s", kraken_symbol)
    while since_funding < end_ms:
        batch = exchange.fetch_funding_rate_history(kraken_symbol, since=since_funding, limit=1000)
        if not batch:
            break
        funding_rows.extend(batch)
        since_funding = batch[-1]["timestamp"] + 1
        time.sleep(0.05)

    if not funding_rows:
        raise RuntimeError(f"No Kraken funding returned for {kraken_symbol}")
    funding = pd.DataFrame(funding_rows)
    funding["timestamp"] = pd.to_datetime(funding["timestamp"], unit="ms", utc=True)
    funding = funding.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    frame = frame.copy()
    frame["funding_rate"] = funding["fundingRate"].reindex(frame.index, method="ffill").fillna(0.0)
    return frame


def _load_or_download_asset_raw(data_cfg: dict[str, Any], force_download: bool = False) -> pd.DataFrame:
    try:
        return load_or_download_raw(data_cfg, force_download=force_download)
    except RuntimeError as exc:
        symbol = data_cfg["symbol"]
        if symbol != "HYPE_USDT" or "No Kraken Futures OHLCV returned" not in str(exc):
            raise
        raw_path = ROOT / data_cfg["raw_dir"] / f"{symbol}_real.parquet"
        if raw_path.exists() and not force_download:
            LOGGER.info("Loading existing real raw data from %s", raw_path)
            return pd.read_parquet(raw_path)

        kraken_symbol = data_cfg["exchange_symbols"]["kraken_futures"][symbol]
        binance_symbol = data_cfg["exchange_symbols"]["binance_vision"][symbol]
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(minutes=15 * int(data_cfg["history_bars"]))
        ohlcv = _download_binance_vision_klines(start, end, binance_symbol).tail(int(data_cfg["history_bars"]))
        ohlcv = _add_kraken_funding(ohlcv, kraken_symbol)
        depth = download_binance_vision_depth(ohlcv.index[0], ohlcv.index[-1], binance_symbol)
        if depth.empty:
            raise RuntimeError("No real Binance Vision depth data downloaded; refusing to continue")
        save_parquet(depth, ROOT / data_cfg["raw_dir"] / f"{symbol}_binance_book_depth.parquet")
        raw = build_bids_asks_from_depth(ohlcv, depth)
        save_parquet(raw, raw_path)
        return raw


def _metric_row(dataset_name: str, predictions: pd.DataFrame) -> dict[str, Any]:
    metrics = classification_metrics(predictions["target"], predictions["prob_up"])
    metrics.update(statistical_tests(predictions))
    return {
        "dataset": dataset_name,
        "rows": int(len(predictions)),
        "up_ratio": float(predictions["target"].mean()),
        "down_ratio": float(1.0 - predictions["target"].mean()),
        "always_down_accuracy": float((predictions["target"] == 0).mean()),
        "always_up_accuracy": float((predictions["target"] == 1).mean()),
        **metrics,
    }


def _regime_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in ["volatility_regime", "session_asia", "session_europe", "session_us"]:
        if col not in predictions:
            continue
        for value, group in predictions.groupby(col, dropna=False, observed=False):
            if len(group) < 50 or group["target"].nunique() < 2:
                continue
            metrics = classification_metrics(group["target"], group["prob_up"])
            rows.append(
                {
                    "regime": f"{col}={value}",
                    "rows": int(len(group)),
                    "up_ratio": float(group["target"].mean()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)


def _markdown_metric_table(rows: pd.DataFrame) -> str:
    lines = [
        "| Dataset | Rows | UP ratio | Accuracy | Balanced accuracy | ROC AUC | F1 | Precision | Recall | MCC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            "| {dataset} | {rows:,} | {up_ratio:.4f} | {accuracy:.4f} | {balanced_accuracy:.4f} | "
            "{roc_auc:.4f} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {mcc:.4f} |".format(**row._asdict())
        )
    return "\n".join(lines)


def _markdown_regime_table(regimes: pd.DataFrame) -> str:
    lines = [
        "| Regime | Rows | UP ratio | Accuracy | Balanced accuracy | ROC AUC | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in regimes.itertuples(index=False):
        lines.append(
            "| {regime} | {rows:,} | {up_ratio:.4f} | {accuracy:.4f} | {balanced_accuracy:.4f} | "
            "{roc_auc:.4f} | {f1:.4f} |".format(**row._asdict())
        )
    return "\n".join(lines)


def _top_feature_table(path: Path, value_col: str, label: str) -> str:
    frame = pd.read_csv(path).sort_values(value_col, ascending=False).head(10)
    lines = [f"| Feature | {label} |", "| --- | ---: |"]
    for row in frame.itertuples(index=False):
        lines.append(f"| `{row.feature}` | {getattr(row, value_col):.5f} |")
    return "\n".join(lines)


def _write_report(
    asset: str,
    symbol: str,
    output_dir: Path,
    feature_cols: list[str],
    split_config: dict[str, Any],
    model_params: dict[str, Any],
    metric_rows: pd.DataFrame,
    test_regimes: pd.DataFrame,
    validation_regimes: pd.DataFrame,
    raw_rows: int,
    dataset_rows: int,
) -> None:
    best_test = test_regimes.iloc[0]
    best_validation = validation_regimes.iloc[0]
    report = f"""# {asset} 15-Minute Direction Model

This folder contains a balanced LightGBM direction model for `{symbol}`. It uses the same 43 feature columns, LightGBM hyperparameters, walk-forward split configuration, balanced train/validation/test sampling, and evaluation metric suite as the latest BTC balanced model.

## Files

- `models/lightgbm_model.pkl`: saved walk-forward LightGBM model ensemble.
- `models/feature_list.csv`: ordered model feature list copied from the BTC balanced model.
- `predictions/test_predictions.parquet`: balanced walk-forward test predictions.
- `predictions/validation_predictions.parquet`: balanced validation predictions.
- `metrics/classification_metrics.json`: test classification metrics.
- `metrics/validation_classification_metrics.json`: validation classification metrics.
- `metrics/regime_metrics.csv`: test metrics split by volatility and trading-session regimes.
- `metrics/validation_regime_metrics.csv`: validation metrics split by volatility and trading-session regimes.
- `figures/validation_calibration_curve.png`: validation calibration curve.
- `figures/validation_confusion_matrix.png`: validation confusion matrix.

## Data

- Raw aligned rows: {raw_rows:,}
- Feature dataset rows: {dataset_rows:,}
- Model features: {len(feature_cols)}
- Target: `1` means {asset} closes higher over the next 15-minute bar; `0` means flat/down.

The class balance report is saved at `metrics/split_class_balance.csv`. Each train, validation, and test split is balanced independently after chronological splitting to avoid cross-contamination.

## Model Architecture

LightGBM parameters:

```json
{json.dumps(model_params, indent=2, sort_keys=True)}
```

Walk-forward split:

```json
{json.dumps(split_config, indent=2, sort_keys=True)}
```

## Performance

{_markdown_metric_table(metric_rows)}

## Regime Performance

Test regimes:

{_markdown_regime_table(test_regimes)}

Validation regimes:

{_markdown_regime_table(validation_regimes)}

Best test regime by balanced accuracy: `{best_test.regime}` with balanced accuracy {best_test.balanced_accuracy:.4f} and ROC AUC {best_test.roc_auc:.4f}.

Best validation regime by balanced accuracy: `{best_validation.regime}` with balanced accuracy {best_validation.balanced_accuracy:.4f} and ROC AUC {best_validation.roc_auc:.4f}.

## Feature Importance

Top features by mean absolute SHAP:

{_top_feature_table(output_dir / "metrics/shap_importance.csv", "mean_abs_shap", "Mean abs SHAP")}

Top features by LightGBM gain:

{_top_feature_table(output_dir / "metrics/feature_importance.csv", "gain", "Gain")}

## Validation Figures

![Validation calibration curve](figures/validation_calibration_curve.png)

![Validation confusion matrix](figures/validation_confusion_matrix.png)
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def train_asset(asset: str, force_download: bool = False) -> dict[str, Any]:
    symbol = ASSET_SYMBOLS[asset]
    data_cfg = load_yaml(ROOT / "config/data.yaml")
    model_cfg = load_yaml(ROOT / "config/model.yaml")
    feature_cfg = load_yaml(ROOT / "config/feature_config.yaml")

    data_cfg = dict(data_cfg)
    data_cfg["symbol"] = symbol
    data_cfg["exchange_symbols"] = {
        "kraken_futures": {symbol: KRAKEN_FUTURES[symbol]},
        "binance_vision": {symbol: BINANCE_VISION[symbol]},
    }

    output_dir = ensure_dir(ROOT / asset)
    btc_feature_cols = pd.read_csv(ROOT / "outputs/balanced_50_50/models/feature_list.csv")["feature"].tolist()

    raw = _load_or_download_asset_raw(data_cfg, force_download=force_download)
    save_parquet(raw, output_dir / "data/raw.parquet")

    dataset, generated_features = build_features(
        raw,
        price_windows=feature_cfg["price_windows"],
        orderbook_levels=feature_cfg["orderbook_levels"],
        target_horizon_bars=int(feature_cfg["target_horizon_bars"]),
    )
    missing_features = [col for col in btc_feature_cols if col not in dataset.columns]
    if missing_features:
        raise RuntimeError(f"{asset} is missing BTC balanced model features: {missing_features}")
    dataset = dataset.dropna(subset=btc_feature_cols + ["target", "future_return_15m"]).copy()
    save_parquet(dataset, output_dir / "data/features.parquet")
    generated_set = set(generated_features)
    if any(col not in generated_set for col in btc_feature_cols):
        LOGGER.warning("%s feature list differs from generated numeric feature list; using BTC balanced order", asset)

    models, predictions, validation_predictions, importance, shap_importance, balance_report = train_walk_forward(
        dataset,
        btc_feature_cols,
        model_params=model_cfg["lightgbm"],
        split_config=model_cfg["walk_forward"],
        early_stopping_rounds=int(model_cfg["early_stopping_rounds"]),
        balance_splits=True,
    )

    save_training_artifacts(models, btc_feature_cols, output_dir / "models")
    save_parquet(predictions, output_dir / "predictions/test_predictions.parquet")
    save_parquet(validation_predictions, output_dir / "predictions/validation_predictions.parquet")
    save_parquet(predictions[["prob_up"]], output_dir / "predictions/probabilities.parquet")
    importance.to_csv(ensure_dir(output_dir / "metrics") / "feature_importance.csv", index=False)
    shap_importance.to_csv(output_dir / "metrics/shap_importance.csv", index=False)
    balance_report.to_csv(output_dir / "metrics/split_class_balance.csv", index=False)

    test_metrics = _metric_row("test", predictions)
    validation_metrics = _metric_row("validation", validation_predictions)
    save_json({k: v for k, v in test_metrics.items() if k != "dataset"}, output_dir / "metrics/classification_metrics.json")
    save_json(
        {k: v for k, v in validation_metrics.items() if k != "dataset"},
        output_dir / "metrics/validation_classification_metrics.json",
    )
    metric_rows = pd.DataFrame([test_metrics, validation_metrics])
    metric_rows.to_csv(output_dir / "metrics/model_comparison.csv", index=False)

    test_regimes = _regime_table(predictions)
    validation_regimes = _regime_table(validation_predictions)
    test_regimes.to_csv(output_dir / "metrics/regime_metrics.csv", index=False)
    validation_regimes.to_csv(output_dir / "metrics/validation_regime_metrics.csv", index=False)
    save_json(regime_metrics(predictions), output_dir / "metrics/regime_metrics.json")
    save_json(regime_metrics(validation_predictions), output_dir / "metrics/validation_regime_metrics.json")

    first_fold = dataset.iloc[: int(model_cfg["walk_forward"]["train_bars"])]
    test_aligned = dataset.loc[predictions.index]
    psi = {
        col: population_stability_index(first_fold[col], test_aligned[col])
        for col in btc_feature_cols
    }
    save_json(psi, output_dir / "metrics/feature_psi.json")

    save_confusion_matrix(predictions["target"], predictions["prob_up"], output_dir / "figures/confusion_matrix.png")
    save_prediction_actual_heatmap(predictions, output_dir / "figures/prediction_actual_heatmap.png")
    save_confusion_matrix(
        validation_predictions["target"],
        validation_predictions["prob_up"],
        output_dir / "figures/validation_confusion_matrix.png",
    )
    save_prediction_actual_heatmap(
        validation_predictions,
        output_dir / "figures/validation_prediction_actual_heatmap.png",
    )
    save_calibration_curve(
        validation_predictions["target"],
        validation_predictions["prob_up"],
        output_dir / "figures/validation_calibration_curve.png",
    )
    save_calibration_curve(predictions["target"], predictions["prob_up"], output_dir / "figures/calibration_curve.png")
    save_feature_importance(importance, output_dir / "figures/feature_importance.png")
    save_shap_importance(shap_importance, output_dir / "figures/shap_importance.png")
    save_regime_performance(regime_metrics(predictions), output_dir / "figures/regime_performance.png")

    _write_report(
        asset=asset,
        symbol=symbol,
        output_dir=output_dir,
        feature_cols=btc_feature_cols,
        split_config=model_cfg["walk_forward"],
        model_params=model_cfg["lightgbm"],
        metric_rows=metric_rows,
        test_regimes=test_regimes,
        validation_regimes=validation_regimes,
        raw_rows=len(raw),
        dataset_rows=len(dataset),
    )

    summary = {
        "asset": asset,
        "symbol": symbol,
        "rows": int(metric_rows.loc[metric_rows["dataset"] == "test", "rows"].iloc[0]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_roc_auc": float(test_metrics["roc_auc"]),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_balanced_accuracy": float(validation_metrics["balanced_accuracy"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "best_test_regime": str(test_regimes.iloc[0]["regime"]),
        "best_test_regime_balanced_accuracy": float(test_regimes.iloc[0]["balanced_accuracy"]),
        "best_validation_regime": str(validation_regimes.iloc[0]["regime"]),
        "best_validation_regime_balanced_accuracy": float(validation_regimes.iloc[0]["balanced_accuracy"]),
    }
    save_json(summary, output_dir / "metrics/summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train balanced non-BTC asset models.")
    parser.add_argument("--assets", nargs="+", default=ASSETS, choices=ASSETS)
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for asset in args.assets:
        LOGGER.info("Training %s", asset)
        summaries.append(train_asset(asset, force_download=args.force_download))
    pd.DataFrame(summaries).to_csv(ROOT / "asset_model_comparison.csv", index=False)
    LOGGER.info("Wrote %s", ROOT / "asset_model_comparison.csv")


if __name__ == "__main__":
    main()
