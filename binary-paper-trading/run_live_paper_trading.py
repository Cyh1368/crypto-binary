from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ccxt
import joblib
import numpy as np
import pandas as pd
import yaml

from src.features.derivatives import add_derivatives_features
from src.features.orderflow import add_orderflow_features
from src.features.regime import add_regime_features
from src.features.technical import add_price_features
from src.features.volatility import add_volatility_regimes


PRICE_WINDOWS = [1, 3, 5, 15, 30, 60]
ORDERBOOK_LEVELS = [5, 10, 20, 50]
LIVE_OBI_HISTORY_BARS = 60
LIVE_OBI_HISTORY_COLUMNS = [
    "timestamp",
    "obi_raw_5",
    "obi_raw_10",
    "obi_raw_20",
    "bid_ask_depth_ratio_5",
    "bid_ask_depth_ratio_10",
    "bid_ask_depth_ratio_20",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(logs_dir: Path) -> logging.Logger:
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("BinaryPaperTrader")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(logs_dir / "paper_trading.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            if existing_fields != fieldnames:
                rows = list(reader)
                merged_fields = list(dict.fromkeys(existing_fields + fieldnames))
                with path.open("w", encoding="utf-8", newline="") as rewrite:
                    writer = csv.DictWriter(rewrite, fieldnames=merged_fields, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                fieldnames = merged_fields
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class LivePaperTrader:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = load_yaml(config_path)
        self.logs_dir = ROOT / self.config["logs_dir"]
        self.logger = setup_logger(self.logs_dir)
        self.exchange = ccxt.krakenfutures({"enableRateLimit": True})
        self.model_name = str(self.config.get("model_name") or Path(self.config["model_path"]).parents[1].name)
        artifact = joblib.load(ROOT / self.config["model_path"])
        self.models = artifact["models"]
        self.feature_cols = artifact["feature_cols"]
        self._validate_model_features()

    def _validate_model_features(self) -> None:
        if not self.feature_cols:
            raise RuntimeError("Model artifact does not contain feature columns")
        for idx, model in enumerate(self.models):
            booster_features = list(model.booster_.feature_name())
            if booster_features != list(self.feature_cols):
                raise RuntimeError(
                    f"Model fold {idx} feature order does not match artifact feature_cols; refusing live prediction"
                )

    def _obi_history_path(self) -> Path:
        return self.logs_dir / "live_obi_history.csv"

    def _load_obi_history(self) -> pd.DataFrame:
        path = self._obi_history_path()
        if not path.exists():
            return pd.DataFrame(columns=LIVE_OBI_HISTORY_COLUMNS)
        history = pd.read_csv(path)
        for col in LIVE_OBI_HISTORY_COLUMNS:
            if col not in history:
                history[col] = np.nan
        history = history[LIVE_OBI_HISTORY_COLUMNS]
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        history = history.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
        return history.sort_values("timestamp")

    def _save_obi_history(self, history: pd.DataFrame) -> None:
        path = self._obi_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        out = history.tail(LIVE_OBI_HISTORY_BARS).copy()
        out["timestamp"] = out["timestamp"].map(lambda value: pd.Timestamp(value).isoformat())
        out.to_csv(path, index=False)

    def _apply_live_obi_history(self, latest_ts: pd.Timestamp, latest_features: pd.Series) -> pd.Series:
        if not any(col.startswith("obi_") or col.startswith("bid_ask_depth_ratio_") for col in self.feature_cols):
            return latest_features

        snapshot: dict[str, Any] = {"timestamp": latest_ts}
        missing = []
        for col in LIVE_OBI_HISTORY_COLUMNS[1:]:
            value = latest_features.get(col)
            if pd.isna(value) or not np.isfinite(float(value)):
                missing.append(col)
            else:
                snapshot[col] = float(value)
        if missing:
            raise RuntimeError(f"Live order book snapshot is missing OBI inputs: {', '.join(missing)}")

        history = self._load_obi_history()
        history = history[history["timestamp"] != latest_ts]
        history = pd.concat([history, pd.DataFrame([snapshot])], ignore_index=True)
        history = history.sort_values("timestamp").tail(LIVE_OBI_HISTORY_BARS)
        self._save_obi_history(history)

        raw_5 = history["obi_raw_5"].astype(float)
        raw_10 = history["obi_raw_10"].astype(float)
        raw_20 = history["obi_raw_20"].astype(float)
        latest_features["obi_delta_5"] = float(raw_5.diff().iloc[-1]) if len(raw_5) > 1 else 0.0
        latest_features["obi_delta_10"] = float(raw_10.diff().iloc[-1]) if len(raw_10) > 1 else 0.0
        latest_features["obi_pressure_ratio"] = float(raw_5.iloc[-1] / (raw_20.iloc[-1] + 1e-8))

        for window in [3, 5, 15]:
            values = raw_5.tail(window)
            latest_features[f"obi_rolling_mean_{window}"] = float(values.mean())
            latest_features[f"obi_rolling_std_{window}"] = float(values.std()) if len(values) > 1 else 0.0

        z_values = raw_5.tail(LIVE_OBI_HISTORY_BARS)
        z_std = float(z_values.std()) if len(z_values) > 1 else 0.0
        latest_features["obi_zscore_5"] = float((raw_5.iloc[-1] - z_values.mean()) / (z_std + 1e-8)) if z_std else 0.0
        if len(history) < LIVE_OBI_HISTORY_BARS:
            self.logger.warning(
                "Live OBI history has %s/%s snapshots; using available history for rolling OBI inputs",
                len(history),
                LIVE_OBI_HISTORY_BARS,
            )
        return latest_features

    def fetch_market_frame(self) -> pd.DataFrame:
        symbol = self.config["symbol"]
        retry_attempts = int(self.config["loop"]["retry_attempts"])
        retry_sleep = float(self.config["loop"]["retry_sleep_seconds"])
        last_error: Exception | None = None
        for attempt in range(1, retry_attempts + 1):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=self.config["timeframe"],
                    limit=int(self.config["history_bars"]),
                )
                orderbook = self.exchange.fetch_order_book(symbol, limit=int(self.config["orderbook_limit"]))
                funding = self.exchange.fetch_funding_rate(symbol)
                return self._build_market_frame(ohlcv, orderbook, funding)
            except Exception as exc:
                last_error = exc
                self.logger.warning("Fetch attempt %s/%s failed: %s", attempt, retry_attempts, exc)
                if attempt < retry_attempts:
                    time.sleep(retry_sleep)
        raise RuntimeError(f"Failed to fetch live market data after {retry_attempts} attempts: {last_error}")

    def _build_market_frame(self, ohlcv: list[list[float]], orderbook: dict, funding: dict) -> pd.DataFrame:
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

        now = pd.Timestamp.now(tz="UTC")
        if len(df) > 1 and df.index[-1] + pd.Timedelta(minutes=15) > now:
            df = df.iloc[:-1].copy()
        if df.empty:
            raise RuntimeError("No closed OHLCV bars available")

        df["bids"] = [[] for _ in range(len(df))]
        df["asks"] = [[] for _ in range(len(df))]
        df["funding_rate"] = np.nan
        last_idx = df.index[-1]
        df.at[last_idx, "bids"] = [[float(p), float(q)] for p, q in orderbook.get("bids", [])]
        df.at[last_idx, "asks"] = [[float(p), float(q)] for p, q in orderbook.get("asks", [])]
        df.at[last_idx, "funding_rate"] = float(funding.get("fundingRate") or 0.0)
        df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        return df

    def build_live_features(self, frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Series, pd.Series]:
        features = frame.copy()
        features = add_price_features(features, PRICE_WINDOWS)
        features = add_orderflow_features(features, ORDERBOOK_LEVELS)
        features = add_derivatives_features(features)
        features = add_regime_features(features)
        features = add_volatility_regimes(features)
        latest_ts = features.index[-1]
        latest_features = features.loc[latest_ts].copy()
        latest_features = self._apply_live_obi_history(latest_ts, latest_features)
        model_row = latest_features.reindex(self.feature_cols).astype(float)
        missing = model_row[model_row.isna()]
        if not missing.empty:
            missing_cols = ", ".join(missing.index.tolist())
            raise RuntimeError(f"Live feature row is missing model inputs: {missing_cols}")
        if not np.isfinite(model_row.to_numpy(dtype=float)).all():
            bad_cols = model_row.index[~np.isfinite(model_row.to_numpy(dtype=float))].tolist()
            raise RuntimeError(f"Live feature row has non-finite model inputs: {', '.join(bad_cols)}")
        return latest_ts, latest_features, model_row

    def predict(self, model_row: pd.Series) -> dict[str, Any]:
        X = pd.DataFrame([model_row.to_dict()], columns=self.feature_cols)
        fold_probabilities = [float(model.predict_proba(X)[0, 1]) for model in self.models]
        prob_up = float(np.mean(fold_probabilities))
        prob_down = 1.0 - prob_up
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        confidence = max(prob_up, prob_down)
        return {
            "prob_up": prob_up,
            "prob_down": prob_down,
            "up_percent": prob_up * 100.0,
            "down_percent": prob_down * 100.0,
            "direction": direction,
            "confidence_percent": confidence * 100.0,
            "fold_probabilities": fold_probabilities,
        }

    def log_cycle(
        self,
        ts: pd.Timestamp,
        frame: pd.DataFrame,
        latest_features: pd.Series,
        prediction: dict[str, Any],
    ) -> None:
        contract_ts = ts + pd.Timedelta(minutes=15)
        raw = frame.loc[ts]
        bids = raw.get("bids", [])
        asks = raw.get("asks", [])
        best_bid = bids[0][0] if isinstance(bids, list) and bids else np.nan
        best_ask = asks[0][0] if isinstance(asks, list) and asks else np.nan
        price_row = {
            "timestamp": ts.isoformat(),
            "open": raw["open"],
            "high": raw["high"],
            "low": raw["low"],
            "close": raw["close"],
            "volume": raw["volume"],
            "funding_rate": raw["funding_rate"],
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread_bps": latest_features.get("spread_bps", np.nan),
            "log_return": latest_features.get("log_return", np.nan),
        }
        append_csv(
            self.logs_dir / "price_actions.csv",
            price_row,
            list(price_row.keys()),
        )

        prediction_row = {
            "timestamp": contract_ts.isoformat(),
            "model_input_timestamp": contract_ts.isoformat(),
            "close": raw["close"],
            "direction": prediction["direction"],
            "prob_up": prediction["prob_up"],
            "prob_down": prediction["prob_down"],
            "up_percent": prediction["up_percent"],
            "down_percent": prediction["down_percent"],
            "confidence_percent": prediction["confidence_percent"],
            "actual_close_15m": "",
            "actual_return_15m": "",
            "actual_direction_15m": "",
            "prediction_correct": "",
            "evaluated_at": "",
        }
        append_csv(self.logs_dir / "predictions.csv", prediction_row, list(prediction_row.keys()))

        feature_payload = {
            "timestamp": ts.isoformat(),
            "model_features": {col: json_default(latest_features.get(col)) for col in self.feature_cols},
            "all_features": {str(k): json_default(v) for k, v in latest_features.to_dict().items() if k not in {"bids", "asks"}},
            "fold_probabilities": prediction["fold_probabilities"],
        }
        with (self.logs_dir / "feature_snapshots.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(feature_payload, default=json_default, sort_keys=True) + "\n")

        self.logger.info(
            "BTC %s | contract=%s | model_input_candle_close=%s | model_input_close=%.2f | p_up=%.2f%% p_down=%.2f%% | signal=%s",
            prediction["direction"],
            contract_ts.isoformat(),
            contract_ts.isoformat(),
            raw["close"],
            prediction["up_percent"],
            prediction["down_percent"],
            prediction["direction"],
        )

    def evaluate_pending_predictions(self, frame: pd.DataFrame) -> None:
        predictions_path = self.logs_dir / "predictions.csv"
        if not predictions_path.exists():
            return

        predictions = pd.read_csv(
            predictions_path,
            dtype={
                "timestamp": object,
                "direction": object,
                "action": object,
                "actual_direction_15m": object,
                "prediction_correct": object,
                "evaluated_at": object,
            },
            keep_default_na=False,
        )
        required_cols = [
            "actual_close_15m",
            "actual_return_15m",
            "actual_direction_15m",
            "prediction_correct",
            "evaluated_at",
        ]
        for col in required_cols:
            if col not in predictions:
                predictions[col] = ""
            predictions[col] = predictions[col].astype(object)

        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True)
        frame = frame.sort_index()
        updates: list[dict[str, Any]] = []

        for idx, row in predictions.iterrows():
            if str(row.get("actual_direction_15m", "")).strip() not in {"", "nan", "NaN"}:
                continue
            has_model_input_ts = str(row.get("model_input_timestamp", "")).strip() not in {"", "nan", "NaN"}
            target_ts = row["timestamp"] if has_model_input_ts else row["timestamp"] + pd.Timedelta(minutes=15)
            if target_ts not in frame.index:
                continue
            start_close = float(row["close"])
            actual_close = float(frame.loc[target_ts, "close"])
            actual_return = float(np.log(actual_close / start_close))
            actual_direction = "UP" if actual_return > 0 else "DOWN"
            prediction_correct = bool(actual_direction == row["direction"])
            evaluated_at = utc_now().isoformat()

            predictions.at[idx, "actual_close_15m"] = actual_close
            predictions.at[idx, "actual_return_15m"] = actual_return
            predictions.at[idx, "actual_direction_15m"] = actual_direction
            predictions.at[idx, "prediction_correct"] = prediction_correct
            predictions.at[idx, "evaluated_at"] = evaluated_at
            updates.append(
                {
                    "prediction_timestamp": row["timestamp"].isoformat(),
                    "model_input_timestamp": row.get("model_input_timestamp", ""),
                    "target_timestamp": target_ts.isoformat(),
                    "predicted_direction": row["direction"],
                    "prob_up": row["prob_up"],
                    "prob_down": row["prob_down"],
                    "start_close": start_close,
                    "actual_close_15m": actual_close,
                    "actual_return_15m": actual_return,
                    "actual_direction_15m": actual_direction,
                    "prediction_correct": prediction_correct,
                    "evaluated_at": evaluated_at,
                }
            )

        if not updates:
            return

        predictions["timestamp"] = predictions["timestamp"].map(lambda value: value.isoformat())
        predictions.to_csv(predictions_path, index=False)
        for update in updates:
            append_csv(self.logs_dir / "actual_outcomes.csv", update, list(update.keys()))
            self.logger.info(
                "Evaluated %s prediction from %s: actual=%s return=%.6f correct=%s",
                update["predicted_direction"],
                update["prediction_timestamp"],
                update["actual_direction_15m"],
                update["actual_return_15m"],
                update["prediction_correct"],
            )

    def step(self) -> dict[str, Any]:
        frame = self.fetch_market_frame()
        self.evaluate_pending_predictions(frame)
        ts, latest_features, model_row = self.build_live_features(frame)
        contract_ts = ts + pd.Timedelta(minutes=15)
        prediction = self.predict(model_row)
        price = float(frame.loc[ts, "close"])
        self.log_cycle(ts, frame, latest_features, prediction)
        return {
            "timestamp": contract_ts.isoformat(),
            "model_input_timestamp": contract_ts.isoformat(),
            "close": price,
            "prediction": prediction,
        }

    def sleep_until_next_boundary(self) -> None:
        delay = float(self.config["loop"]["seconds_after_boundary"])
        now = utc_now()
        minute = (now.minute // 15 + 1) * 15
        boundary = now.replace(second=0, microsecond=0)
        if minute >= 60:
            boundary = boundary.replace(minute=0) + timedelta(hours=1)
        else:
            boundary = boundary.replace(minute=minute)
        wake = boundary + timedelta(seconds=delay)
        sleep_seconds = max(0.0, (wake - now).total_seconds())
        self.logger.info("Waiting %.1fs until %s", sleep_seconds, wake.isoformat())
        time.sleep(sleep_seconds)

    def run(self, once: bool = False) -> None:
        self.logger.info("Starting BTC LightGBM live direction predictor. model=%s once=%s", self.model_name, once)
        while True:
            try:
                result = self.step()
                self.logger.info("Completed cycle: %s", json.dumps(result, default=json_default))
            except Exception:
                self.logger.exception("Paper trading cycle failed")
                if once:
                    raise
            if once:
                return
            self.sleep_until_next_boundary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live BTC paper trading prediction loop.")
    parser.add_argument("--config", default="binary-paper-trading/config.yaml", help="Path to config YAML.")
    parser.add_argument("--once", action="store_true", help="Run one prediction cycle and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trader = LivePaperTrader(ROOT / args.config)
    trader.run(once=args.once)
