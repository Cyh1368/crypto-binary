from __future__ import annotations

import argparse
import csv
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import yaml
except ImportError:  # pragma: no cover - the project venv already includes PyYAML.
    yaml = None

try:
    import ccxt
except ImportError:  # pragma: no cover - the project venv already includes ccxt.
    ccxt = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "binary-paper-trading" / "config.yaml"
DEFAULT_LOGS_DIR = ROOT / "binary-paper-trading" / "logs"


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_candidate = (ROOT / candidate).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return cwd_candidate


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_timestamp(value: str) -> datetime | None:
    value = str(value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def truthy(value: Any) -> bool | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def load_config(config_path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is not installed; cannot load dashboard config")
    if not config_path.exists():
        raise FileNotFoundError(f"Dashboard config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def infer_model_name(config: dict[str, Any]) -> str:
    configured = str(config.get("model_name") or "").strip()
    if configured:
        return configured
    model_path = Path(str(config.get("model_path") or ""))
    if len(model_path.parents) > 1:
        return model_path.parents[1].name
    return model_path.stem or "unknown"


@dataclass(frozen=True)
class DashboardConfig:
    logs_dir: Path
    symbol: str
    model_name: str
    model_path: str
    timeframe: str
    seconds_after_boundary: int
    uses_obi_backfill: bool
    obi_required_snapshots: int
    poll_seconds: int
    live_price_seconds: int


def build_config(config_path: Path, logs_dir: Path | None, poll_seconds: int, live_price_seconds: int) -> DashboardConfig:
    config = load_config(config_path)
    configured_logs = config.get("logs_dir")
    resolved_logs_dir = logs_dir
    if resolved_logs_dir is None and configured_logs:
        resolved_logs_dir = ROOT / str(configured_logs)
    if resolved_logs_dir is None:
        resolved_logs_dir = DEFAULT_LOGS_DIR

    loop_config = config.get("loop") or {}
    obi_config = config.get("obi_backfill") or {}
    seconds_after_boundary = int(loop_config.get("seconds_after_boundary", 0))
    return DashboardConfig(
        logs_dir=resolved_logs_dir,
        symbol=str(config.get("symbol", "BTC/USD:USD")),
        model_name=infer_model_name(config),
        model_path=str(config.get("model_path", "")),
        timeframe=str(config.get("timeframe", "15m")),
        seconds_after_boundary=seconds_after_boundary,
        uses_obi_backfill="obi_backfill" in config,
        obi_required_snapshots=int(obi_config.get("min_history_snapshots", 60)),
        poll_seconds=poll_seconds,
        live_price_seconds=live_price_seconds,
    )


def prediction_probability(row: dict[str, Any]) -> float | None:
    direction = str(row.get("direction") or "").upper()
    if direction == "UP":
        return parse_float(row.get("prob_up"))
    if direction == "DOWN":
        return parse_float(row.get("prob_down"))
    return parse_float(row.get("confidence_percent"))


def format_probability(value: float | None) -> float | None:
    if value is None:
        return None
    if value > 1:
        return value
    return value * 100


def candle_close_time(open_time: datetime) -> datetime:
    return open_time + timedelta(minutes=15)


def read_predictions(logs_dir: Path) -> list[dict[str, Any]]:
    path = logs_dir / "predictions.csv"
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    deduped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for sequence, raw in enumerate(rows):
        ts = parse_timestamp(raw.get("timestamp", ""))
        if ts is None:
            continue
        model_input_ts = parse_timestamp(raw.get("model_input_timestamp", ""))
        contract_ts = ts if model_input_ts else ts + timedelta(minutes=15)
        key = contract_ts.isoformat()
        row = dict(raw)
        row["_sequence"] = sequence
        row["_parsed_ts"] = contract_ts
        row["_raw_timestamp"] = ts
        if key not in deduped:
            order.append(key)
            deduped[key] = row
            continue

        previous = deduped[key]
        if not row.get("actual_direction_15m") and previous.get("actual_direction_15m"):
            for col in ("actual_close_15m", "actual_return_15m", "actual_direction_15m", "prediction_correct", "evaluated_at"):
                row[col] = previous.get(col, "")
        deduped[key] = row

    unique_rows = [deduped[key] for key in order]
    unique_rows.sort(key=lambda row: (row["_parsed_ts"], row["_sequence"]))
    return unique_rows


def serialize_prediction(row: dict[str, Any]) -> dict[str, Any]:
    predicted = str(row.get("direction") or "").upper()
    actual = str(row.get("actual_direction_15m") or "").upper()
    probability = format_probability(prediction_probability(row))
    correct = truthy(row.get("prediction_correct"))
    stored_model_input_ts = parse_timestamp(row.get("model_input_timestamp", ""))
    model_input_close_ts = stored_model_input_ts
    if stored_model_input_ts and stored_model_input_ts < row["_parsed_ts"]:
        model_input_close_ts = candle_close_time(stored_model_input_ts)
    if model_input_close_ts is None and row.get("_raw_timestamp"):
        model_input_close_ts = candle_close_time(row["_raw_timestamp"])

    return {
        "timestamp": row["_parsed_ts"].isoformat(),
        "model_input_timestamp": model_input_close_ts.isoformat() if model_input_close_ts else None,
        "model_input_price": parse_float(row.get("close")),
        "predicted_direction": predicted or None,
        "probability": probability,
        "prob_up": format_probability(parse_float(row.get("prob_up"))),
        "prob_down": format_probability(parse_float(row.get("prob_down"))),
        "actual_direction": actual or None,
        "prediction_correct": correct,
        "evaluated_at": row.get("evaluated_at") or None,
    }


def confusion_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {
        "UP": {"UP": 0, "DOWN": 0},
        "DOWN": {"UP": 0, "DOWN": 0},
    }
    for row in rows:
        predicted = str(row.get("direction") or "").upper()
        actual = str(row.get("actual_direction_15m") or "").upper()
        if predicted in matrix and actual in matrix[predicted]:
            matrix[predicted][actual] += 1
    return matrix


def read_latest_obi_warning(logs_dir: Path) -> str | None:
    path = logs_dir / "paper_trading.log"
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-300:]):
        if "[WARNING]" not in line:
            continue
        if "OBI" in line or "bookDepth" in line or "Binance Vision" in line:
            return line
    return None


def read_obi_status(logs_dir: Path, required_snapshots: int, uses_obi_backfill: bool) -> dict[str, Any]:
    if not uses_obi_backfill:
        return {
            "state": "not_required",
            "required_snapshots": 0,
            "snapshot_count": 0,
            "continuous_snapshots": 0,
            "remaining_ticks": 0,
            "last_obi_timestamp": None,
            "last_warning": None,
        }

    path = logs_dir / "live_obi_history.csv"
    timestamps: list[datetime] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ts = parse_timestamp(row.get("timestamp", ""))
                if ts is not None:
                    timestamps.append(ts)

    unique = sorted(set(timestamps))
    continuous_tail = 0
    if unique:
        expected = unique[-1]
        available = set(unique)
        while expected in available:
            continuous_tail += 1
            expected -= timedelta(minutes=15)

    ready = continuous_tail >= required_snapshots
    remaining = max(0, required_snapshots - continuous_tail)
    return {
        "state": "ready" if ready else "warming_up",
        "required_snapshots": required_snapshots,
        "snapshot_count": len(unique),
        "continuous_snapshots": continuous_tail,
        "remaining_ticks": remaining,
        "last_obi_timestamp": unique[-1].isoformat() if unique else None,
        "last_warning": read_latest_obi_warning(logs_dir),
    }


def next_prediction_time(now: datetime, seconds_after_boundary: int) -> datetime:
    boundary = now.replace(second=0, microsecond=0)
    next_minute = (now.minute // 15 + 1) * 15
    if next_minute >= 60:
        boundary = boundary.replace(minute=0) + timedelta(hours=1)
    else:
        boundary = boundary.replace(minute=next_minute)
    return boundary + timedelta(seconds=seconds_after_boundary)


def dashboard_payload(config: DashboardConfig) -> dict[str, Any]:
    rows = read_predictions(config.logs_dir)
    serialized = [serialize_prediction(row) for row in rows]
    evaluated = [
        row
        for row in rows
        if str(row.get("direction") or "").upper() in {"UP", "DOWN"}
        and str(row.get("actual_direction_15m") or "").upper() in {"UP", "DOWN"}
    ]
    current = serialized[-1] if serialized else None
    now = utc_now()
    next_prediction = next_prediction_time(now, config.seconds_after_boundary)
    prediction_status = read_obi_status(config.logs_dir, config.obi_required_snapshots, config.uses_obi_backfill)
    return {
        "generated_at": now.isoformat(),
        "logs_dir": str(config.logs_dir),
        "model_name": config.model_name,
        "model_path": config.model_path,
        "current": current,
        "history": list(reversed(serialized)),
        "confusion_matrix": confusion_matrix(evaluated),
        "prediction_status": prediction_status,
        "counts": {
            "contracts": len(rows),
            "evaluated": len(evaluated),
            "correct": sum(1 for row in evaluated if truthy(row.get("prediction_correct")) is True),
            "pending": len(rows) - len(evaluated),
        },
        "next_prediction_at": next_prediction.isoformat(),
        "seconds_after_boundary": config.seconds_after_boundary,
        "poll_seconds": config.poll_seconds,
        "live_price_seconds": config.live_price_seconds,
    }


class LivePriceSource:
    def __init__(self, config: DashboardConfig):
        self.config = config
        self._exchange: Any = None
        self._lock = threading.Lock()

    def _get_exchange(self) -> Any:
        if ccxt is None:
            raise RuntimeError("ccxt is not installed")
        if self._exchange is None:
            self._exchange = ccxt.krakenfutures({"enableRateLimit": True})
        return self._exchange

    def get(self) -> dict[str, Any]:
        with self._lock:
            now = utc_now()
            try:
                exchange = self._get_exchange()
                ohlcv = exchange.fetch_ohlcv(self.config.symbol, timeframe=self.config.timeframe, limit=2)
                orderbook = exchange.fetch_order_book(self.config.symbol, limit=1)
                if not ohlcv:
                    raise RuntimeError("No OHLCV rows returned")
                latest = ohlcv[-1]
                open_time = datetime.fromtimestamp(float(latest[0]) / 1000.0, tz=timezone.utc)
                best_bid = float(orderbook["bids"][0][0]) if orderbook.get("bids") else None
                best_ask = float(orderbook["asks"][0][0]) if orderbook.get("asks") else None
                orderbook_mid = None
                if best_bid is not None and best_ask is not None:
                    orderbook_mid = (best_bid + best_ask) / 2.0
                payload = {
                    "generated_at": now.isoformat(),
                    "symbol": self.config.symbol,
                    "timeframe": self.config.timeframe,
                    "price": orderbook_mid if orderbook_mid is not None else float(latest[4]),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "ohlcv_latest_close": float(latest[4]),
                    "candle_open_timestamp": open_time.isoformat(),
                    "candle_close_timestamp": candle_close_time(open_time).isoformat(),
                    "source": "Kraken Futures fetch_order_book midpoint",
                    "refresh_seconds": self.config.live_price_seconds,
                }
            except Exception as exc:
                payload = {
                    "generated_at": now.isoformat(),
                    "symbol": self.config.symbol,
                    "timeframe": self.config.timeframe,
                    "price": None,
                    "best_bid": None,
                    "best_ask": None,
                    "ohlcv_latest_close": None,
                    "candle_open_timestamp": None,
                    "candle_close_timestamp": None,
                    "source": "Kraken Futures fetch_order_book midpoint",
                    "refresh_seconds": self.config.live_price_seconds,
                }
                payload["error"] = str(exc)
            return payload


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Trading Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --text: #1f2937;
      --muted: #64748b;
      --line: #d8dee8;
      --accent: #1d4ed8;
      --green-bg: #dcfce7;
      --green-text: #166534;
      --red-bg: #fee2e2;
      --red-text: #991b1b;
      --pending-bg: #eef2f7;
      --warn-bg: #fff7ed;
      --warn-text: #9a3412;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      width: min(1400px, calc(100% - 32px));
      margin: 0 auto;
      padding: 24px 0;
    }
    .top {
      display: grid;
      grid-template-columns: 1.5fr repeat(5, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .model-banner {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      margin-bottom: 12px;
    }
    .model-banner strong {
      font-size: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
    }
    .metric {
      padding: 16px;
      min-height: 92px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
    }
    .value {
      font-size: 24px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .subtle {
      color: var(--muted);
      font-size: 13px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 16px;
      align-items: start;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    h1, h2 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
    }
    .table-wrap {
      max-height: calc(100vh - 190px);
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #475569;
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    tr.correct td { background: var(--green-bg); color: var(--green-text); }
    tr.wrong td { background: var(--red-bg); color: var(--red-text); }
    tr.pending td { background: var(--pending-bg); }
    .metric.warning {
      background: var(--warn-bg);
      color: var(--warn-text);
      border-color: #fed7aa;
    }
    .matrix {
      padding: 16px;
    }
    .matrix table th, .matrix table td {
      text-align: center;
      font-size: 15px;
    }
    .matrix table th:first-child {
      text-align: left;
    }
    .matrix .axis-cell {
      position: relative;
      height: 58px;
      min-width: 104px;
      padding: 0;
      background:
        linear-gradient(to top right, transparent calc(50% - 1px), var(--line) 50%, transparent calc(50% + 1px)),
        #f8fafc;
      text-align: initial;
    }
    .matrix .axis-cell span {
      position: absolute;
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.06em;
      color: #475569;
      line-height: 1;
      white-space: nowrap;
    }
    .matrix .axis-cell .axis-actual {
      top: 10px;
      right: 10px;
    }
    .matrix .axis-cell .axis-pred {
      left: 10px;
      bottom: 10px;
    }
    .count-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      padding: 0 16px 16px;
    }
    .count {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fbfdff;
    }
    .count strong {
      display: block;
      font-size: 20px;
      margin-top: 3px;
    }
    .empty {
      padding: 32px 16px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 900px) {
      main { width: min(100% - 20px, 1400px); padding: 10px 0; }
      .top, .layout { grid-template-columns: 1fr; }
      .table-wrap { max-height: none; }
    }
  </style>
</head>
<body>
  <main>
    <section class="model-banner panel">
      <div>
        <div class="label">Live Trading Model</div>
        <strong id="model-name">-</strong>
      </div>
      <div class="subtle" id="model-path">-</div>
    </section>
    <section class="top">
      <div class="metric panel">
        <div class="label">Current Contract Timestamp</div>
        <div class="value" id="current-ts">-</div>
        <div class="subtle" id="last-updated">Waiting for data</div>
      </div>
      <div class="metric panel">
        <div class="label">Prediction</div>
        <div class="value" id="current-direction">-</div>
        <div class="subtle" id="prediction-status">Predicted direction</div>
      </div>
      <div class="metric panel">
        <div class="label">Probability</div>
        <div class="value" id="current-probability">-</div>
        <div class="subtle">Probability for predicted direction</div>
      </div>
      <div class="metric panel">
        <div class="label">Live BTC Price</div>
        <div class="value" id="live-price">-</div>
        <div class="subtle" id="live-price-source">Kraken Futures</div>
      </div>
      <div class="metric panel">
        <div class="label">Prediction Input Close</div>
        <div class="value" id="current-price">-</div>
        <div class="subtle" id="price-source">Model input candle close</div>
      </div>
      <div class="metric panel">
        <div class="label">Next Prediction</div>
        <div class="value" id="countdown">--:--</div>
        <div class="subtle" id="next-at">-</div>
      </div>
    </section>

    <section class="layout">
      <div class="panel">
        <div class="section-head">
          <h1>Historical Predictions <a href="https://polymarket.com/crypto/15M">Polymarket</a> <a href="https://kalshi.com/markets/kxbtc15m/bitcoin-price-up-down">Kalshi</a></h1>
          <div class="subtle" id="contract-count">0 contracts</div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Predicted</th>
                <th>Probability</th>
                <th>Actual</th>
              </tr>
            </thead>
            <tbody id="history-body"></tbody>
          </table>
          <div class="empty" id="empty-state">No predictions found.</div>
        </div>
      </div>

      <aside class="panel">
        <div class="section-head">
          <h2>Confusion Matrix</h2>
        </div>
        <div class="matrix">
          <table>
            <thead>
              <tr>
                <th class="axis-cell"><span class="axis-actual">ACTUAL</span><span class="axis-pred">PRED</span></th>
                <th>UP</th>
                <th>DOWN</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <th>UP</th>
                <td id="m-up-up" style="background-color: #b9fab9;">0</td>
                <td id="m-up-down">0</td>
              </tr>
              <tr>
                <th>DOWN</th>
                <td id="m-down-up">0</td>
                <td id="m-down-down" style="background-color: #b9fab9;">0</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="count-grid">
          <div class="count"><span class="subtle">Evaluated</span><strong id="evaluated-count">0</strong></div>
          <div class="count"><span class="subtle">Correct</span><strong id="correct-count">0</strong></div>
          <div class="count"><span class="subtle">Pending</span><strong id="pending-count">0</strong></div>
          <div class="count"><span class="subtle">Accuracy</span><strong id="accuracy">-</strong></div>
        </div>
      </aside>
    </section>
  </main>

  <script>
    let nextPredictionAt = null;
    let secondsAfterBoundary = 0;
    let refreshTimer = null;
    let livePriceTimer = null;

    const fmt = new Intl.DateTimeFormat(undefined, {
      year: "numeric", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit", second: "2-digit",
      timeZoneName: "short"
    });

    function text(id, value) {
      document.getElementById(id).textContent = value;
    }

    function pct(value) {
      return Number.isFinite(value) ? `${value.toFixed(2)}%` : "-";
    }

    function usd(value) {
      return Number.isFinite(value) ? `$${value.toLocaleString(undefined, {maximumFractionDigits: 2})}` : "-";
    }

    function formatTime(value) {
      if (!value) return "-";
      const date = new Date(value);
      return Number.isNaN(date.getTime()) ? value : fmt.format(date);
    }

    function rowClass(row) {
      if (row.prediction_correct === true) return "correct";
      if (row.prediction_correct === false) return "wrong";
      return "pending";
    }

    function render(data) {
      secondsAfterBoundary = data.seconds_after_boundary || 0;
      nextPredictionAt = data.next_prediction_at ? new Date(data.next_prediction_at) : null;
      scheduleRefresh(data.poll_seconds || 5);

      const current = data.current;
      const status = data.prediction_status || {};
      const warmingUp = status.state === "warming_up";
      text("model-name", data.model_name || "-");
      text("model-path", data.model_path || "-");
      document.getElementById("current-direction").closest(".metric").classList.toggle("warning", warmingUp);
      if (warmingUp) {
        text("current-ts", "No active prediction");
        text("current-direction", "WARMUP");
        text("prediction-status", `${status.remaining_ticks || 0} ticks remaining`);
        text("current-probability", "-");
        text("current-price", current ? usd(current.model_input_price) : "-");
        text(
          "price-source",
          `OBI window ${status.continuous_snapshots || 0}/${status.required_snapshots || 60}; ${status.remaining_ticks || 0} ticks remaining; no CSV prediction row is written`
        );
        text("last-updated", status.last_warning || `Dashboard refreshed ${formatTime(data.generated_at)}`);
      } else {
        text("current-ts", current ? formatTime(current.timestamp) : "-");
        text("current-direction", current?.predicted_direction || "-");
        text("prediction-status", "Predicted direction");
        text("current-probability", current ? pct(current.probability) : "-");
        text("current-price", current ? usd(current.model_input_price) : "-");
        text("price-source", current?.model_input_timestamp ? `Candle close ${formatTime(current.model_input_timestamp)}` : "Model input candle close");
        text("last-updated", `Dashboard refreshed ${formatTime(data.generated_at)}`);
      }
      text("next-at", nextPredictionAt ? formatTime(nextPredictionAt.toISOString()) : "-");

      const body = document.getElementById("history-body");
      body.innerHTML = "";
      for (const row of data.history) {
        const tr = document.createElement("tr");
        tr.className = rowClass(row);
        const cells = [
          formatTime(row.timestamp),
          row.predicted_direction || "-",
          pct(row.probability),
          row.actual_direction || "Pending"
        ];
        for (const value of cells) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.appendChild(td);
        }
        body.appendChild(tr);
      }
      document.getElementById("empty-state").style.display = data.history.length ? "none" : "block";

      const matrix = data.confusion_matrix || {UP: {UP: 0, DOWN: 0}, DOWN: {UP: 0, DOWN: 0}};
      text("m-up-up", matrix.UP.UP);
      text("m-up-down", matrix.UP.DOWN);
      text("m-down-up", matrix.DOWN.UP);
      text("m-down-down", matrix.DOWN.DOWN);

      const counts = data.counts || {};
      const evaluated = counts.evaluated || 0;
      const correct = counts.correct || 0;
      text("contract-count", `${counts.contracts || 0} unique contracts`);
      text("evaluated-count", evaluated);
      text("correct-count", correct);
      text("pending-count", counts.pending || 0);
      text("accuracy", evaluated ? `${((correct / evaluated) * 100).toFixed(1)}%` : "-");
    }

    function computeNextPrediction(now) {
      const next = new Date(now);
      next.setMilliseconds(0);
      next.setSeconds(0);
      const minutes = next.getMinutes();
      const nextMinutes = (Math.floor(minutes / 15) + 1) * 15;
      if (nextMinutes >= 60) {
        next.setHours(next.getHours() + 1, 0, secondsAfterBoundary, 0);
      } else {
        next.setMinutes(nextMinutes, secondsAfterBoundary, 0);
      }
      return next;
    }

    function updateCountdown() {
      const now = new Date();
      if (!nextPredictionAt || nextPredictionAt <= now) {
        nextPredictionAt = computeNextPrediction(now);
        text("next-at", formatTime(nextPredictionAt.toISOString()));
      }
      const ms = Math.max(0, nextPredictionAt.getTime() - now.getTime());
      const totalSeconds = Math.ceil(ms / 1000);
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      text("countdown", `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`);
    }

    async function loadData() {
      try {
        const response = await fetch("/api/data", {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        text("last-updated", `Dashboard refresh failed: ${error.message}`);
      }
    }

    function renderLivePrice(data) {
      text("live-price", usd(data.price));
      const sourceBits = [];
      if (data.best_bid && data.best_ask) sourceBits.push(`Bid ${usd(data.best_bid)} / Ask ${usd(data.best_ask)}`);
      if (data.candle_close_timestamp) sourceBits.push(`Current candle ends ${formatTime(data.candle_close_timestamp)}`);
      if (data.generated_at) sourceBits.push(`Updated ${formatTime(data.generated_at)}`);
      if (data.error) sourceBits.push(`Refresh failed: ${data.error}`);
      text("live-price-source", sourceBits.join(" | ") || data.source || "Kraken Futures");
    }

    async function loadLivePrice() {
      try {
        const response = await fetch("/api/live-price", {cache: "no-store"});
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        renderLivePrice(await response.json());
      } catch (error) {
        text("live-price-source", `Live price refresh failed: ${error.message}`);
      }
    }

    function scheduleRefresh(seconds) {
      const intervalMs = Math.max(1, seconds) * 1000;
      if (refreshTimer?.intervalMs === intervalMs) return;
      if (refreshTimer) clearInterval(refreshTimer.id);
      refreshTimer = {
        id: setInterval(loadData, intervalMs),
        intervalMs
      };
    }

    loadData();
    loadLivePrice();
    scheduleRefresh(5);
    livePriceTimer = setInterval(loadLivePrice, 10000);
    setInterval(updateCountdown, 250);
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig
    live_price_source: LivePriceSource

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self.send_html(INDEX_HTML)
            return
        if parsed.path == "/api/data":
            params = parse_qs(parsed.query)
            limit = int(params.get("limit", ["0"])[0] or 0)
            payload = dashboard_payload(self.config)
            if limit > 0:
                payload["history"] = payload["history"][:limit]
            self.send_json(payload)
            return
        if parsed.path == "/api/live-price":
            self.send_json(self.live_price_source.get())
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a live dashboard for binary paper trading logs.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to paper trading config YAML.")
    parser.add_argument("--logs-dir", default=None, help="Override the logs directory from the config.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8097, help="Port to bind.")
    parser.add_argument("--poll-seconds", type=int, default=5, help="Browser data refresh interval.")
    parser.add_argument("--live-price-seconds", type=int, default=10, help="Live BTC price refresh interval.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = resolve_repo_path(args.logs_dir) if args.logs_dir else None
    config = build_config(resolve_repo_path(args.config), logs_dir, args.poll_seconds, args.live_price_seconds)

    class ConfiguredDashboardHandler(DashboardHandler):
        pass

    ConfiguredDashboardHandler.config = config
    ConfiguredDashboardHandler.live_price_source = LivePriceSource(config)
    server = ThreadingHTTPServer((args.host, args.port), ConfiguredDashboardHandler)
    print(f"Serving paper trading dashboard at http://{args.host}:{args.port}")
    print(f"Reading predictions from {config.logs_dir / 'predictions.csv'}")
    print(f"Fetching live BTC price from Kraken Futures {config.symbol} {config.timeframe} every {config.live_price_seconds}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
