from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _download_book_depth_day(day: pd.Timestamp, binance_symbol: str, timeout: int = 20) -> pd.DataFrame | None:
    day_str = day.strftime("%Y-%m-%d")
    url = (
        "https://data.binance.vision/data/futures/um/daily/bookDepth/"
        f"{binance_symbol}/{binance_symbol}-bookDepth-{day_str}.zip"
    )
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_name = zf.namelist()[0]
            df = pd.read_csv(zf.open(csv_name))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["source"] = "binance_vision_bookDepth"
        return df
    except Exception as exc:  # public archive has gaps; continue and report aggregate coverage
        LOGGER.warning("Failed Binance Vision bookDepth download for %s %s: %s", binance_symbol, day_str, exc)
        return None


def download_binance_vision_depth(
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    binance_symbol: str = "BTCUSDT",
    max_workers: int = 12,
) -> pd.DataFrame:
    """Download real Binance Vision bookDepth summaries for inclusive date range."""
    dates = pd.date_range(pd.Timestamp(start).date(), pd.Timestamp(end).date(), freq="D")
    LOGGER.info("Downloading Binance Vision bookDepth for %s over %s days", binance_symbol, len(dates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        rows = list(executor.map(lambda day: _download_book_depth_day(day, binance_symbol), dates))
    frames = [frame for frame in rows if frame is not None and not frame.empty]
    time.sleep(0.1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def build_bids_asks_from_depth(ohlcv: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Align real Binance depth summaries to bars and store raw bid/ask arrays."""
    if depth.empty:
        raise ValueError("depth is empty; refusing to create synthetic order books")

    bars = ohlcv.copy().sort_index()
    depth = depth.copy().sort_values("timestamp")
    depth["timestamp"] = pd.to_datetime(depth["timestamp"], utc=True).dt.as_unit("ns")
    bar_times = pd.DatetimeIndex(bars.index).as_unit("ns")
    unique_depth_ts = pd.DatetimeIndex(depth["timestamp"].drop_duplicates()).sort_values()

    matches = pd.merge_asof(
        pd.DataFrame(index=bar_times),
        pd.DataFrame({"match_ts": unique_depth_ts}, index=unique_depth_ts),
        left_index=True,
        right_index=True,
        direction="backward",
    )
    grouped = depth.groupby("timestamp", sort=True)

    bids: list[list[list[float]]] = []
    asks: list[list[list[float]]] = []
    for ts, match_ts in zip(bar_times, matches["match_ts"], strict=False):
        if pd.isna(match_ts):
            bids.append([])
            asks.append([])
            continue
        snapshot = grouped.get_group(match_ts)
        close = float(bars.loc[ts, "close"])
        bid_levels: list[list[float]] = []
        ask_levels: list[list[float]] = []
        for row in snapshot.itertuples(index=False):
            pct = float(getattr(row, "percentage"))
            qty = float(getattr(row, "depth"))
            price = close * (1.0 + pct / 100.0)
            if pct < 0:
                bid_levels.append([price, qty])
            elif pct > 0:
                ask_levels.append([price, qty])
        bids.append(sorted(bid_levels, key=lambda level: level[0], reverse=True))
        asks.append(sorted(ask_levels, key=lambda level: level[0]))

    bars["bids"] = bids
    bars["asks"] = asks
    return bars[bars["bids"].map(len).gt(0) & bars["asks"].map(len).gt(0)].copy()
