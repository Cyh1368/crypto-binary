from __future__ import annotations

import time

import ccxt
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


def fetch_kraken_futures_data(symbol: str = "PI_XBTUSD", limit: int = 50000) -> pd.DataFrame:
    """Fetch real Kraken Futures OHLCV and funding history."""
    exchange = ccxt.krakenfutures()
    ohlcv_rows: list[list[float]] = []
    since = exchange.milliseconds() - (limit * 15 * 60 * 1000)

    LOGGER.info("Fetching %s Kraken Futures 15m OHLCV bars for %s", limit, symbol)
    while len(ohlcv_rows) < limit:
        batch = exchange.fetch_ohlcv(symbol, timeframe="15m", since=since, limit=500)
        if not batch:
            break
        ohlcv_rows.extend(batch)
        since = batch[-1][0] + 1
        time.sleep(0.05)

    if not ohlcv_rows:
        raise RuntimeError(f"No Kraken Futures OHLCV returned for {symbol}")

    df = pd.DataFrame(
        ohlcv_rows[:limit],
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()

    LOGGER.info("Fetching Kraken Futures funding history for %s", symbol)
    funding_rows: list[dict] = []
    since_funding = int(df.index[0].timestamp() * 1000)
    end_ms = int(df.index[-1].timestamp() * 1000)
    while since_funding < end_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=since_funding, limit=1000)
        if not batch:
            break
        funding_rows.extend(batch)
        since_funding = batch[-1]["timestamp"] + 1
        time.sleep(0.05)

    funding = pd.DataFrame(funding_rows)
    if funding.empty:
        df["funding_rate"] = 0.0
        return df

    funding["timestamp"] = pd.to_datetime(funding["timestamp"], unit="ms", utc=True)
    funding = funding.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df["funding_rate"] = funding["fundingRate"].reindex(df.index, method="ffill").fillna(0.0)
    return df
