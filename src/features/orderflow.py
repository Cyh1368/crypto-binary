from __future__ import annotations

import numpy as np
import pandas as pd


def _levels(levels: object, n: int) -> list[list[float]]:
    if not isinstance(levels, list):
        return []
    return [[float(price), float(size)] for price, size in levels[:n] if float(size) >= 0]


def _levels_from_snapshot(levels: object, n: int) -> list[list[float]]:
    if not isinstance(levels, (list, tuple, np.ndarray)):
        return []
    parsed = []
    for level in list(levels)[:n]:
        if not isinstance(level, (list, tuple, np.ndarray)) or len(level) < 2:
            continue
        price, size = level[0], level[1]
        if pd.isna(price) or pd.isna(size) or float(size) < 0:
            continue
        parsed.append([float(price), float(size)])
    return parsed


def _sum_size(levels: list[list[float]]) -> float:
    return float(sum(size for _, size in levels))


def _weighted_price(levels: list[list[float]]) -> float:
    total = _sum_size(levels)
    if total <= 0:
        return np.nan
    return float(sum(price * size for price, size in levels) / total)


def _slope(levels: list[list[float]]) -> float:
    if len(levels) < 2:
        return np.nan
    prices = np.array([p for p, _ in levels], dtype=float)
    sizes = np.array([s for _, s in levels], dtype=float)
    if sizes.sum() <= 0:
        return np.nan
    x = np.arange(len(levels), dtype=float)
    return float(np.polyfit(x, sizes / sizes.sum(), 1)[0])


def add_orderflow_features(df: pd.DataFrame, levels: list[int]) -> pd.DataFrame:
    out = df.copy()
    best_bid = out["bids"].map(lambda bids: _levels(bids, 1)[0][0] if _levels(bids, 1) else np.nan)
    best_ask = out["asks"].map(lambda asks: _levels(asks, 1)[0][0] if _levels(asks, 1) else np.nan)
    bid_qty = out["bids"].map(lambda bids: _levels(bids, 1)[0][1] if _levels(bids, 1) else np.nan)
    ask_qty = out["asks"].map(lambda asks: _levels(asks, 1)[0][1] if _levels(asks, 1) else np.nan)

    mid = (best_bid + best_ask) / 2.0
    out["spread_bps"] = (best_ask - best_bid) / mid * 10000.0
    out["microprice"] = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
    out["queue_imbalance"] = (bid_qty - ask_qty) / (bid_qty + ask_qty)

    for n in levels:
        bid_levels = out["bids"].map(lambda bids, n=n: _levels(bids, n))
        ask_levels = out["asks"].map(lambda asks, n=n: _levels(asks, n))
        bid_volume = bid_levels.map(_sum_size)
        ask_volume = ask_levels.map(_sum_size)
        denom = bid_volume + ask_volume
        out[f"obi_{n}"] = (bid_volume - ask_volume) / denom
        out[f"depth_ratio_{n}"] = bid_volume / ask_volume.replace(0.0, np.nan)
        out[f"weighted_bid_{n}"] = bid_levels.map(_weighted_price)
        out[f"weighted_ask_{n}"] = ask_levels.map(_weighted_price)
        out[f"weighted_midprice_{n}"] = (out[f"weighted_bid_{n}"] + out[f"weighted_ask_{n}"]) / 2.0
        out[f"book_pressure_{n}"] = out[f"weighted_midprice_{n}"] - mid
        out[f"bid_depth_slope_{n}"] = bid_levels.map(_slope)
        out[f"ask_depth_slope_{n}"] = ask_levels.map(_slope)
        out[f"depth_slope_{n}"] = out[f"bid_depth_slope_{n}"] - out[f"ask_depth_slope_{n}"]

    if "taker_buy_volume" in out and "volume" in out:
        buy = out["taker_buy_volume"]
        sell = out["volume"] - buy
        out["trade_imbalance"] = (buy - sell) / (buy + sell)
        out["volume_imbalance"] = out["trade_imbalance"]
    else:
        out["trade_imbalance"] = np.nan
        out["volume_imbalance"] = np.nan
    out["trade_arrival_rate"] = out.get("trade_count", pd.Series(index=out.index, dtype=float))
    out["cancel_rate"] = np.nan

    obi_levels = [5, 10, 20]
    bid_volumes: dict[int, pd.Series] = {}
    ask_volumes: dict[int, pd.Series] = {}
    for n in obi_levels:
        bid_levels = out["bids"].map(lambda bids, n=n: _levels_from_snapshot(bids, n))
        ask_levels = out["asks"].map(lambda asks, n=n: _levels_from_snapshot(asks, n))
        bid_volume = bid_levels.map(_sum_size)
        ask_volume = ask_levels.map(_sum_size)
        bid_volumes[n] = bid_volume
        ask_volumes[n] = ask_volume
        denom = bid_volume + ask_volume
        out[f"obi_raw_{n}"] = (bid_volume - ask_volume) / denom.replace(0.0, np.nan)
        out[f"bid_ask_depth_ratio_{n}"] = bid_volume / (ask_volume + 1e-8)

    out["obi_delta_5"] = out["obi_raw_5"].diff()
    out["obi_delta_10"] = out["obi_raw_10"].diff()
    out["obi_pressure_ratio"] = out["obi_raw_5"] / (out["obi_raw_20"] + 1e-8)

    for window in [3, 5, 15]:
        out[f"obi_rolling_mean_{window}"] = out["obi_raw_5"].rolling(window).mean()
        out[f"obi_rolling_std_{window}"] = out["obi_raw_5"].rolling(window).std()

    rolling_mean = out["obi_raw_5"].rolling(60).mean()
    rolling_std = out["obi_raw_5"].rolling(60).std()
    out["obi_zscore_5"] = (out["obi_raw_5"] - rolling_mean) / (rolling_std + 1e-8)
    return out
