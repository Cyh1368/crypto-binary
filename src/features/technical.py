from __future__ import annotations

import numpy as np
import pandas as pd


def add_price_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    out["log_return"] = np.log(out["close"] / out["close"].shift(1))
    out["close_open_range"] = (out["close"] - out["open"]) / out["open"]
    out["high_low_range"] = (out["high"] - out["low"]) / out["close"]
    out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3.0
    out["vwap_distance"] = (out["close"] - out["vwap"]) / out["vwap"]

    for window in windows:
        ret = np.log(out["close"] / out["close"].shift(window))
        out[f"rolling_return_{window}"] = ret
        out[f"rolling_volatility_{window}"] = out["log_return"].rolling(window).std()
        out[f"realized_volatility_{window}"] = np.sqrt((out["log_return"] ** 2).rolling(window).sum())
        hl = np.log(out["high"] / out["low"])
        out[f"parkinson_volatility_{window}"] = np.sqrt((hl**2).rolling(window).sum() / (4.0 * window * np.log(2.0)))
    return out
