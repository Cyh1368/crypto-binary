from __future__ import annotations

import numpy as np
import pandas as pd


def _entropy(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    hist, _ = np.histogram(values, bins=10)
    probs = hist[hist > 0] / hist.sum()
    return float(-(probs * np.log(probs)).sum())


def _hurst(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 20:
        return np.nan
    lags = np.arange(2, min(20, len(values) // 2))
    tau = [np.std(values[lag:] - values[:-lag]) for lag in lags]
    tau = np.array(tau)
    valid = tau > 0
    if valid.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(lags[valid]), np.log(tau[valid]), 1)[0] * 2.0)


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rv = out["log_return"].rolling(60).std()
    out["realized_vol_percentile"] = rv.rolling(240).rank(pct=True)
    out["rolling_entropy"] = out["log_return"].rolling(60).apply(_entropy, raw=False)
    out["hurst_exponent"] = out["close"].rolling(120).apply(_hurst, raw=False)
    out["trend_strength"] = out["close"].pct_change(60) / out["log_return"].rolling(60).std()
    hours = out.index.hour
    out["session_asia"] = ((hours >= 0) & (hours < 8)).astype(int)
    out["session_europe"] = ((hours >= 7) & (hours < 16)).astype(int)
    out["session_us"] = ((hours >= 13) & (hours < 22)).astype(int)
    return out
