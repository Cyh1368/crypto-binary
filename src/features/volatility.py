from __future__ import annotations

import pandas as pd


def add_volatility_regimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    vol = out["log_return"].rolling(60).std()
    ranked = vol.rank(method="first").dropna()
    out["volatility_regime"] = "unknown"
    if len(ranked) >= 3:
        out.loc[ranked.index, "volatility_regime"] = pd.qcut(ranked, q=3, labels=["low", "medium", "high"])
    return out
