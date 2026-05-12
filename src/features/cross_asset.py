from __future__ import annotations

import numpy as np
import pandas as pd


def add_cross_asset_features(df: pd.DataFrame, assets: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    out = df.copy()
    assets = assets or {}
    for name in ["eth", "sol", "spy", "qqq", "vix", "dxy", "gold"]:
        frame = assets.get(name)
        col = f"{name}_return"
        if frame is not None and not frame.empty:
            close = frame["close"].reindex(out.index, method="ffill")
            out[col] = np.log(close / close.shift(1))
        else:
            out[col] = np.nan
    out["btc_eth_relative_strength"] = out["log_return"] - out["eth_return"]
    return out
