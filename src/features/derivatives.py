from __future__ import annotations

import numpy as np
import pandas as pd


def add_derivatives_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "funding_rate" not in out:
        out["funding_rate"] = np.nan
    out["funding_change"] = out["funding_rate"].diff()
    out["funding_zscore"] = (
        (out["funding_rate"] - out["funding_rate"].rolling(240).mean())
        / out["funding_rate"].rolling(240).std()
    )
    if "open_interest" in out:
        out["oi_change"] = out["open_interest"].diff()
        out["oi_acceleration"] = out["oi_change"].diff()
    else:
        out["open_interest"] = np.nan
        out["oi_change"] = np.nan
        out["oi_acceleration"] = np.nan
    for col in ["basis", "liquidation_volume", "long_short_ratio"]:
        if col not in out:
            out[col] = np.nan
    out["basis_change"] = out["basis"].diff()
    return out
