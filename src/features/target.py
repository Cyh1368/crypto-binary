from __future__ import annotations

import numpy as np
import pandas as pd


def add_direction_target(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    out = df.copy()
    out["future_return_15m"] = np.log(out["close"].shift(-horizon_bars) / out["close"])
    out["target"] = (out["future_return_15m"] > 0).astype(int)
    return out
