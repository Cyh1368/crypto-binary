from __future__ import annotations

import pandas as pd


def utc_index(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], utc=True)
    return out.set_index(column).sort_index()


def floor_to_15m(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index).floor("15min")
