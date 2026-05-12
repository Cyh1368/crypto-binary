from __future__ import annotations

import pandas as pd

from src.data.download_binance import build_bids_asks_from_depth


def attach_real_order_books(ohlcv: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    return build_bids_asks_from_depth(ohlcv, depth)
