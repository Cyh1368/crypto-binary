from __future__ import annotations

import pandas as pd


def fetch_coinbase_candles(*_: object, **__: object) -> pd.DataFrame:
    raise NotImplementedError(
        "Coinbase historical candles/order books require authenticated Advanced Trade API access. "
        "Provide exported real data under data/raw; this pipeline will not synthesize Coinbase feeds."
    )
