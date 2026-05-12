from __future__ import annotations

import pandas as pd


def fetch_bybit_derivatives(*_: object, **__: object) -> pd.DataFrame:
    raise NotImplementedError("Bybit derivatives ingestion must use real Bybit API/archive data; no proxy feed is generated.")
