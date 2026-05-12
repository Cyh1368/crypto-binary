from __future__ import annotations

import pandas as pd


def fetch_okx_derivatives(*_: object, **__: object) -> pd.DataFrame:
    raise NotImplementedError("OKX derivatives ingestion must use real OKX API/archive data; no proxy feed is generated.")
