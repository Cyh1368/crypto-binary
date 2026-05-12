from __future__ import annotations

import pandas as pd

from src.features.cross_asset import add_cross_asset_features
from src.features.derivatives import add_derivatives_features
from src.features.orderflow import add_orderflow_features
from src.features.regime import add_regime_features
from src.features.target import add_direction_target
from src.features.technical import add_price_features
from src.features.volatility import add_volatility_regimes


NON_FEATURE_COLUMNS = {"target", "future_return_15m", "bids", "asks", "volatility_regime"}


def build_features(
    raw: pd.DataFrame,
    price_windows: list[int],
    orderbook_levels: list[int],
    target_horizon_bars: int = 1,
    cross_assets: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = raw.sort_index().copy()
    df = add_price_features(df, price_windows)
    df = add_orderflow_features(df, orderbook_levels)
    df = add_derivatives_features(df)
    df = add_regime_features(df)
    df = add_cross_asset_features(df, cross_assets)
    df = add_volatility_regimes(df)
    df = add_direction_target(df, target_horizon_bars)
    df = df.dropna(subset=["target", "future_return_15m"])
    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    return df, feature_cols
