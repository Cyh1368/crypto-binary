from __future__ import annotations

import pandas as pd


def build_liquidity_weighted_reference_price(frames: dict[str, pd.DataFrame], weights: dict[str, float]) -> pd.Series:
    """Build P_t from real venue close prices using configured liquidity weights."""
    aligned: list[pd.Series] = []
    used_weights: list[float] = []
    for venue, frame in frames.items():
        if frame.empty or venue not in weights:
            continue
        aligned.append(frame["close"].rename(venue))
        used_weights.append(float(weights[venue]))
    if not aligned:
        raise ValueError("No real venue price frames supplied for reference price")
    prices = pd.concat(aligned, axis=1).sort_index().ffill()
    w = pd.Series(used_weights, index=prices.columns, dtype=float)
    w = w / w.sum()
    return prices.mul(w, axis=1).sum(axis=1).rename("reference_price")
