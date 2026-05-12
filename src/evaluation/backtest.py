from __future__ import annotations

import numpy as np
import pandas as pd


def build_strategy_returns(
    predictions: pd.DataFrame,
    long_threshold: float = 0.60,
    short_threshold: float = 0.40,
    transaction_cost_bps: float = 2.0,
    slippage_bps: float = 1.0,
) -> pd.DataFrame:
    out = predictions.copy()
    out["position"] = 0
    out.loc[out["prob_up"] > long_threshold, "position"] = 1
    out.loc[out["prob_up"] < short_threshold, "position"] = -1
    out["turnover"] = out["position"].diff().abs().fillna(out["position"].abs())
    cost = out["turnover"] * (transaction_cost_bps + slippage_bps) / 10000.0
    out["strategy_return"] = out["position"] * out["future_return_15m"] - cost
    out["cumulative_return"] = out["strategy_return"].fillna(0.0).cumsum()
    return out


def financial_metrics(backtest: pd.DataFrame) -> dict[str, float]:
    returns = backtest["strategy_return"].dropna()
    downside = returns[returns < 0]
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    equity = returns.cumsum()
    drawdown = equity - equity.cummax()
    periods_per_year = 365 * 24 * 4
    return {
        "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(periods_per_year)) if returns.std() else 0.0,
        "sortino_ratio": float(returns.mean() / downside.std() * np.sqrt(periods_per_year)) if downside.std() else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else float("inf"),
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else 0.0,
        "hit_rate": float((returns > 0).mean()) if len(returns) else 0.0,
        "turnover": float(backtest["turnover"].mean()) if "turnover" in backtest else 0.0,
    }
