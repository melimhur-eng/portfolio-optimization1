"""Portfolio backtest simulator: buy-and-hold or periodic rebalancing, vs a benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_portfolio(price_df: pd.DataFrame, weights: dict[str, float], rebalance: str | None = "M") -> pd.Series:
    """Simulate growth of $1 invested at `weights` on the first date of `price_df`.

    rebalance=None: buy-and-hold, weights drift with prices.
    rebalance="M":  rebalance back to target weights at the start of each period (e.g. month).
    """
    tickers = list(weights.keys())
    w_target = np.array([weights[t] for t in tickers])
    px = price_df[tickers]

    equity = pd.Series(index=px.index, dtype=float)
    shares = None
    current_period = None
    value = 1.0

    for date, row in px.iterrows():
        period = date.to_period(rebalance) if rebalance else None
        prices = row.values

        if shares is None or (rebalance and period != current_period):
            shares = (value * w_target) / prices
            current_period = period

        value = float(np.dot(shares, prices))
        equity[date] = value

    return equity


def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> dict:
    """Total return, annualized return, Sharpe ratio, and max drawdown for an equity curve."""
    returns = equity.pct_change().dropna()
    n_years = len(returns) / periods_per_year
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    annualized_return = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = (annualized_return - risk_free_rate) / vol if vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
    }
