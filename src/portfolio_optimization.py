"""Modern Portfolio Theory helpers: expected returns, covariance, and the efficient frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt import risk_models

TRADING_DAYS = 252


def historical_annual_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    return returns.mean() * periods_per_year


def sample_covariance(returns_df: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Annualized sample covariance matrix from a DataFrame of daily returns."""
    return risk_models.sample_cov(returns_df, returns_data=True, frequency=periods_per_year)


def build_expected_returns(
    returns_df: pd.DataFrame,
    forecast_overrides: dict[str, float],
    periods_per_year: int = TRADING_DAYS,
) -> pd.Series:
    """Expected annual return per asset: use `forecast_overrides` where given (e.g. TSLA's
    model-derived forecast), otherwise fall back to the historical annualized mean."""
    mu = returns_df.mean() * periods_per_year
    for ticker, value in forecast_overrides.items():
        mu[ticker] = value
    return mu


def efficient_frontier_points(mu: pd.Series, cov: pd.DataFrame, n_points: int = 100) -> pd.DataFrame:
    """Sweep target returns to trace the efficient frontier (volatility, return) pairs."""
    lows, highs = mu.min(), mu.max()
    targets = np.linspace(lows, highs, n_points)
    vols, rets = [], []
    for target in targets:
        try:
            ef = EfficientFrontier(mu, cov)
            ef.efficient_return(target_return=target)
            ret, vol, _ = ef.portfolio_performance()
            vols.append(vol)
            rets.append(ret)
        except Exception:
            continue
    return pd.DataFrame({"volatility": vols, "return": rets})


def max_sharpe_portfolio(mu: pd.Series, cov: pd.DataFrame, risk_free_rate: float = 0.0) -> dict:
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}


def min_volatility_portfolio(mu: pd.Series, cov: pd.DataFrame) -> dict:
    ef = EfficientFrontier(mu, cov)
    ef.min_volatility()
    weights = ef.clean_weights()
    ret, vol, sharpe = ef.portfolio_performance()
    return {"weights": weights, "return": ret, "volatility": vol, "sharpe": sharpe}
