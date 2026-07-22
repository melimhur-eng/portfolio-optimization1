"""Foundational risk metrics: historical VaR and Sharpe ratio."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical (empirical) daily Value at Risk at the given confidence level.

    Returned as a positive number representing the magnitude of the loss threshold.
    """
    returns = returns.dropna()
    var_quantile = 1 - confidence
    return -np.percentile(returns, var_quantile * 100)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = TRADING_DAYS) -> float:
    """Annualized Sharpe ratio from a series of periodic (daily) returns."""
    returns = returns.dropna()
    excess = returns - risk_free_rate / periods_per_year
    if excess.std() == 0:
        return np.nan
    return (excess.mean() / excess.std()) * np.sqrt(periods_per_year)


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> float:
    returns = returns.dropna()
    return returns.std() * np.sqrt(periods_per_year)


def max_drawdown(cumulative: pd.Series) -> float:
    """Maximum drawdown of a cumulative growth-of-$1 series."""
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return drawdown.min()


def risk_summary(returns: pd.Series, risk_free_rate: float = 0.0, confidence: float = 0.95) -> dict:
    equity = (1 + returns.dropna()).cumprod()
    return {
        "annualized_return": annualized_return(returns),
        "annualized_volatility": annualized_volatility(returns),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate),
        "historical_var_95": historical_var(returns, confidence),
        "max_drawdown": max_drawdown(equity),
    }
