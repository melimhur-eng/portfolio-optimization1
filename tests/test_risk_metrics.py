import numpy as np
import pandas as pd
import pytest

from src.risk_metrics import (
    annualized_return,
    annualized_volatility,
    historical_var,
    max_drawdown,
    risk_summary,
    sharpe_ratio,
)


def test_historical_var_positive_for_typical_returns():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.02, 1000))
    var95 = historical_var(returns, confidence=0.95)
    assert var95 > 0


def test_historical_var_known_distribution():
    returns = pd.Series(np.linspace(-0.10, 0.10, 1001))
    var95 = historical_var(returns, confidence=0.95)
    assert var95 == pytest.approx(0.09, abs=1e-2)


def test_sharpe_ratio_zero_vol_returns_nan():
    returns = pd.Series([0.0] * 10)
    assert np.isnan(sharpe_ratio(returns))


def test_sharpe_ratio_positive_for_positive_drift():
    returns = pd.Series([0.01] * 100)
    assert sharpe_ratio(returns) > 0


def test_annualized_return_scales_by_trading_days():
    returns = pd.Series([0.001] * 10)
    assert annualized_return(returns) == pytest.approx(0.001 * 252)


def test_annualized_volatility_scales_by_sqrt_trading_days():
    returns = pd.Series(np.random.default_rng(1).normal(0, 0.01, 500))
    assert annualized_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_max_drawdown_on_known_path():
    # max_drawdown is a negative number: (trough / peak) - 1
    equity = pd.Series([1.0, 1.2, 0.9, 1.5, 0.75])
    dd = max_drawdown(equity)
    assert dd == pytest.approx(0.75 / 1.5 - 1)


def test_risk_summary_contains_expected_keys():
    returns = pd.Series(np.random.default_rng(2).normal(0.0005, 0.02, 300))
    summary = risk_summary(returns)
    assert set(summary.keys()) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "historical_var_95",
        "max_drawdown",
    }
