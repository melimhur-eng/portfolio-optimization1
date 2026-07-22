import numpy as np
import pandas as pd
import pytest

from src.backtesting import compute_metrics, simulate_portfolio


def _flat_price_df(n_days=30):
    dates = pd.date_range("2025-01-01", periods=n_days, freq="B")
    return pd.DataFrame({"A": np.full(n_days, 100.0), "B": np.full(n_days, 50.0)}, index=dates)


def test_simulate_portfolio_flat_prices_holds_value():
    prices = _flat_price_df()
    equity = simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, rebalance=None)
    assert equity.iloc[0] == pytest.approx(1.0)
    assert equity.iloc[-1] == pytest.approx(1.0)


def test_simulate_portfolio_growth_matches_weighted_return():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [50.0, 50.0, 50.0]}, index=dates)
    equity = simulate_portfolio(prices, {"A": 1.0, "B": 0.0}, rebalance=None)
    assert equity.iloc[-1] == pytest.approx(1.21)


def test_simulate_portfolio_rebalance_matches_buyhold_when_single_asset():
    prices = _flat_price_df()
    equity_hold = simulate_portfolio(prices, {"A": 1.0, "B": 0.0}, rebalance=None)
    equity_rebal = simulate_portfolio(prices, {"A": 1.0, "B": 0.0}, rebalance="M")
    pd.testing.assert_series_equal(equity_hold, equity_rebal)


def test_compute_metrics_on_flat_equity():
    equity = pd.Series([1.0] * 10, index=pd.date_range("2025-01-01", periods=10, freq="B"))
    metrics = compute_metrics(equity)
    assert metrics["total_return"] == pytest.approx(0.0)
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_compute_metrics_max_drawdown_detects_dip():
    equity = pd.Series([1.0, 1.2, 0.6, 1.1], index=pd.date_range("2025-01-01", periods=4, freq="B"))
    metrics = compute_metrics(equity)
    assert metrics["max_drawdown"] == pytest.approx(0.6 / 1.2 - 1)
