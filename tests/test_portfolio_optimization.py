import numpy as np
import pandas as pd
import pytest

from src.portfolio_optimization import (
    build_expected_returns,
    max_sharpe_portfolio,
    min_volatility_portfolio,
    sample_covariance,
)


@pytest.fixture
def synthetic_returns():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    tsla = rng.normal(0.002, 0.03, 500)
    bnd = rng.normal(0.0002, 0.003, 500)
    spy = rng.normal(0.0006, 0.012, 500)
    return pd.DataFrame({"TSLA": tsla, "BND": bnd, "SPY": spy}, index=dates)


def test_build_expected_returns_applies_override(synthetic_returns):
    mu = build_expected_returns(synthetic_returns, {"TSLA": 0.5})
    assert mu["TSLA"] == 0.5
    assert mu["BND"] != 0.5


def test_sample_covariance_is_positive_semidefinite(synthetic_returns):
    cov = sample_covariance(synthetic_returns)
    eigvals = np.linalg.eigvalsh(cov.values)
    assert (eigvals >= -1e-8).all()


def test_min_volatility_weights_sum_to_one(synthetic_returns):
    mu = build_expected_returns(synthetic_returns, {"TSLA": 0.3})
    cov = sample_covariance(synthetic_returns)
    result = min_volatility_portfolio(mu, cov)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-4)


def test_min_volatility_has_lower_or_equal_vol_than_max_sharpe(synthetic_returns):
    mu = build_expected_returns(synthetic_returns, {"TSLA": 0.3})
    cov = sample_covariance(synthetic_returns)
    min_vol = min_volatility_portfolio(mu, cov)
    max_sharpe = max_sharpe_portfolio(mu, cov)
    assert min_vol["volatility"] <= max_sharpe["volatility"] + 1e-6
