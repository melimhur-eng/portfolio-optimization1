"""EDA helpers: trend/volatility plots, stationarity testing, outlier detection."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, name: str = "") -> dict:
    """Run the Augmented Dickey-Fuller test and return a summary dict."""
    series = series.dropna()
    result = adfuller(series, autolag="AIC")
    summary = {
        "series": name,
        "adf_statistic": result[0],
        "p_value": result[1],
        "n_lags": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }
    return summary


def print_adf_summary(summary: dict) -> None:
    print(f"ADF Test: {summary['series']}")
    print(f"  ADF statistic : {summary['adf_statistic']:.4f}")
    print(f"  p-value       : {summary['p_value']:.4f}")
    print(f"  # lags used   : {summary['n_lags']}")
    print(f"  # obs used    : {summary['n_obs']}")
    for key, val in summary["critical_values"].items():
        print(f"  critical ({key}): {val:.4f}")
    verdict = "STATIONARY" if summary["is_stationary"] else "NON-STATIONARY"
    print(f"  => {verdict} (alpha=0.05)")


def rolling_stats(series: pd.Series, window: int = 30) -> pd.DataFrame:
    """Rolling mean and std for a return/price series."""
    return pd.DataFrame(
        {
            "rolling_mean": series.rolling(window).mean(),
            "rolling_std": series.rolling(window).std(),
        }
    )


def detect_outliers(series: pd.Series, n_std: float = 3.0) -> pd.Series:
    """Flag points more than n_std standard deviations from the mean."""
    mean, std = series.mean(), series.std()
    mask = (series - mean).abs() > n_std * std
    return series[mask]


def plot_price_history(prices: dict[str, pd.Series], title: str = "Adjusted Close Price", save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    for ticker, s in prices.items():
        ax.plot(s.index, s.values, label=ticker)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_daily_returns(returns: pd.Series, ticker: str, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(returns.index, returns.values, linewidth=0.6, color="steelblue")
    ax.set_title(f"{ticker}: Daily Percentage Change")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Return")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_rolling_volatility(returns: pd.Series, ticker: str, window: int = 30, save_path=None):
    roll = rolling_stats(returns, window)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(roll.index, roll["rolling_mean"], color="darkorange")
    axes[0].set_title(f"{ticker}: {window}-Day Rolling Mean Return")
    axes[0].grid(alpha=0.3)
    axes[1].plot(roll.index, roll["rolling_std"], color="firebrick")
    axes[1].set_title(f"{ticker}: {window}-Day Rolling Volatility (Std Dev)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
