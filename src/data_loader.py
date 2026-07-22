"""Fetch and clean historical price data for GMF's portfolio assets via YFinance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["TSLA", "BND", "SPY"]
DEFAULT_START = "2015-01-01"
DEFAULT_END = "2026-06-30"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def fetch_asset_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    """Download OHLCV data for each ticker and return one DataFrame per ticker."""
    data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if df is None or df.empty:
            raise RuntimeError(
                f"No price data returned for {ticker} between {start} and {end}. "
                "Check your internet connection or retry later (Yahoo Finance may be rate-limiting)."
            )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index.name = "Date"
        data[ticker] = df
    return data


def combine_long(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-ticker OHLCV frames into one long DataFrame with a Ticker column."""
    frames = []
    for ticker, df in data.items():
        f = df.copy()
        f["Ticker"] = ticker
        frames.append(f)
    combined = pd.concat(frames).reset_index()
    return combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def clean_asset_data(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes, reindex to a full business-day calendar, and fill gaps."""
    df = df.copy()
    numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="B")
    df = df.reindex(full_index)
    df.index.name = "Date"

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill()
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    df = df.dropna(subset=price_cols)
    return df


def add_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Add daily simple and log return columns."""
    df = df.copy()
    df["Daily Return"] = df[price_col].pct_change()
    df["Log Return"] = np.log(df[price_col] / df[price_col].shift(1))
    return df


def load_and_prepare(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """Full pipeline: fetch -> clean -> add returns -> (optionally) persist to data/processed."""
    raw = fetch_asset_data(tickers, start, end)
    prepared = {}
    for ticker, df in raw.items():
        cleaned = clean_asset_data(df)
        cleaned = add_returns(cleaned)
        prepared[ticker] = cleaned
        if save:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(PROCESSED_DIR / f"{ticker}.csv")
    return prepared


def load_processed(ticker: str) -> pd.DataFrame:
    """Load a previously saved processed CSV for a single ticker."""
    path = PROCESSED_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df
