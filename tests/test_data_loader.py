import numpy as np
import pandas as pd
import pytest

from src.data_loader import add_returns, clean_asset_data


def _raw_df_with_gap():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-06", "2025-01-07"])
    return pd.DataFrame(
        {
            "Open": [10, 11, np.nan, 13],
            "High": [10, 11, np.nan, 13],
            "Low": [10, 11, np.nan, 13],
            "Close": [10, 11, np.nan, 13],
            "Adj Close": [10, 11, np.nan, 13],
            "Volume": [100, 200, np.nan, 400],
        },
        index=dates,
    )


def test_clean_asset_data_fills_business_day_gaps():
    df = _raw_df_with_gap()
    cleaned = clean_asset_data(df)
    # 2025-01-03 (Fri) is a missing business day not in the source at all -> forward-filled
    business_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
    assert len(cleaned) == len(business_days)


def test_clean_asset_data_no_nans_in_price_columns():
    df = _raw_df_with_gap()
    cleaned = clean_asset_data(df)
    assert not cleaned[["Open", "High", "Low", "Close", "Adj Close"]].isna().any().any()


def test_add_returns_first_value_is_nan():
    dates = pd.date_range("2025-01-01", periods=3, freq="B")
    df = pd.DataFrame({"Adj Close": [100.0, 110.0, 121.0]}, index=dates)
    out = add_returns(df)
    assert pd.isna(out["Daily Return"].iloc[0])
    assert out["Daily Return"].iloc[1] == pytest.approx(0.1)


def test_add_returns_log_return_matches_formula():
    dates = pd.date_range("2025-01-01", periods=2, freq="B")
    df = pd.DataFrame({"Adj Close": [100.0, 110.0]}, index=dates)
    out = add_returns(df)
    assert out["Log Return"].iloc[1] == np.log(110.0 / 100.0)
