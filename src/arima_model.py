"""ARIMA / SARIMA modeling helpers built on pmdarima + statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pmdarima as pm


def fit_auto_arima(
    train: pd.Series,
    seasonal: bool = False,
    m: int = 1,
    max_p: int = 5,
    max_q: int = 5,
    max_d: int = 2,
    trace: bool = True,
) -> pm.arima.ARIMA:
    """Grid-search ARIMA/SARIMA order via pmdarima's auto_arima and fit on train."""
    model = pm.auto_arima(
        train,
        seasonal=seasonal,
        m=m if seasonal else 1,
        max_p=max_p,
        max_q=max_q,
        max_d=max_d,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        trace=trace,
    )
    return model


def forecast(model: pm.arima.ARIMA, n_periods: int, alpha: float = 0.05):
    """Return point forecast and confidence interval as (mean, lower, upper) Series-like arrays."""
    mean, conf_int = model.predict(n_periods=n_periods, return_conf_int=True, alpha=alpha)
    return np.asarray(mean), conf_int[:, 0], conf_int[:, 1]


def forecast_with_index(model: pm.arima.ARIMA, index: pd.DatetimeIndex, alpha: float = 0.05) -> pd.DataFrame:
    """Forecast over a given future DatetimeIndex, returning a tidy DataFrame."""
    mean, lower, upper = forecast(model, n_periods=len(index), alpha=alpha)
    return pd.DataFrame({"forecast": mean, "lower": lower, "upper": upper}, index=index)
