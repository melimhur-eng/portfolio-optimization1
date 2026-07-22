import numpy as np
import pytest

from src.evaluation import evaluate_forecast, mae, mape, rmse


def test_mae_zero_for_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0])
    assert mae(y, y) == 0


def test_rmse_matches_manual_calc():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    expected = np.sqrt(np.mean([(1 - 2) ** 2, 0, (3 - 2) ** 2]))
    assert rmse(y_true, y_pred) == pytest.approx(expected)


def test_mape_percentage_scale():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    expected = np.mean([10 / 100, 20 / 200]) * 100
    assert mape(y_true, y_pred) == pytest.approx(expected)


def test_evaluate_forecast_returns_all_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([11.0, 19.0, 31.0])
    metrics = evaluate_forecast(y_true, y_pred)
    assert set(metrics.keys()) == {"MAE", "RMSE", "MAPE"}
    assert metrics["MAE"] == pytest.approx(1.0)
