"""LSTM sequence modeling helpers for single-step-ahead price forecasting."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


def make_sequences(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Turn a 1D array into overlapping (X, y) sequences of length `window`."""
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window : i, 0])
        y.append(values[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def build_lstm(window: int, units: int = 50, dropout: float = 0.2, learning_rate: float = 1e-3) -> keras.Model:
    """A small stacked-LSTM regressor: two LSTM layers + dense output."""
    model = keras.Sequential(
        [
            layers.Input(shape=(window, 1)),
            layers.LSTM(units, return_sequences=True),
            layers.Dropout(dropout),
            layers.LSTM(units // 2),
            layers.Dropout(dropout),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


def prepare_train_test_sequences(
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    window: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Scale on train only, build sequences for train and test (test uses the
    trailing `window` days of train as warm-up context to predict the first test point)."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_prices.reshape(-1, 1))

    X_train, y_train = make_sequences(train_scaled, window)

    full_series = np.concatenate([train_prices, test_prices]).reshape(-1, 1)
    full_scaled = scaler.transform(full_series)
    test_start = len(train_prices) - window
    test_block = full_scaled[test_start:]
    X_test, y_test = make_sequences(test_block, window)

    return X_train, y_train, X_test, y_test, scaler


def iterative_forecast(model: keras.Model, last_window: np.ndarray, n_steps: int, scaler: MinMaxScaler) -> np.ndarray:
    """Iteratively predict `n_steps` ahead, feeding each prediction back as input.

    `last_window` is the last `window` scaled prices, shape (window,).
    Returns unscaled forecasts, shape (n_steps,).
    """
    window = last_window.shape[0]
    current = last_window.reshape(1, window, 1).copy()
    preds_scaled = []
    for _ in range(n_steps):
        next_val = model.predict(current, verbose=0)[0, 0]
        preds_scaled.append(next_val)
        current = np.append(current[:, 1:, :], [[[next_val]]], axis=1)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()
