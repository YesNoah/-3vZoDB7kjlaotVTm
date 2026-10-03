"""LSTM forecasts of the next two closing prices.

The network is trained once, on windows whose targets fall on or before
``train_end``, and every forecast uses only the ``window`` days up to
the decision day. Two input/target encodings are available:

- ``"level"``: prices scaled to 0-1 with the training period's min and
  max, predicting the scaled price (the encoding of Value_Notebook.03).
- ``"return"``: daily log returns divided by the training period's
  standard deviation, predicting the log return to each target day.
"""
import numpy as np
import pandas as pd

HORIZON = 2


def make_features(close, train_end, target="return", window=60):
    """Inputs, targets and the inverse transform for every possible day.

    Returns ``(X, Y, to_price)``. Row ``t`` of ``X`` holds the ``window``
    encoded values ending on day t. Row ``t`` of ``Y`` holds the encoded
    prices for the next ``HORIZON`` trading days (NaN where unknown).
    ``to_price(t, y)`` turns an encoded forecast made on day t back into
    prices. Scaling uses only prices up to ``train_end``.
    """
    train = close.loc[:train_end]
    log_price = np.log(close)

    if target == "level":
        low, high = train.min(), train.max()
        encoded = (close - low) / (high - low)
        future = [encoded.shift(-h) for h in range(1, HORIZON + 1)]

        def to_price(t, y):
            return low + np.asarray(y) * (high - low)

    elif target == "return":
        sigma = np.log(train).diff().std()
        encoded = log_price.diff() / sigma
        future = [(log_price.shift(-h) - log_price) / sigma
                  for h in range(1, HORIZON + 1)]

        def to_price(t, y):
            return close.loc[t] * np.exp(np.asarray(y) * sigma)

    else:
        raise ValueError(f"unknown target {target!r}")

    lagged = {f"x{k}": encoded.shift(window - 1 - k) for k in range(window)}
    X = pd.DataFrame(lagged).dropna()
    Y = pd.concat(future, axis=1, keys=[f"y{h}" for h in
                                        range(1, HORIZON + 1)])
    return X, Y.reindex(X.index), to_price


def training_rows(X, Y, close, train_end):
    """Days whose targets are all known by ``train_end``."""
    last_target = close.index.to_series().shift(-HORIZON).reindex(X.index)
    keep = (last_target <= pd.Timestamp(train_end)) & Y.notna().all(axis=1)
    return X.index[keep]


def build_model(window, units=50):
    import keras

    model = keras.Sequential([
        keras.Input(shape=(window, 1)),
        keras.layers.LSTM(units, return_sequences=True),
        keras.layers.LSTM(units),
        keras.layers.Dense(HORIZON),
    ])
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def lstm_forecasts(close, train_end, decision_dates, target="return",
                   window=60, units=50, epochs=100, batch_size=32, seed=0):
    """Train once and forecast ``h1``/``h2`` from each decision day.

    The result has the same columns as ``prophet_forecast.walk_forward``,
    with ``h0`` set to the actual close on the decision day.
    """
    import keras
    import tensorflow as tf

    # Same seed, same model: fixes weight initialisation and shuffling
    # and makes TensorFlow's CPU kernels deterministic.
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    # Building a new model per call makes Keras warn about "retracing".
    tf.get_logger().setLevel("ERROR")
    X, Y, to_price = make_features(close, train_end, target, window)
    rows = training_rows(X, Y, close, train_end)
    model = build_model(window, units)
    model.fit(X.loc[rows].to_numpy()[..., None], Y.loc[rows].to_numpy(),
              epochs=epochs, batch_size=batch_size, verbose=0)

    # Keep days that have a full window behind them and HORIZON trading
    # days after them, as walk_forward does.
    dates = pd.DatetimeIndex(decision_dates)
    position = close.index.get_indexer(dates)
    dates = dates[dates.isin(X.index) & (position + HORIZON < len(close))]
    encoded = model.predict(X.loc[dates].to_numpy()[..., None], verbose=0)
    prices = np.array([to_price(t, y) for t, y in zip(dates, encoded)])

    forecasts = pd.DataFrame(prices, index=dates,
                             columns=[f"h{h}" for h in range(1, HORIZON + 1)])
    forecasts.insert(0, "h0", close.reindex(dates).to_numpy())
    forecasts["fitted_on"] = pd.Timestamp(train_end)
    return forecasts
