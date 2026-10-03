import numpy as np
import pandas as pd
import pytest

from src.models.lstm_forecast import (HORIZON, make_features,
                                      training_rows)


def random_walk(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
                     index=idx)


TRAIN_END = "2019-09-30"


@pytest.mark.parametrize("target", ["level", "return"])
def test_inputs_only_use_past_prices(target):
    close = random_walk()
    k = 250
    altered = close.copy()
    altered.iloc[k + 1:] *= 1.5

    X1, _, _ = make_features(close, TRAIN_END, target, window=20)
    X2, _, _ = make_features(altered, TRAIN_END, target, window=20)
    day = close.index[k]
    pd.testing.assert_frame_equal(X1.loc[:day], X2.loc[:day])
    assert not X1.equals(X2)


@pytest.mark.parametrize("target", ["level", "return"])
def test_targets_decode_to_the_next_prices(target):
    close = random_walk()
    X, Y, to_price = make_features(close, TRAIN_END, target, window=20)
    t = X.index[50]
    i = close.index.get_loc(t)
    np.testing.assert_allclose(to_price(t, Y.loc[t]),
                               close.iloc[i + 1:i + 1 + HORIZON])


@pytest.mark.parametrize("target", ["level", "return"])
def test_training_targets_stay_inside_the_training_period(target):
    close = random_walk()
    X, Y, _ = make_features(close, TRAIN_END, target, window=20)
    rows = training_rows(X, Y, close, TRAIN_END)
    last = close.index.get_indexer(rows).max() + HORIZON
    assert close.index[last] <= pd.Timestamp(TRAIN_END)
    assert close.index[last + 1] > pd.Timestamp(TRAIN_END)


def test_window_order_is_oldest_to_newest():
    close = pd.Series(np.arange(1.0, 31.0),
                      index=pd.bdate_range("2020-01-01", periods=30))
    X, _, _ = make_features(close, "2020-02-28", "level", window=5)
    t = X.index[0]
    assert t == close.index[4]
    assert (np.diff(X.loc[t].to_numpy()) > 0).all()
