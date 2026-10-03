import numpy as np
import pandas as pd
import pytest

from src.data.prices import load_prices
from src.evaluation.backtest import backtest, metrics
from src.models import indicator_strategies as strat


def random_walk(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))),
                     index=idx)


STRATEGIES = {
    "sma": strat.sma_crossover,
    "macd": strat.macd_crossover,
    "macd_stop": lambda c: strat.macd_crossover(c, stop=0.025),
    "bollinger": strat.bollinger_reversion,
}


@pytest.mark.parametrize("name", STRATEGIES)
def test_no_lookahead(name):
    """Changing prices after day k must not change anything up to k."""
    close = random_walk()
    k = 400
    altered = close.copy()
    altered.iloc[k + 1:] *= np.linspace(0.5, 2.0, len(close) - k - 1)

    rule = STRATEGIES[name]
    original = backtest(close, rule(close))
    changed = backtest(altered, rule(altered))
    pd.testing.assert_frame_equal(original.iloc[:k + 1],
                                  changed.iloc[:k + 1])


@pytest.mark.parametrize("lag", [0, 1, 2])
def test_position_starts_lag_plus_one_days_after_signal(lag):
    close = random_walk(20)
    state = pd.Series(0.0, index=close.index)
    state.iloc[5:] = 1.0
    bt = backtest(close, state, lag=lag)
    first_long = int(np.argmax(bt["position"].to_numpy() > 0))
    assert first_long == 5 + lag + 1


def test_round_trip_costs_twice_the_rate():
    close = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=10))
    state = pd.Series(0.0, index=close.index)
    state.iloc[2:5] = 1.0
    bt = backtest(close, state, cost_bps=10)
    assert (1 + bt["ret"]).prod() == pytest.approx((1 - 0.001) ** 2)


def test_buy_and_hold_matches_asset():
    close = random_walk(50)
    bt = backtest(close, strat.buy_and_hold(close))
    assert bt["equity"].iloc[-1] == pytest.approx(
        close.iloc[-1] / close.iloc[1])


def test_macd_stop_loss_exits_on_a_drop():
    close = pd.Series(np.r_[np.linspace(100, 80, 40),
                            np.linspace(80, 100, 30), [97.0], [97.5]],
                      index=pd.bdate_range("2020-01-01", periods=72))
    plain = strat.macd_crossover(close)
    stopped = strat.macd_crossover(close, stop=0.025)
    assert plain.iloc[-2] == 1.0
    assert stopped.iloc[-2] == 0.0
    # No re-entry until MACD crosses back down and up again.
    assert stopped.iloc[-1] == 0.0


def test_trade_count_and_win_rate():
    close = pd.Series([100, 100, 110, 110, 100, 100, 90, 90, 100.0],
                      index=pd.bdate_range("2020-01-01", periods=9))
    state = pd.Series([1, 1, 0, 0, 1, 0, 0, 0, 0.0], index=close.index)
    m = metrics(backtest(close, state, lag=0))
    assert m["trades"] == 2
    assert m["win_rate"] == 0.5


def test_load_prices_sorts_and_validates(tmp_path):
    rows = ("Date,Open,High,Low,Close,Adj Close,Volume\n"
            "2020-01-03,2,2,2,2,2,1\n"
            "2020-01-02,1,1,1,1,1,1\n")
    (tmp_path / "X.csv").write_text(rows)
    df = load_prices("X", raw_dir=tmp_path)
    assert df.index.is_monotonic_increasing

    (tmp_path / "Y.csv").write_text(rows + "2020-01-02,1,1,1,1,1,1\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_prices("Y", raw_dir=tmp_path)
