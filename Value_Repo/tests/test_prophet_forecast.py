import numpy as np
import pandas as pd
import pytest

from src.data.prices import load_investing_sheets
from src.models import indicator_strategies as rules
from src.models.prophet_forecast import (forecast_accuracy, forecast_series,
                                         forecast_signal, rule_on_forecasts,
                                         walk_forward)


def make_close(n=30):
    idx = pd.bdate_range("2021-01-01", periods=n)
    return pd.Series(np.arange(100.0, 100.0 + n), index=idx)


class RecordingFitter:
    """Fake model: predicts the last price it was fitted on plus 1/day."""

    def __init__(self):
        self.fitted_up_to = []

    def __call__(self, history):
        self.fitted_up_to.append(history.index[-1])
        last_date, last_price = history.index[-1], history.iloc[-1]

        def predict(dates):
            days = np.array([(d - last_date).days for d in dates])
            return last_price + days

        return predict


@pytest.mark.parametrize("refit_every", [1, 3, None])
def test_walk_forward_only_uses_past_prices(refit_every):
    close = make_close()
    decisions = close.index[10:]
    fit = RecordingFitter()
    fc = walk_forward(close, decisions, fit, horizon=2,
                      refit_every=refit_every)

    assert (fc["fitted_on"] <= fc.index).all()
    assert all(d in set(decisions) for d in fit.fitted_up_to)
    expected_fits = {1: len(fc), 3: -(-len(fc) // 3), None: 1}[refit_every]
    assert len(fit.fitted_up_to) == expected_fits
    # The last two decision days have no two following trading days.
    assert fc.index[-1] == close.index[-3]


def test_signal_compares_the_holding_day():
    close = make_close(6)
    fc = pd.DataFrame({"h0": [1.0, 5.0], "h1": [2.0, 4.0],
                       "h2": [1.0, 6.0]}, index=close.index[:2])
    lag0 = forecast_signal(close, fc, lag=0)
    lag1 = forecast_signal(close, fc, lag=1)
    assert lag0.iloc[:2].tolist() == [1.0, 0.0]
    assert lag1.iloc[:2].tolist() == [0.0, 1.0]
    assert lag1.iloc[2:].eq(0).all()


def test_forecast_series_is_placed_on_the_forecast_day():
    close = make_close(6)
    fc = pd.DataFrame({"h0": [1.0, 2.0], "h1": [10.0, 20.0],
                       "h2": [0.0, 0.0]}, index=close.index[[1, 2]])
    series = forecast_series(close, fc)
    assert series.index.tolist() == close.index[[2, 3]].tolist()
    assert series.tolist() == [10.0, 20.0]


def momentum_fitter(history):
    """Fake model: extends the average move of the last five days."""
    last_date, last_price = history.index[-1], history.iloc[-1]
    drift = history.diff().iloc[-5:].mean()

    def predict(dates):
        bdays = np.array([np.busday_count(last_date.date(), d.date())
                          for d in dates])
        return last_price + drift * bdays

    return predict


@pytest.mark.parametrize("rule", [rules.bollinger_reversion,
                                  rules.macd_crossover])
def test_rule_on_forecasts_has_no_lookahead(rule):
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", periods=200)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 200)), index=idx)
    k = 150
    altered = close.copy()
    altered.iloc[k + 1:] += 25

    def states(c):
        fc = walk_forward(c, c.index[5:], momentum_fitter)
        return rule_on_forecasts(c, fc, rule)

    original, changed = states(close), states(altered)
    pd.testing.assert_series_equal(original.iloc[:k + 1],
                                   changed.iloc[:k + 1])
    assert not original.equals(changed)


def test_accuracy_perfect_forecast():
    close = make_close(10)
    fc = pd.DataFrame({"h0": close, "h1": close.shift(-1)}).iloc[:-1]
    acc = forecast_accuracy(close, fc)
    assert acc["mape_model"] == 0
    assert acc["direction_hit_rate"] == 1
    assert acc["mape_naive"] > 0


def test_load_investing_sheets(tmp_path):
    sheet = pd.DataFrame({
        "Date": [pd.Timestamp("2021-01-05"), pd.Timestamp("2021-01-04"),
                 "Highest: 2"],
        "Price": [2.0, 1.0, "Lowest: 1"],
        "Open": [2, 1, "x"], "High": [2, 1, "x"], "Low": [2, 1, "x"],
        "Vol.": ["1M", "2M", None], "Change %": [0.1, 0.0, None],
    })
    path = tmp_path / "x.xlsx"
    with pd.ExcelWriter(path) as writer:
        sheet.to_excel(writer, sheet_name="Country - Stock ", index=False)

    data = load_investing_sheets(path)
    df = data["Country - Stock"]
    assert df.index.is_monotonic_increasing
    assert df["Close"].tolist() == [1.0, 2.0]
