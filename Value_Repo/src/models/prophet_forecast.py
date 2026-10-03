"""Walk-forward Prophet forecasts and the trading rule built on them.

Each forecast is made from prices up to and including the decision day
only, for that day and the next trading days.
"""
import logging

import numpy as np
import pandas as pd


def prophet_fitter(**params):
    """Return ``fit(history) -> predict(dates)`` using Prophet."""
    # Prophet logs an error at import when plotly is missing; it is only
    # needed for Prophet's own interactive plots.
    logging.getLogger("prophet.plot").disabled = True
    from prophet import Prophet

    # cmdstanpy adds a handler printing two INFO lines per fit unless the
    # logger already has one; a NullHandler keeps warnings and errors.
    stan_log = logging.getLogger("cmdstanpy")
    stan_log.addHandler(logging.NullHandler())
    stan_log.setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    def fit(history):
        model = Prophet(**params)
        model.fit(pd.DataFrame({"ds": history.index, "y": history.values}))

        def predict(dates):
            future = pd.DataFrame({"ds": dates})
            return model.predict(future)["yhat"].to_numpy()

        return predict

    return fit


def walk_forward(close, decision_dates, fit, horizon=2, refit_every=1):
    """Forecast prices from each decision day onward.

    Row ``t`` holds ``h0`` (the model's value for day ``t`` itself) and
    ``h1``..``h<horizon>`` for the following trading days, using a model
    fitted on prices up to ``t`` at the latest. With ``refit_every=1``
    the model is refitted every day; with ``refit_every=None`` it is
    fitted once, on the first decision day, and reused.
    """
    positions = close.index.get_indexer(pd.DatetimeIndex(decision_dates))
    if (positions < 0).any():
        raise ValueError("decision dates must be trading days in close")

    rows, fitted_on, predict = {}, {}, None
    for n, i in enumerate(positions):
        dates = close.index[i:i + horizon + 1]
        if len(dates) < horizon + 1:
            break
        if predict is None or (refit_every and n % refit_every == 0):
            predict = fit(close.iloc[:i + 1])
            last_fit = close.index[i]
        rows[close.index[i]] = predict(dates)
        fitted_on[close.index[i]] = last_fit

    columns = [f"h{h}" for h in range(horizon + 1)]
    forecasts = pd.DataFrame.from_dict(rows, orient="index", columns=columns)
    forecasts["fitted_on"] = pd.Series(fitted_on)
    return forecasts


def forecast_signal(close, forecasts, lag=1):
    """Long when the forecast rises over the day the position is held.

    With the backtest's ``lag``, a decision on day t holds the stock
    from the close of t+lag to the close of t+lag+1, so the rule
    compares the forecasts for those two days.
    """
    up = forecasts[f"h{lag + 1}"] > forecasts[f"h{lag}"]
    return up.astype(float).reindex(close.index).fillna(0.0)


def forecast_series(close, forecasts):
    """Each next-day forecast ``h1``, placed on the day it is for.

    The value on day d was made at the close of the trading day before d.
    """
    targets = close.index[close.index.get_indexer(forecasts.index) + 1]
    return pd.Series(forecasts["h1"].to_numpy(), index=targets)


def rule_on_forecasts(close, forecasts, rule):
    """Apply an indicator rule to the forecast series instead of prices.

    The rule's state on day d uses forecasts up to d, all of which are
    known at the close of the day before d, so it becomes the decision
    for that earlier day.
    """
    state = rule(forecast_series(close, forecasts))
    return state.shift(-1).reindex(close.index).fillna(0.0)


def forecast_accuracy(close, forecasts):
    """One-day-ahead accuracy of ``h1`` against "tomorrow = today"."""
    nxt = close.shift(-1).reindex(forecasts.index)
    today = close.reindex(forecasts.index)
    actual_move = np.sign(nxt - today)
    predicted_move = np.sign(forecasts["h1"] - forecasts["h0"])
    return {
        "mape_model": (abs(forecasts["h1"] - nxt) / nxt).mean(),
        "mape_naive": (abs(today - nxt) / nxt).mean(),
        "direction_hit_rate": (predicted_move == actual_move).mean(),
        "up_day_share": (actual_move > 0).mean(),
    }
