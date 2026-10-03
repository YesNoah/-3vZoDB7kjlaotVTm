"""Long/flat trading rules built on technical indicators.

Each function returns a 0/1 series: 1 means "be long after this day's
close", 0 means "be flat". The value for a day uses only prices up to
and including that day. The backtest decides when the trade happens.
"""
import numpy as np
import pandas as pd

from src.features.build_features import bollinger, macd, sma


def _hold_between(enter, exit_):
    """1 from each ``enter`` day until the next ``exit_`` day, else 0."""
    state = pd.Series(np.nan, index=enter.index)
    state[exit_] = 0.0
    state[enter] = 1.0
    return state.ffill().fillna(0.0)


def buy_and_hold(close):
    return pd.Series(1.0, index=close.index)


def sma_crossover(close, fast=30, slow=100):
    """Long while the fast SMA is above the slow SMA."""
    fast_ma, slow_ma = sma(close, fast), sma(close, slow)
    return _hold_between(fast_ma > slow_ma, fast_ma < slow_ma)


def macd_crossover(close, stop=None, fast=12, slow=26, signal=9):
    """Long while the MACD line is above its signal line.

    With ``stop`` set (for example 0.025), a long position is also
    closed when the close falls more than ``stop`` below the entry-day
    close, or more than ``stop`` below the previous close. After a stop,
    the rule waits for the MACD line to cross back below and then above
    the signal line before buying again.
    """
    m = macd(close, fast, slow, signal)
    bullish = m["macd"] > m["signal"]
    bearish = m["macd"] < m["signal"]
    if stop is None:
        return _hold_between(bullish, bearish)

    state = np.zeros(len(close))
    long, armed, entry = False, True, np.nan
    prices = close.to_numpy()
    for i in range(len(close)):
        if bearish.iloc[i]:
            armed = True
        if long:
            stopped = (prices[i] < entry * (1 - stop)
                       or prices[i] < prices[i - 1] * (1 - stop))
            if bearish.iloc[i] or stopped:
                long = False
                armed = bool(bearish.iloc[i])
        elif armed and bullish.iloc[i]:
            long, entry = True, prices[i]
        state[i] = float(long)
    return pd.Series(state, index=close.index)


def bollinger_reversion(close, length=20, n_std=2.0):
    """Buy below the lower band, sell above the upper band."""
    bands = bollinger(close, length, n_std)
    return _hold_between(close < bands["lower"], close > bands["upper"])
