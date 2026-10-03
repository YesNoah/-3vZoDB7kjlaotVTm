"""Technical indicators computed with plain pandas.

Every indicator uses only the current and earlier rows, so its value on
a given day is known at that day's close.
"""
import pandas as pd


def sma(close, length):
    """Simple moving average over ``length`` rows."""
    return close.rolling(length, min_periods=length).mean()


def ema(close, length):
    """Exponential moving average with span ``length``."""
    return close.ewm(span=length, adjust=False, min_periods=length).mean()


def macd(close, fast=12, slow=26, signal=9):
    """MACD line, signal line and histogram."""
    line = ema(close, fast) - ema(close, slow)
    signal_line = line.ewm(
        span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({
        "macd": line,
        "signal": signal_line,
        "hist": line - signal_line,
    })


def bollinger(close, length=20, n_std=2.0):
    """Lower, middle and upper Bollinger bands (population std)."""
    middle = sma(close, length)
    std = close.rolling(length, min_periods=length).std(ddof=0)
    return pd.DataFrame({
        "lower": middle - n_std * std,
        "middle": middle,
        "upper": middle + n_std * std,
    })
