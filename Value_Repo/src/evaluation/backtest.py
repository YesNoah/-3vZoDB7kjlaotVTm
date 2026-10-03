"""Daily long/flat backtest and the metrics every notebook reports.

Timing convention: a strategy's state for day t is decided at day t's
close. With ``lag=1`` (the default) the trade happens at the next
day's close, so the position first earns the return of day t+2.
``lag=0`` trades at the same close the signal was computed from, which
is optimistic and is only useful as a comparison.
"""
import numpy as np
import pandas as pd

from src.data.prices import SPLITS, period

TRADING_DAYS = 252


def backtest(close, state, lag=1, cost_bps=0.0):
    """Daily returns of following ``state`` (0/1) on ``close`` prices.

    ``cost_bps`` is charged on every change in position, in basis
    points of the amount traded (so a round trip costs it twice).
    """
    asset_ret = close.pct_change().fillna(0.0)
    position = state.shift(lag + 1).fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())
    strat_ret = position * asset_ret - turnover * cost_bps / 1e4
    return pd.DataFrame({
        "position": position,
        "asset_ret": asset_ret,
        "turnover": turnover,
        "ret": strat_ret,
        "equity": (1 + strat_ret).cumprod(),
    })


def _trade_returns(bt):
    """Return of each trade within ``bt``.

    A position already open on the first row counts as a trade entered
    on that row, and one still open on the last row is marked to that
    row's close.
    """
    pos = bt["position"]
    entries = (pos > 0) & (pos.shift(1, fill_value=0.0) == 0)
    trade_id = entries.cumsum()
    # The exit day has position 0 but carries the exit cost.
    in_trade = (pos > 0) | ((bt["turnover"] > 0) & (trade_id > 0))
    grouped = (1 + bt.loc[in_trade, "ret"]).groupby(trade_id[in_trade])
    return grouped.prod() - 1


def metrics(bt):
    """Summary statistics for one backtest slice."""
    ret = bt["ret"]
    n = len(ret)
    total = (1 + ret).prod() - 1
    equity = (1 + ret).cumprod()
    drawdown = equity / np.maximum(equity.cummax(), 1.0) - 1
    std = ret.std()
    trades = _trade_returns(bt)
    return {
        "total_return": total,
        "cagr": (1 + total) ** (TRADING_DAYS / n) - 1,
        "ann_vol": std * np.sqrt(TRADING_DAYS),
        "sharpe": (ret.mean() / std * np.sqrt(TRADING_DAYS)
                   if std > 0 else np.nan),
        "max_drawdown": drawdown.min(),
        "exposure": bt["position"].mean(),
        "trades": len(trades),
        "win_rate": (trades > 0).mean() if len(trades) else np.nan,
    }


def evaluate(backtests, periods=("train", "validation", "test"),
             splits=SPLITS):
    """Metrics table for several named backtests over several periods."""
    rows = {}
    for period_name in periods:
        for name, bt in backtests.items():
            part = period(bt, period_name, splits)
            if len(part):
                rows[(period_name, name)] = metrics(part)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["period", "strategy"]
    return table
