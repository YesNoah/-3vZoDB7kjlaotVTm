"""Load saved daily price snapshots and split them into fixed periods.

Every notebook should read prices through ``load_prices`` so that all
results come from the same files, in date order, with the same checks.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"

# Fixed evaluation periods, chosen before looking at any results.
# - train: anything that is fitted (models, scalers) may use only this
# - validation: used for choosing parameters
# - test: looked at once, for the final numbers
SPLITS = {
    "train": ("2010-01-01", "2017-12-31"),
    "validation": ("2018-01-01", "2019-12-31"),
    "test": ("2020-01-01", "2021-03-31"),
}

# The emerging-market sheets in stock_dataXL.xlsx only cover 2020 to
# Q1 2021, too short for a validation period, so nothing may be tuned
# on them.
EM_SPLITS = {
    "train": ("2020-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2021-03-31"),
}

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def load_prices(ticker, raw_dir=RAW_DIR):
    """Return the saved Yahoo Finance snapshot for ``ticker``.

    The result is indexed by date, sorted oldest-first, and checked for
    duplicate dates, missing columns and non-positive prices.
    """
    path = Path(raw_dir) / f"{ticker}.csv"
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    _check(df[["Open", "High", "Low", "Close", "Adj Close"]], path)
    return df


def load_investing_sheets(path=RAW_DIR / "stock_dataXL.xlsx"):
    """Return {sheet name: prices} from an investing.com Excel export.

    Each sheet lists prices newest-first with a text summary row at the
    bottom. That row is dropped, rows are sorted oldest-first and the
    ``Price`` column is renamed ``Close``. These prices are not adjusted
    for dividends.
    """
    result = {}
    for name, sheet in pd.read_excel(path, sheet_name=None).items():
        dates = pd.to_datetime(sheet["Date"], errors="coerce")
        df = sheet[dates.notna()].set_index(dates[dates.notna()])
        df = df[["Price", "Open", "High", "Low"]].apply(pd.to_numeric)
        df = df.rename(columns={"Price": "Close"}).sort_index()
        df.index.name = "Date"
        _check(df, f"{path} [{name}]")
        result[name.strip()] = df
    return result


def _check(prices, source):
    if prices.index.has_duplicates:
        raise ValueError(f"{source} has duplicate dates")
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError(f"{source} has missing or non-positive prices")


def period(data, name, splits=SPLITS):
    """Return the rows of ``data`` that fall inside the named period."""
    start, end = splits[name]
    return data.loc[start:end]
