"""
gather_data.py

Downloads 5 years of historical daily adjusted close prices for each asset
in ASSET_UNIVERSE using the Yahoo Finance API (yfinance). Stores the raw,
unprocessed data as a CSV file in the ../data/ directory.
"""

import os
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Asset universe — fill in your desired ticker symbols here.
# Example: ASSET_UNIVERSE = ["SPY", "QQQ", "IWM", "TLT", "GLD", ...]
# ---------------------------------------------------------------------------
ASSET_UNIVERSE: list[str] = ["SPY", "QQQ", "IWM", "TLT", "LQD", "GLD", "SLV", "USO", "AAPL", "MSFT", "NVDA", 
                             "TSLA", "ARKK", "COIN", "XLE", "CAT", "LMT", "XLF", "JPM", "VNQ", "JNJ", "PG", 
                             "WMT", "DIS", "AMD"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEARS_OF_HISTORY = 5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_FILENAME = "unprocessed_adj_close.csv"


def gather_data() -> pd.DataFrame:
    """
    Download adjusted close prices for all tickers in ASSET_UNIVERSE
    over the last 5 years and return as a DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with one column per asset ticker,
        containing daily adjusted close prices.
    """
    if not ASSET_UNIVERSE:
        raise ValueError(
            "ASSET_UNIVERSE is empty. Please add ticker symbols before running."
        )

    end_date = datetime.today()
    start_date = end_date - timedelta(days=YEARS_OF_HISTORY * 365)

    print(
        f"Downloading adjusted close data from {start_date.date()} to {end_date.date()} "
        f"for {len(ASSET_UNIVERSE)} assets..."
    )

    # yfinance can download multiple tickers in a single call.
    # Setting auto_adjust=False preserves the "Adj Close" column explicitly.
    raw = yf.download(
        tickers=ASSET_UNIVERSE,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=True,
    )

    # When multiple tickers are requested, yf.download returns a MultiIndex
    # DataFrame with (Price, Ticker) columns. Extract only "Adj Close".
    if len(ASSET_UNIVERSE) == 1:
        adj_close = raw[["Adj Close"]].copy()
        adj_close.columns = ASSET_UNIVERSE
    else:
        adj_close = raw["Adj Close"].copy()

    # Ensure column order matches ASSET_UNIVERSE
    adj_close = adj_close[ASSET_UNIVERSE]

    print(f"Downloaded data shape: {adj_close.shape}")
    return adj_close


def save_data(df: pd.DataFrame) -> str:
    """
    Persist the unprocessed adjusted close DataFrame to disk as a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The raw adjusted close price data.

    Returns
    -------
    str
        The absolute path to the saved CSV file.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    df.to_csv(output_path)
    print(f"Unprocessed data saved to: {os.path.abspath(output_path)}")
    return output_path


if __name__ == "__main__":
    df = gather_data()
    save_data(df)
