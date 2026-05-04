"""
process_data.py

Processing pipeline for adjusted close price data. Transforms raw unprocessed
data into components needed for short-volatility portfolio optimization.

Pipeline Steps (in order):
    1. clean       — Load & clean the raw adjusted close CSV (forward-fill NaNs,
                     drop leading NaN rows, remove duplicate dates, ensure
                     monotonically increasing date index, drop weekends/holidays
                     with no data).
    2. logreturns  — Compute daily log returns: R_t = ln(P_t / P_{t-1}).
    3. realvar     — Compute daily realized variance: RV_t = R_t^2.
    4. impliedvar  — Compute rolling implied variance proxy scaled by a variance
                     risk premium factor.
    5. shortvolret — Compute simulated short-vol returns: implied_var - realized_var.

Usage:
    Run the full pipeline:
        python utils/process_data.py --all

    Run individual steps (each step assumes its prerequisite CSVs exist):
        python utils/process_data.py --clean
        python utils/process_data.py --logreturns
        python utils/process_data.py --realvar
        python utils/process_data.py --impliedvar
        python utils/process_data.py --shortvolret

    Combine any subset of steps:
        python utils/process_data.py --clean --logreturns --realvar

    Override global parameters:
        python utils/process_data.py --all --window 42 --premium 1.20

Outputs (all saved to data/processed_data/):
    - daily_log_returns.csv
    - realized_variance.csv
    - implied_var-<WINDOW>window-<PREMIUM_PCT>premium.csv
    - short-vol-returns-<WINDOW>window-<PREMIUM_PCT>premium.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global Configuration
# ---------------------------------------------------------------------------

# Rolling window in trading sessions (1 trading month ~ 21 sessions)
ROLLING_WINDOW: int = 21

# Variance risk premium factor (15% premium → multiply realized vol by 1.15)
VARIANCE_RISK_PREMIUM: float = 1.15

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "data")
_PROCESSED_DIR = os.path.join(_DATA_DIR, "processed_data")
_UNPROCESSED_CSV = os.path.join(_DATA_DIR, "unprocessed_adj_close.csv")


def _ensure_processed_dir() -> None:
    """Create the processed_data output directory if it does not exist."""
    os.makedirs(_PROCESSED_DIR, exist_ok=True)


# ===========================================================================
# Step 1: Clean Data
# ===========================================================================

def clean_data() -> pd.DataFrame:
    """
    Load raw adjusted close prices and apply data-cleaning procedures.

    Cleaning steps performed:
        1. Parse the Date column as datetime and set as index.
        2. Sort index chronologically to guarantee monotonically increasing dates.
        3. Remove any duplicate date entries (keep the first occurrence).
        4. Forward-fill NaN/None values so that missing prices inherit the
           most recent valid observation (standard market-data convention).
        5. Drop any leading rows that are still NaN after forward-fill (these
           would be assets that started trading after the series begins).
        6. Coerce all price columns to float64 to prevent dtype issues
           downstream.

    Returns
    -------
    pd.DataFrame
        Cleaned adjusted close prices indexed by Date.
    """
    print("[Step 1] Cleaning raw adjusted close data...")

    df = pd.read_csv(_UNPROCESSED_CSV, parse_dates=["Date"], index_col="Date")

    # Ensure chronological ordering
    df.sort_index(inplace=True)

    # Remove duplicate dates (keep first observation)
    duplicates_removed = df.index.duplicated(keep="first").sum()
    df = df[~df.index.duplicated(keep="first")]

    # Forward-fill missing values (standard convention for market data gaps)
    nan_before = df.isna().sum().sum()
    df.ffill(inplace=True)
    nan_after_ffill = df.isna().sum().sum()

    # Drop any remaining leading NaN rows (assets with no earlier valid price)
    df.dropna(inplace=True)

    # Coerce to float64 for numerical stability
    df = df.astype(np.float64)

    print(f"    Duplicate dates removed: {duplicates_removed}")
    print(f"    NaN cells forward-filled: {nan_before - nan_after_ffill}")
    print(f"    Remaining NaN rows dropped: {nan_after_ffill}")
    print(f"    Final cleaned shape: {df.shape}")

    return df


# ===========================================================================
# Step 2: Daily Log Returns
# ===========================================================================

def compute_daily_log_returns(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute daily log returns: R_t = ln(P_t / P_{t-1}).

    Parameters
    ----------
    df : pd.DataFrame, optional
        Cleaned adjusted close prices. If None, loads from the unprocessed CSV
        and cleans on the fly.

    Returns
    -------
    pd.DataFrame
        Daily log returns (first row will be NaN and is dropped).
    """
    print("[Step 2] Computing daily log returns...")

    if df is None:
        df = clean_data()

    log_returns = np.log(df / df.shift(1))

    # The first row is NaN by construction — drop it
    log_returns.dropna(inplace=True)

    _ensure_processed_dir()
    out_path = os.path.join(_PROCESSED_DIR, "daily_log_returns.csv")
    log_returns.to_csv(out_path)
    print(f"    Saved daily log returns ({log_returns.shape}) -> {out_path}")

    return log_returns


# ===========================================================================
# Step 3: Realized Variance
# ===========================================================================

def compute_realized_variance(log_returns: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Compute daily realized variance as the square of daily log returns.

    RV_t = R_t^2

    Parameters
    ----------
    log_returns : pd.DataFrame, optional
        Daily log returns. If None, reads from the saved CSV.

    Returns
    -------
    pd.DataFrame
        Daily realized variance.
    """
    print("[Step 3] Computing realized variance...")

    if log_returns is None:
        csv_path = os.path.join(_PROCESSED_DIR, "daily_log_returns.csv")
        log_returns = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    realized_var = log_returns ** 2

    _ensure_processed_dir()
    out_path = os.path.join(_PROCESSED_DIR, "realized_variance.csv")
    realized_var.to_csv(out_path)
    print(f"    Saved realized variance ({realized_var.shape}) -> {out_path}")

    return realized_var


# ===========================================================================
# Step 4: Implied Variance Proxy
# ===========================================================================

def compute_implied_variance(
    realized_var: pd.DataFrame | None = None,
    window: int = ROLLING_WINDOW,
    premium: float = VARIANCE_RISK_PREMIUM,
) -> pd.DataFrame:
    """
    Construct the implied variance proxy using a rolling historical mean of
    realized variance, scaled by a variance risk premium factor.

    Formula:
        implied_var_t = mean(RV_{t-window}...RV_{t-1}) * VARIANCE_RISK_PREMIUM

    The .shift(1) prevents look-ahead bias: today's implied variance is based
    only on information available up to yesterday.

    Parameters
    ----------
    realized_var : pd.DataFrame, optional
        Daily realized variance. If None, reads from saved CSV.
    window : int
        Rolling window size in trading sessions.
    premium : float
        Multiplicative variance risk premium factor (e.g. 1.15 for 15%).

    Returns
    -------
    pd.DataFrame
        Implied variance proxy.
    """
    print(f"[Step 4] Computing implied variance (window={window}, premium={premium})...")

    if realized_var is None:
        csv_path = os.path.join(_PROCESSED_DIR, "realized_variance.csv")
        realized_var = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    implied_var = realized_var.rolling(window=window).mean().shift(1) * premium

    # Drop leading NaN rows produced by the rolling window + shift
    implied_var.dropna(inplace=True)

    _ensure_processed_dir()
    premium_pct = int(round((premium - 1.0) * 100))
    filename = f"implied_var-{window}window-{premium_pct}premium.csv"
    out_path = os.path.join(_PROCESSED_DIR, filename)
    implied_var.to_csv(out_path)
    print(f"    Saved implied variance ({implied_var.shape}) -> {out_path}")

    return implied_var


# ===========================================================================
# Step 5: Simulated Short-Vol Returns
# ===========================================================================

def compute_short_vol_returns(
    implied_var: pd.DataFrame | None = None,
    realized_var: pd.DataFrame | None = None,
    window: int = ROLLING_WINDOW,
    premium: float = VARIANCE_RISK_PREMIUM,
) -> pd.DataFrame:
    """
    Compute simulated short-volatility returns using Ito's lemma approximation.

    The short-vol P&L for a sold straddle is approximated by the difference
    between the implied variance (premium collected) and the realized variance
    (actual volatility experienced):

        short_vol_return_t = implied_var_t - realized_var_t

    Parameters
    ----------
    implied_var : pd.DataFrame, optional
        Implied variance proxy. If None, reads from saved CSV.
    realized_var : pd.DataFrame, optional
        Realized variance. If None, reads from saved CSV.
    window : int
        Rolling window (used for locating the correct implied_var file).
    premium : float
        Variance risk premium factor (used for locating the correct file).

    Returns
    -------
    pd.DataFrame
        Simulated short-vol daily returns.
    """
    print("[Step 5] Computing simulated short-vol returns...")

    premium_pct = int(round((premium - 1.0) * 100))

    if implied_var is None:
        filename = f"implied_var-{window}window-{premium_pct}premium.csv"
        csv_path = os.path.join(_PROCESSED_DIR, filename)
        implied_var = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    if realized_var is None:
        csv_path = os.path.join(_PROCESSED_DIR, "realized_variance.csv")
        realized_var = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")

    # Align indices — implied_var starts later due to rolling window + shift
    common_idx = implied_var.index.intersection(realized_var.index)
    short_vol_returns = implied_var.loc[common_idx] - realized_var.loc[common_idx]

    _ensure_processed_dir()
    filename = f"short-vol-returns-{window}window-{premium_pct}premium.csv"
    out_path = os.path.join(_PROCESSED_DIR, filename)
    short_vol_returns.to_csv(out_path)
    print(f"    Saved short-vol returns ({short_vol_returns.shape}) -> {out_path}")

    return short_vol_returns


# ===========================================================================
# Full Pipeline
# ===========================================================================

def run_full_pipeline(window: int = ROLLING_WINDOW, premium: float = VARIANCE_RISK_PREMIUM) -> None:
    """Execute all processing steps sequentially."""
    print("=" * 70)
    print("  RUNNING FULL PROCESSING PIPELINE")
    print("=" * 70)

    cleaned = clean_data()
    log_returns = compute_daily_log_returns(cleaned)
    realized_var = compute_realized_variance(log_returns)
    implied_var = compute_implied_variance(realized_var, window=window, premium=premium)
    compute_short_vol_returns(implied_var, realized_var, window=window, premium=premium)

    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process adjusted close data into short-vol return components.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Run the full pipeline (steps 1-5 in sequence).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Step 1: Clean raw data (forward-fill, dedup, etc.).",
    )
    parser.add_argument(
        "--logreturns", action="store_true",
        help="Step 2: Compute daily log returns.",
    )
    parser.add_argument(
        "--realvar", action="store_true",
        help="Step 3: Compute realized variance.",
    )
    parser.add_argument(
        "--impliedvar", action="store_true",
        help="Step 4: Compute implied variance proxy.",
    )
    parser.add_argument(
        "--shortvolret", action="store_true",
        help="Step 5: Compute simulated short-vol returns.",
    )
    parser.add_argument(
        "--window", type=int, default=ROLLING_WINDOW,
        help=f"Rolling window in trading sessions (default: {ROLLING_WINDOW}).",
    )
    parser.add_argument(
        "--premium", type=float, default=VARIANCE_RISK_PREMIUM,
        help=f"Variance risk premium factor (default: {VARIANCE_RISK_PREMIUM}).",
    )

    args = parser.parse_args()

    # If no flags provided, print help and exit
    if not any([args.all, args.clean, args.logreturns, args.realvar,
                args.impliedvar, args.shortvolret]):
        parser.print_help()
        sys.exit(1)

    if args.all:
        run_full_pipeline(window=args.window, premium=args.premium)
        return

    # Run individual steps in pipeline order
    if args.clean:
        clean_data()

    if args.logreturns:
        compute_daily_log_returns()

    if args.realvar:
        compute_realized_variance()

    if args.impliedvar:
        compute_implied_variance(window=args.window, premium=args.premium)

    if args.shortvolret:
        compute_short_vol_returns(window=args.window, premium=args.premium)


if __name__ == "__main__":
    main()
