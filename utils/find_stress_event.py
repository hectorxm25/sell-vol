"""
find_stress_event.py

Identifies the worst stress event window in the short-vol returns data and
computes cumulative returns for both the CVaR-optimized portfolio and a naive
equal-weighted portfolio over that window.

This script:
    1. Loads the short-vol returns and the N=25, k=5 optimal solution.
    2. Constructs a naive equal-weighted portfolio over the top-5 assets by
       mean return (to represent what an unsophisticated short-vol trader
       might do without CVaR constraints).
    3. Scans the full history to find the worst 2-3 month drawdown window
       for the naive portfolio (the period where naive short-vol gets hit
       hardest).
    4. Computes daily and cumulative returns for both portfolios over that
       stress window.
    5. Saves the results to experiments/tail_event_data.csv.

Usage:
    python utils/find_stress_event.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
_PROCESSED_DIR = os.path.join(_PROJECT_DIR, "data", "processed_data")
_EXPERIMENTS_DIR = os.path.join(_PROJECT_DIR, "experiments")

# Stress window search parameters (in trading days)
_MIN_WINDOW_DAYS = 40   # ~2 months of trading days
_MAX_WINDOW_DAYS = 65   # ~3 months of trading days


def load_optimal_portfolio() -> tuple[list[str], dict[str, float]]:
    """
    Load the N=25, k=5 optimal portfolio solution.

    Returns
    -------
    selected_assets : list[str]
        The 5 selected ticker symbols.
    weights : dict[str, float]
        Mapping of ticker -> optimal weight.
    """
    solution_path = os.path.join(_EXPERIMENTS_DIR, "N25", "k5", "solution.json")
    with open(solution_path, "r") as f:
        solution = json.load(f)

    return solution["selected_assets"], solution["weights"]


def build_naive_portfolio(returns_df: pd.DataFrame, k: int = 5) -> tuple[list[str], dict[str, float]]:
    """
    Construct a naive equal-weighted portfolio from the top-k assets ranked
    by mean historical return. This represents an unsophisticated approach:
    pick the highest-returning assets and weight them equally, with no
    consideration of tail risk or correlation structure.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Full short-vol returns data (all 25 assets).
    k : int
        Number of assets to include.

    Returns
    -------
    selected_assets : list[str]
        The k selected ticker symbols.
    weights : dict[str, float]
        Equal weight (1/k) for each selected asset.
    """
    mean_returns = returns_df.mean().sort_values(ascending=False)
    top_k = mean_returns.head(k).index.tolist()
    weight = 1.0 / k
    weights = {asset: weight for asset in top_k}
    return top_k, weights


def compute_portfolio_returns(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """
    Compute daily portfolio returns given asset returns and weights.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns for all assets.
    weights : dict[str, float]
        Mapping of ticker -> portfolio weight.

    Returns
    -------
    pd.Series
        Daily portfolio return series.
    """
    assets = list(weights.keys())
    w = np.array([weights[a] for a in assets])
    return returns_df[assets].values @ w


def find_worst_drawdown_window(
    portfolio_returns: pd.Series,
    dates: pd.DatetimeIndex,
    min_days: int = _MIN_WINDOW_DAYS,
    max_days: int = _MAX_WINDOW_DAYS,
) -> tuple[int, int, int]:
    """
    Find the contiguous window of min_days to max_days where the cumulative
    return of the portfolio is most negative (worst drawdown).

    We scan all possible windows in [min_days, max_days] and return the one
    with the lowest cumulative sum (worst total loss).

    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Daily portfolio returns.
    dates : pd.DatetimeIndex
        Corresponding dates.
    min_days : int
        Minimum window length.
    max_days : int
        Maximum window length.

    Returns
    -------
    best_start : int
        Start index of the worst window.
    best_end : int
        End index (exclusive) of the worst window.
    best_length : int
        Length of the worst window.
    """
    returns = np.asarray(portfolio_returns)
    n = len(returns)

    best_sum = np.inf
    best_start = 0
    best_end = min_days

    for window_len in range(min_days, max_days + 1):
        # Sliding window sum
        cumsum = np.cumsum(returns)
        # sum of returns[i:i+window_len] = cumsum[i+window_len] - cumsum[i]
        padded = np.concatenate([[0], cumsum])
        window_sums = padded[window_len:] - padded[:n - window_len + 1]

        worst_idx = np.argmin(window_sums)
        if window_sums[worst_idx] < best_sum:
            best_sum = window_sums[worst_idx]
            best_start = worst_idx
            best_end = worst_idx + window_len
            best_length = window_len

    return best_start, best_end, best_length


def run_stress_analysis() -> str:
    """
    Main function: identify stress event and compute portfolio comparisons.

    Returns
    -------
    str
        Path to the saved tail_event_data.csv.
    """
    print("=" * 70)
    print("  TAIL EVENT STRESS TEST ANALYSIS")
    print("=" * 70)

    # Load short-vol returns
    data_path = os.path.join(_PROCESSED_DIR, "short-vol-returns-21window-15premium.csv")
    returns_df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
    print(f"  Loaded returns: {returns_df.shape[0]} days, {returns_df.shape[1]} assets")

    # Load optimized portfolio (N=25, k=5)
    opt_assets, opt_weights = load_optimal_portfolio()
    print(f"  Optimized portfolio: {opt_assets}")
    print(f"  Weights: { {a: f'{w:.4f}' for a, w in opt_weights.items()} }")

    # Build naive equal-weight portfolio (top-5 by mean return)
    naive_assets, naive_weights = build_naive_portfolio(returns_df, k=5)
    print(f"  Naive portfolio (top-5 equal-weight): {naive_assets}")
    print(f"  Weights: { {a: f'{w:.4f}' for a, w in naive_weights.items()} }")

    # Compute daily returns for both portfolios
    opt_daily = compute_portfolio_returns(returns_df, opt_weights)
    naive_daily = compute_portfolio_returns(returns_df, naive_weights)

    # Find the worst stress window based on the naive portfolio's performance
    # (since that's the one we expect to blow up)
    print(f"\n  Scanning for worst {_MIN_WINDOW_DAYS}-{_MAX_WINDOW_DAYS} day window...")
    start_idx, end_idx, window_len = find_worst_drawdown_window(
        naive_daily, returns_df.index
    )

    stress_dates = returns_df.index[start_idx:end_idx]
    print(f"  Worst stress window found:")
    print(f"    Start: {stress_dates[0].strftime('%Y-%m-%d')}")
    print(f"    End:   {stress_dates[-1].strftime('%Y-%m-%d')}")
    print(f"    Length: {window_len} trading days (~{window_len/21:.1f} months)")

    # Extract stress period returns
    opt_stress = opt_daily[start_idx:end_idx]
    naive_stress = naive_daily[start_idx:end_idx]

    # Compute cumulative returns (compounded): (1 + r1)(1 + r2)... - 1
    # Since these are variance-scale returns (very small), simple cumsum is
    # nearly identical to compounding, but we use proper compounding anyway
    opt_cumulative = np.cumprod(1 + opt_stress) - 1
    naive_cumulative = np.cumprod(1 + naive_stress) - 1

    # Build output DataFrame
    output_df = pd.DataFrame({
        "Date": stress_dates,
        "optimized_daily_return": opt_stress,
        "naive_daily_return": naive_stress,
        "optimized_cumulative_return": opt_cumulative,
        "naive_cumulative_return": naive_cumulative,
    })
    output_df.set_index("Date", inplace=True)

    # Save to experiments/
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)
    out_path = os.path.join(_EXPERIMENTS_DIR, "tail_event_data.csv")
    output_df.to_csv(out_path)

    # Print summary statistics
    opt_total = opt_cumulative[-1]
    naive_total = naive_cumulative[-1]
    opt_max_dd = np.min(opt_cumulative)
    naive_max_dd = np.min(naive_cumulative)

    print(f"\n  Results over stress window:")
    print(f"    {'':30s} {'Optimized':>12s} {'Naive':>12s}")
    print(f"    {'Total cumulative return':30s} {opt_total:>12.6f} {naive_total:>12.6f}")
    print(f"    {'Max drawdown':30s} {opt_max_dd:>12.6f} {naive_max_dd:>12.6f}")
    print(f"    {'Daily return std':30s} {np.std(opt_stress):>12.6e} {np.std(naive_stress):>12.6e}")

    print(f"\n  Saved -> {os.path.abspath(out_path)}")
    print("=" * 70)

    return out_path


if __name__ == "__main__":
    run_stress_analysis()
