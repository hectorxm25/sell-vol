"""
backtest.py

Backtest the CVaR-optimized k=5 portfolio against two baselines using the
full out-of-sample short-vol returns history:

    1. Optimized (k=5, CVaR):  N=25, k=5 solution from the MILP solver.
    2. Equal-Weight (k=5):     Top-5 assets by mean return, each at 20%.
    3. Unconstrained (N=25):   Equal-weight across all 25 assets (1/25 each).

Outputs:
    experiments/backtest.csv
        Columns: Date, optimized, equal_weight_k5, unconstrained_n25
        (daily returns), plus cumulative log-return columns.

    experiments/backtest_statistics.csv
        Summary statistics for each strategy.

Usage:
    python utils/backtest.py
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


def load_returns() -> pd.DataFrame:
    """Load the short-vol returns for all 25 assets."""
    path = os.path.join(_PROCESSED_DIR, "short-vol-returns-21window-15premium.csv")
    return pd.read_csv(path, parse_dates=["Date"], index_col="Date")


def load_optimized_portfolio() -> tuple[list[str], dict[str, float]]:
    """Load the N=25, k=5 optimal solution."""
    path = os.path.join(_EXPERIMENTS_DIR, "N25", "k5", "solution.json")
    with open(path, "r") as f:
        sol = json.load(f)
    return sol["selected_assets"], sol["weights"]


def compute_statistics(daily_returns: np.ndarray, label: str) -> dict:
    """
    Compute key performance statistics for a backtest return stream.

    Metrics:
        - Total cumulative return (compounded)
        - Annualized return (252 trading days)
        - Annualized volatility
        - Sharpe ratio (assuming 0 risk-free rate for short-vol)
        - Maximum drawdown
        - Skewness
        - Excess kurtosis
        - CVaR at 95% (empirical)
        - Calmar ratio (annualized return / max drawdown)
        - Win rate (% of days with positive return)
    """
    n = len(daily_returns)
    cum = np.cumprod(1 + daily_returns)

    total_return = cum[-1] - 1
    ann_return = (1 + total_return) ** (252 / n) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

    # Maximum drawdown
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_dd = np.min(drawdowns)

    # CVaR at 95%
    sorted_returns = np.sort(daily_returns)
    cutoff = int(np.floor(0.05 * n))
    cvar_95 = sorted_returns[:cutoff].mean() if cutoff > 0 else sorted_returns[0]

    # Skewness and kurtosis
    from scipy import stats as sp_stats
    skew = sp_stats.skew(daily_returns)
    kurt = sp_stats.kurtosis(daily_returns)

    # Calmar ratio
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.inf

    # Win rate
    win_rate = np.mean(daily_returns > 0)

    return {
        "strategy": label,
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility_pct": ann_vol * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calmar,
        "cvar_95_daily": cvar_95,
        "skewness": skew,
        "excess_kurtosis": kurt,
        "win_rate_pct": win_rate * 100,
        "num_days": n,
    }


def run_backtest() -> pd.DataFrame:
    """
    Execute the full backtest and save results.

    Returns
    -------
    pd.DataFrame
        The backtest statistics summary.
    """
    print("=" * 70)
    print("  PORTFOLIO BACKTEST")
    print("=" * 70)

    # Load data
    returns_df = load_returns()
    all_assets = returns_df.columns.tolist()
    N = len(all_assets)
    print(f"  Data: {returns_df.shape[0]} days, {N} assets")
    print(f"  Period: {returns_df.index[0].date()} to {returns_df.index[-1].date()}")

    # --- Strategy 1: CVaR-Optimized (k=5) ---
    opt_assets, opt_weights = load_optimized_portfolio()
    opt_w = np.array([opt_weights[a] for a in opt_assets])
    opt_daily = returns_df[opt_assets].values @ opt_w
    print(f"\n  [1] Optimized (k=5 CVaR): {opt_assets}")
    print(f"      Weights: {[f'{w:.4f}' for w in opt_w]}")

    # --- Strategy 2: Equal-Weight Top-5 ---
    mean_rets = returns_df.mean().sort_values(ascending=False)
    ew5_assets = mean_rets.head(5).index.tolist()
    ew5_w = np.ones(5) / 5
    ew5_daily = returns_df[ew5_assets].values @ ew5_w
    print(f"\n  [2] Equal-Weight k=5 (top-5 by mean): {ew5_assets}")
    print(f"      Weights: [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]")

    # --- Strategy 3: Unconstrained (all 25, equal-weight) ---
    unc_w = np.ones(N) / N
    unc_daily = returns_df.values @ unc_w
    print(f"\n  [3] Unconstrained (N=25 equal-weight): all {N} assets")
    print(f"      Weight per asset: {1/N:.4f}")

    # --- Build backtest DataFrame ---
    backtest_df = pd.DataFrame({
        "optimized": opt_daily,
        "equal_weight_k5": ew5_daily,
        "unconstrained_n25": unc_daily,
    }, index=returns_df.index)
    backtest_df.index.name = "Date"

    # Cumulative log returns (sum of log(1+r) for each day)
    backtest_df["optimized_cum_logret"] = np.log(1 + backtest_df["optimized"]).cumsum()
    backtest_df["equal_weight_k5_cum_logret"] = np.log(1 + backtest_df["equal_weight_k5"]).cumsum()
    backtest_df["unconstrained_n25_cum_logret"] = np.log(1 + backtest_df["unconstrained_n25"]).cumsum()

    # Save backtest time series
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)
    backtest_path = os.path.join(_EXPERIMENTS_DIR, "backtest.csv")
    backtest_df.to_csv(backtest_path)
    print(f"\n  Backtest time series saved -> {backtest_path}")

    # --- Compute Statistics ---
    stats_opt = compute_statistics(opt_daily, "Optimized (k=5, CVaR)")
    stats_ew5 = compute_statistics(ew5_daily, "Equal-Weight (k=5)")
    stats_unc = compute_statistics(unc_daily, "Unconstrained (N=25)")

    stats_df = pd.DataFrame([stats_opt, stats_ew5, stats_unc])
    stats_df.set_index("strategy", inplace=True)

    stats_path = os.path.join(_EXPERIMENTS_DIR, "backtest_statistics.csv")
    stats_df.to_csv(stats_path)
    print(f"  Backtest statistics saved -> {stats_path}")

    # --- Print Report ---
    print("\n" + "=" * 70)
    print("  BACKTEST RESULTS")
    print("=" * 70)

    col_w = 18
    header = f"  {'Metric':<30s} {'Optimized':>{col_w}s} {'EW k=5':>{col_w}s} {'Uncons. N=25':>{col_w}s}"
    print(header)
    print("  " + "-" * (30 + 3 * col_w + 2))

    metrics = [
        ("Total Return (%)", "total_return_pct", ".4f"),
        ("Annualized Return (%)", "annualized_return_pct", ".4f"),
        ("Annualized Volatility (%)", "annualized_volatility_pct", ".4f"),
        ("Sharpe Ratio", "sharpe_ratio", ".4f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".4f"),
        ("Calmar Ratio", "calmar_ratio", ".4f"),
        ("CVaR 95% (daily)", "cvar_95_daily", ".6e"),
        ("Skewness", "skewness", ".4f"),
        ("Excess Kurtosis", "excess_kurtosis", ".4f"),
        ("Win Rate (%)", "win_rate_pct", ".2f"),
    ]

    for label, key, fmt in metrics:
        v1 = format(stats_opt[key], fmt)
        v2 = format(stats_ew5[key], fmt)
        v3 = format(stats_unc[key], fmt)
        print(f"  {label:<30s} {v1:>{col_w}s} {v2:>{col_w}s} {v3:>{col_w}s}")

    print("=" * 70)

    return stats_df


if __name__ == "__main__":
    run_backtest()
