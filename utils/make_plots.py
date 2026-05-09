"""
make_plots.py

Generate publication-quality plots from experiment data for the short-volatility
portfolio optimization paper.

==============================================================================
Available Plots
==============================================================================

    1. --fat-tail      Non-Normal Return Distribution (Motivation)
                       Histogram of TSLA simulated short-vol daily returns
                       overlaid with a fitted normal distribution to visually
                       demonstrate the fat left tail that motivates CVaR over
                       mean-variance optimization.

    2. --complexity    Empirical Complexity Curve (Computational Difficulty)
                       Line graph of worst-case solve time vs. universe size N,
                       demonstrating the exponential scaling of Branch-and-Bound
                       for the MILP formulation.

    3. --stress-test   Tail Event Stress Test (Financial Validation)
                       Time-series of cumulative returns during a severe market
                       stress period, comparing the CVaR-optimized portfolio
                       against a naive equal-weighted portfolio. Demonstrates
                       that the optimizer successfully constrains tail risk.

    4. --equity-curve  Cumulative Log-Return Equity Curve (Empirical Results)
                       Full-period equity curve comparing the CVaR-optimized
                       portfolio against two baselines (equal-weight k=5 and
                       unconstrained N=25).

==============================================================================
Usage
==============================================================================

    # Generate all plots:
    python utils/make_plots.py --all

    # Generate individual plots:
    python utils/make_plots.py --fat-tail
    python utils/make_plots.py --complexity
    python utils/make_plots.py --stress-test
    python utils/make_plots.py --equity-curve
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.join(_SCRIPT_DIR, "..")
_VIZ_DIR = os.path.join(_PROJECT_DIR, "visualizations")
_PROCESSED_DIR = os.path.join(_PROJECT_DIR, "data", "processed_data")
_EXPERIMENTS_DIR = os.path.join(_PROJECT_DIR, "experiments")

# ---------------------------------------------------------------------------
# Plot Style Configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def _ensure_viz_dir() -> None:
    """Create the visualizations output directory if it does not exist."""
    os.makedirs(_VIZ_DIR, exist_ok=True)


# ===========================================================================
# Plot 1: Non-Normal Return Distribution (Fat Tail Motivation)
# ===========================================================================

def plot_fat_tail_distribution(asset: str = "TSLA") -> str:
    """
    Plot a histogram of simulated short-vol daily returns for a single asset,
    overlaid with a normal distribution having the same mean and variance.

    This visually demonstrates the fat left tail that standard mean-variance
    (Markowitz) optimization fails to account for, motivating the use of CVaR.

    Parameters
    ----------
    asset : str
        Ticker symbol to plot (default: TSLA).

    Returns
    -------
    str
        Path to the saved figure.
    """
    print(f"[Plot 1] Generating fat-tail distribution plot for {asset}...")

    # Load short-vol returns
    data_path = os.path.join(_PROCESSED_DIR, "short-vol-returns-21window-15premium.csv")
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    if asset not in df.columns:
        raise ValueError(f"Asset '{asset}' not found in data. Available: {list(df.columns)}")

    returns = df[asset].dropna().values

    mu = np.mean(returns)
    sigma = np.std(returns)

    # Compute skewness and kurtosis for annotation
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)  # excess kurtosis (normal = 0)

    print(f"    {asset} short-vol returns: n={len(returns)}")
    print(f"    Mean:     {mu:.6e}")
    print(f"    Std:      {sigma:.6e}")
    print(f"    Skewness: {skewness:.4f} (normal = 0)")
    print(f"    Excess Kurtosis: {kurtosis:.4f} (normal = 0)")

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Histogram of empirical returns
    n_bins = 80
    counts, bin_edges, patches = ax.hist(
        returns, bins=n_bins, density=True, alpha=0.65,
        color="#2196F3", edgecolor="white", linewidth=0.3,
        label=f"{asset} Short-Vol Returns (empirical)",
    )

    # Fitted normal distribution overlay
    x_range = np.linspace(returns.min() - sigma, returns.max() + sigma, 500)
    normal_pdf = stats.norm.pdf(x_range, loc=mu, scale=sigma)
    ax.plot(
        x_range, normal_pdf, color="#E53935", linewidth=2.2,
        linestyle="--", label=f"Normal fit ($\\mu$={mu:.2e}, $\\sigma$={sigma:.2e})",
    )

    # Shade the left tail region beyond -2 sigma for emphasis
    tail_threshold = mu - 2 * sigma
    tail_x = x_range[x_range <= tail_threshold]
    tail_y = stats.norm.pdf(tail_x, loc=mu, scale=sigma)
    ax.fill_between(
        tail_x, tail_y, alpha=0.15, color="#E53935",
        label=f"Normal left tail (< $\\mu - 2\\sigma$)",
    )

    # Annotations
    ax.axvline(mu, color="gray", linestyle=":", linewidth=1, alpha=0.7)

    stats_text = (
        f"Skewness = {skewness:.2f}\n"
        f"Excess Kurtosis = {kurtosis:.2f}\n"
        f"n = {len(returns)} days"
    )
    ax.text(
        0.02, 0.95, stats_text, transform=ax.transAxes,
        verticalalignment="top", fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
    )

    ax.set_xlabel("Daily Short-Volatility Return")
    ax.set_ylabel("Probability Density")
    ax.set_title(
        f"Distribution of Simulated Short-Vol Returns: {asset}\n"
        f"Fat Left Tail Motivates CVaR over Mean-Variance Optimization"
    )
    ax.legend(loc="upper right")

    plt.tight_layout()

    _ensure_viz_dir()
    out_path = os.path.join(_VIZ_DIR, f"fat_tail_distribution_{asset}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved -> {os.path.abspath(out_path)}")
    return out_path


# ===========================================================================
# Plot 2: Empirical Complexity Curve
# ===========================================================================

def plot_complexity_curve() -> str:
    """
    Plot the worst-case solve time for each universe size N, demonstrating
    the empirical exponential complexity of Branch-and-Bound for this MILP.

    Reads from experiments/summary.csv and finds the maximum solve_time_sec
    for each unique N value.

    Returns
    -------
    str
        Path to the saved figure.
    """
    print("[Plot 2] Generating empirical complexity curve...")

    summary_path = os.path.join(_EXPERIMENTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found: {summary_path}\n"
            "Run 'python run_experiments.py' first."
        )

    df = pd.read_csv(summary_path)

    # For each N, find the worst-case (maximum) solve time
    worst_cases = df.loc[df.groupby("N")["solve_time_sec"].idxmax()]
    worst_cases = worst_cases.sort_values("N").reset_index(drop=True)

    print("    Worst-case solve times per N:")
    for _, row in worst_cases.iterrows():
        print(f"      N={int(row['N']):>2}, worst-case k={int(row['k']):>2} -> {row['solve_time_sec']:.2f}s")

    N_vals = worst_cases["N"].values
    times = worst_cases["solve_time_sec"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Main line plot
    ax.plot(
        N_vals, times, "o-", color="#1565C0", markersize=8,
        markerfacecolor="#1E88E5", markeredgecolor="#0D47A1",
        linewidth=2, label="Worst-case solve time",
    )

    # Fill under curve for visual emphasis
    ax.fill_between(N_vals, times, alpha=0.08, color="#1565C0")

    # Annotate each point with its solve time
    for n, t in zip(N_vals, times):
        if t >= 1.0:
            label = f"{t:.1f}s"
        else:
            label = f"{t*1000:.0f}ms"
        ax.annotate(
            label, (n, t), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=9,
            fontweight="bold", color="#0D47A1",
        )

    ax.set_xlabel("Universe Size (N)")
    ax.set_ylabel("CPU Solve Time (seconds)")
    ax.set_title(
        "Empirical Complexity of CVaR Portfolio MILP\n"
        "Branch-and-Bound Solve Time vs. Universe Size"
    )
    ax.set_xticks(N_vals)
    ax.set_xticklabels([str(int(n)) for n in N_vals])

    # Use log scale for y-axis if the range spans multiple orders of magnitude
    if times.max() / max(times.min(), 1e-6) > 50:
        ax.set_yscale("log")
        ax.set_ylabel("CPU Solve Time (seconds, log scale)")

    ax.legend(loc="upper left")

    plt.tight_layout()

    _ensure_viz_dir()
    out_path = os.path.join(_VIZ_DIR, "complexity_curve.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved -> {os.path.abspath(out_path)}")
    return out_path


# ===========================================================================
# Plot 3: Tail Event Stress Test
# ===========================================================================

def plot_stress_test() -> str:
    """
    Plot cumulative returns during a severe stress window for both the
    CVaR-optimized portfolio and a naive equal-weighted portfolio.

    Requires that utils/find_stress_event.py has been run first (or runs it
    automatically if tail_event_data.csv does not exist).

    The plot demonstrates that during a tail event:
    - The naive portfolio takes a massive nosedive as high-vol assets
      correlate during a crash.
    - The optimized portfolio remains much flatter because the solver
      selected assets with constrained joint-tail risk.

    Returns
    -------
    str
        Path to the saved figure.
    """
    print("[Plot 3] Generating tail event stress test plot...")

    tail_data_path = os.path.join(_EXPERIMENTS_DIR, "tail_event_data.csv")

    # Run the stress analysis if the data doesn't exist yet
    if not os.path.exists(tail_data_path):
        print("    tail_event_data.csv not found, running find_stress_event.py...")
        from find_stress_event import run_stress_analysis
        run_stress_analysis()

    df = pd.read_csv(tail_data_path, parse_dates=["Date"], index_col="Date")

    dates = df.index
    opt_cum = df["optimized_cumulative_return"].values
    naive_cum = df["naive_cumulative_return"].values

    # Convert to percentage for readability
    opt_cum_pct = opt_cum * 100
    naive_cum_pct = naive_cum * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Plot optimized portfolio
    ax.plot(
        dates, opt_cum_pct, color="#1B5E20", linewidth=2.2,
        label="CVaR-Optimized Portfolio (k=5)",
    )

    # Plot naive portfolio
    ax.plot(
        dates, naive_cum_pct, color="#B71C1C", linewidth=2.2,
        linestyle="--", label="Naive Equal-Weight Portfolio (k=5)",
    )

    # Zero line
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="-", alpha=0.5)

    # Shade the region between the two curves to emphasize divergence
    ax.fill_between(
        dates, opt_cum_pct, naive_cum_pct,
        where=(opt_cum_pct > naive_cum_pct),
        alpha=0.12, color="#1B5E20", interpolate=True,
    )
    ax.fill_between(
        dates, opt_cum_pct, naive_cum_pct,
        where=(opt_cum_pct <= naive_cum_pct),
        alpha=0.12, color="#B71C1C", interpolate=True,
    )

    # Annotate final cumulative returns
    final_opt = opt_cum_pct[-1]
    final_naive = naive_cum_pct[-1]
    ax.annotate(
        f"{final_opt:+.3f}%", (dates[-1], final_opt),
        textcoords="offset points", xytext=(8, 0),
        fontsize=9.5, fontweight="bold", color="#1B5E20", va="center",
    )
    ax.annotate(
        f"{final_naive:+.3f}%", (dates[-1], final_naive),
        textcoords="offset points", xytext=(8, 0),
        fontsize=9.5, fontweight="bold", color="#B71C1C", va="center",
    )

    # Mark the point of maximum divergence
    divergence = opt_cum_pct - naive_cum_pct
    max_div_idx = np.argmax(divergence)
    if divergence[max_div_idx] > 0:
        ax.annotate(
            f"Max protection:\n{divergence[max_div_idx]:.3f}%",
            (dates[max_div_idx], (opt_cum_pct[max_div_idx] + naive_cum_pct[max_div_idx]) / 2),
            textcoords="offset points", xytext=(-60, -20),
            fontsize=8.5, ha="center",
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(
        "Tail Event Stress Test: CVaR-Optimized vs. Naive Portfolio\n"
        f"Window: {dates[0].strftime('%b %d, %Y')} - {dates[-1].strftime('%b %d, %Y')}"
    )
    ax.legend(loc="lower left")

    # Format x-axis dates
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    _ensure_viz_dir()
    out_path = os.path.join(_VIZ_DIR, "stress_test_comparison.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved -> {os.path.abspath(out_path)}")
    return out_path


# ===========================================================================
# Plot 4: Cumulative Log-Return Equity Curve
# ===========================================================================

def plot_equity_curve() -> str:
    """
    Plot the cumulative log-return equity curve for the CVaR-optimized portfolio
    against two baselines over the full backtest period.

    Requires that utils/backtest.py has been run first (or runs it automatically
    if backtest.csv does not exist).

    Returns
    -------
    str
        Path to the saved figure.
    """
    print("[Plot 4] Generating cumulative log-return equity curve...")

    backtest_path = os.path.join(_EXPERIMENTS_DIR, "backtest.csv")

    if not os.path.exists(backtest_path):
        print("    backtest.csv not found, running backtest.py...")
        from backtest import run_backtest
        run_backtest()

    df = pd.read_csv(backtest_path, parse_dates=["Date"], index_col="Date")

    dates = df.index
    opt_cum = df["optimized_cum_logret"].values * 100
    ew5_cum = df["equal_weight_k5_cum_logret"].values * 100
    unc_cum = df["unconstrained_n25_cum_logret"].values * 100

    # Create figure — clean, minimal design
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(dates, opt_cum, color="#1B5E20", linewidth=2.0,
            label="CVaR-Optimized (k=5)")
    ax.plot(dates, ew5_cum, color="#B71C1C", linewidth=1.6,
            linestyle="--", label="Equal-Weight (k=5)")
    ax.plot(dates, unc_cum, color="#4A148C", linewidth=1.6,
            linestyle=":", label="Unconstrained (N=25)")

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="-", alpha=0.4)

    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log-Return (%)")
    ax.set_title("Equity Curve: CVaR-Optimized vs. Baseline Portfolios")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    _ensure_viz_dir()
    out_path = os.path.join(_VIZ_DIR, "equity_curve.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"    Saved -> {os.path.abspath(out_path)}")
    return out_path


# ===========================================================================
# CLI Entry Point
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate plots for the short-vol portfolio optimization paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--all", action="store_true",
        help="Generate all plots.",
    )
    parser.add_argument(
        "--fat-tail", action="store_true", dest="fat_tail",
        help="Plot 1: Non-normal return distribution with fat left tail.",
    )
    parser.add_argument(
        "--complexity", action="store_true",
        help="Plot 2: Empirical complexity curve (solve time vs N).",
    )
    parser.add_argument(
        "--stress-test", action="store_true", dest="stress_test",
        help="Plot 3: Tail event stress test (optimized vs naive).",
    )
    parser.add_argument(
        "--equity-curve", action="store_true", dest="equity_curve",
        help="Plot 4: Cumulative log-return equity curve.",
    )

    args = parser.parse_args()

    if not any([args.all, args.fat_tail, args.complexity, args.stress_test,
                args.equity_curve]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.fat_tail:
        plot_fat_tail_distribution()

    if args.all or args.complexity:
        plot_complexity_curve()

    if args.all or args.stress_test:
        plot_stress_test()

    if args.all or args.equity_curve:
        plot_equity_curve()


if __name__ == "__main__":
    main()
