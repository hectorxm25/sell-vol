"""
run_experiments.py

Grid-search experiment runner that sweeps over combinations of N (universe size)
and k (portfolio cardinality) to analyze how the MILP solver behaves under
different problem dimensions.

For each valid (N, k) pair where k <= N, the solver is invoked and results are
saved in an organized directory structure under experiments/.

==============================================================================
Usage
==============================================================================

    # Run the full grid with default N and k values:
    python run_experiments.py

    # Override alpha or gamma for the entire sweep:
    python run_experiments.py --alpha 0.99 --gamma 0.01

    # Use a different solver:
    python run_experiments.py --solver GLPK_MI

==============================================================================
Output Structure
==============================================================================

    experiments/
        summary.csv              — One row per (N, k) run with key metrics
        N5/
            k2/
                solution.json    — Full solver output
                details.csv      — Per-asset weights and selection
            k3/
                ...
        N10/
            k3/
                ...
            k5/
                ...
        ...
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Add project root to path for solver imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from solver import (
    ASSET_UNIVERSE,
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    DEFAULT_SOLVER,
    load_returns,
    solve_portfolio,
)

# ---------------------------------------------------------------------------
# Experiment Grid Configuration
# ---------------------------------------------------------------------------

# Universe sizes to sweep (number of assets from ASSET_UNIVERSE to consider)
N_VALUES: list[int] = [5, 10, 15, 20, 25]

# Portfolio cardinalities to sweep (exact number of assets in portfolio)
K_VALUES: list[int] = [2, 3, 5, 7, 10]

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_EXPERIMENTS_DIR = os.path.join(_SCRIPT_DIR, "experiments")
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "processed_data")
_DEFAULT_DATA_FILE = os.path.join(
    _DATA_DIR, "short-vol-returns-21window-15premium.csv"
)


def run_single_experiment(
    N: int,
    k: int,
    data_path: str,
    alpha: float,
    gamma: float,
    solver: str,
    output_dir: str,
) -> dict:
    """
    Run solver for a single (N, k) pair and save results.

    Returns a summary dict with key metrics for the summary table.
    """
    experiment_dir = os.path.join(output_dir, f"N{N}", f"k{k}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Load data
    _, R, tickers = load_returns(data_path, N)
    S = R.shape[0]

    # Solve (suppress verbose output for cleaner experiment logs)
    result = solve_portfolio(
        R=R,
        tickers=tickers,
        k=k,
        alpha=alpha,
        gamma=gamma,
        solver=solver,
        verbose=False,
    )

    # Save JSON solution
    json_output = {k_: v for k_, v in result.items() if k_ != "all_assets"}
    json_path = os.path.join(experiment_dir, "solution.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # Save CSV details
    csv_path = os.path.join(experiment_dir, "details.csv")
    if result["all_assets"]:
        pd.DataFrame(result["all_assets"]).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(
            columns=["ticker", "selected", "weight", "expected_return"]
        ).to_csv(csv_path, index=False)

    # Build summary row
    summary = {
        "N": N,
        "k": k,
        "S": S,
        "alpha": alpha,
        "gamma": gamma,
        "solver": solver,
        "status": result["status"],
        "objective_value": result["objective_value"],
        "portfolio_mu_daily": result["portfolio_mu"],
        "portfolio_mu_annualized": result["portfolio_mu"] * 252 if result["portfolio_mu"] else None,
        "portfolio_cvar": result["portfolio_cvar"],
        "var_estimate": result["var_estimate"],
        "solve_time_sec": result["solve_time_sec"],
        "selected_assets": ", ".join(result["selected_assets"]) if result["selected_assets"] else "",
        "num_variables": (N + N + 1 + S),  # w + z + VaR + u
        "num_constraints": (1 + 1 + N + N + S + S + 1),  # budget + cardinality + bigM + nonneg_w + loss + nonneg_u + cvar
    }

    return summary


def run_experiments(
    n_values: list[int],
    k_values: list[int],
    data_path: str,
    alpha: float,
    gamma: float,
    solver: str,
) -> pd.DataFrame:
    """
    Run the full grid of experiments for all valid (N, k) combinations.

    A combination is valid when k <= N.

    Returns
    -------
    pd.DataFrame
        Summary table with one row per experiment.
    """
    os.makedirs(_EXPERIMENTS_DIR, exist_ok=True)

    # Determine valid (N, k) pairs
    pairs = [(N, k) for N in sorted(n_values) for k in sorted(k_values) if k <= N]
    total = len(pairs)

    print("=" * 70)
    print("  EXPERIMENT GRID SEARCH")
    print("=" * 70)
    print(f"  N values: {sorted(n_values)}")
    print(f"  k values: {sorted(k_values)}")
    print(f"  Valid (N, k) pairs: {total}")
    print(f"  Alpha: {alpha}, Gamma: {gamma}, Solver: {solver}")
    print(f"  Output: {_EXPERIMENTS_DIR}/")
    print("=" * 70)

    summaries = []
    total_start = time.time()

    for idx, (N, k) in enumerate(pairs, 1):
        print(f"\n[{idx}/{total}] Running N={N}, k={k}...", end=" ", flush=True)

        try:
            summary = run_single_experiment(
                N=N, k=k, data_path=data_path,
                alpha=alpha, gamma=gamma, solver=solver,
                output_dir=_EXPERIMENTS_DIR,
            )
            print(
                f"Done in {summary['solve_time_sec']:.2f}s | "
                f"status={summary['status']} | "
                f"obj={summary['objective_value']:.6e}" if summary['objective_value'] else
                f"Done in {summary['solve_time_sec']:.2f}s | "
                f"status={summary['status']}"
            )
        except Exception as e:
            print(f"FAILED: {e}")
            summary = {
                "N": N, "k": k, "S": None, "alpha": alpha, "gamma": gamma,
                "solver": solver, "status": f"error: {e}",
                "objective_value": None, "portfolio_mu_daily": None,
                "portfolio_mu_annualized": None, "portfolio_cvar": None,
                "var_estimate": None, "solve_time_sec": None,
                "selected_assets": "", "num_variables": None,
                "num_constraints": None,
            }

        summaries.append(summary)

    total_elapsed = time.time() - total_start

    # Build and save summary DataFrame
    df_summary = pd.DataFrame(summaries)
    summary_path = os.path.join(_EXPERIMENTS_DIR, "summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("  GRID SEARCH COMPLETE")
    print("=" * 70)
    print(f"  Total experiments: {total}")
    print(f"  Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Summary saved to: {summary_path}")
    print("=" * 70)

    # Print condensed results table
    print("\n  Summary Table:")
    print(f"  {'N':>3} {'k':>3} {'Status':>12} {'Obj (daily)':>14} {'Time (s)':>10} {'Selected':>40}")
    print("  " + "-" * 86)
    for _, row in df_summary.iterrows():
        obj_str = f"{row['objective_value']:.6e}" if row['objective_value'] else "N/A"
        time_str = f"{row['solve_time_sec']:.2f}" if row['solve_time_sec'] else "N/A"
        print(
            f"  {row['N']:>3} {row['k']:>3} {row['status']:>12} "
            f"{obj_str:>14} {time_str:>10} "
            f"{row['selected_assets'][:40]:>40}"
        )

    return df_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grid-search experiments over (N, k) pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"CVaR confidence level for all runs (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Maximum CVaR risk budget for all runs (default: {DEFAULT_GAMMA}).",
    )
    parser.add_argument(
        "--data", type=str, default=_DEFAULT_DATA_FILE,
        help="Path to the short-vol returns CSV.",
    )
    parser.add_argument(
        "--solver", type=str, default=DEFAULT_SOLVER,
        help=f"CVXPY MILP solver (default: {DEFAULT_SOLVER}).",
    )

    args = parser.parse_args()

    run_experiments(
        n_values=N_VALUES,
        k_values=K_VALUES,
        data_path=args.data,
        alpha=args.alpha,
        gamma=args.gamma,
        solver=args.solver,
    )


if __name__ == "__main__":
    main()
