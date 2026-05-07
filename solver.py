"""
solver.py

Mixed Integer Linear Program (MILP) solver for the k-asset short-volatility
portfolio optimization problem with a Conditional Value-at-Risk (CVaR) constraint.

Uses the Rockafellar & Uryasev (2000) linearization of CVaR and solves via
Branch-and-Bound through CVXPY's mixed-integer programming interface.

==============================================================================
Problem Formulation
==============================================================================

    max  sum_{i=1}^{N} mu_i * w_i          (maximize expected portfolio return)

    subject to:
        sum_{i=1}^{N} w_i = 1              (fully invested / budget constraint)
        sum_{i=1}^{N} z_i = k              (exactly k assets selected)
        w_i <= z_i          for all i       (Big-M linking: weight=0 if not selected)
        w_i >= 0            for all i       (long-only)
        u_s >= -sum_i r_{i,s} w_i - VaR    (loss exceeding VaR captured in u_s)
        u_s >= 0            for all s       (shortfall is non-negative)
        VaR + (1 / (S*(1-alpha))) * sum_s u_s <= gamma   (CVaR risk limit)
        z_i in {0, 1}      for all i       (binary asset selection)

==============================================================================
Parameters
==============================================================================

    N       : int   — Number of assets from ASSET_UNIVERSE to consider.
                      (e.g., N=5 uses the first 5 tickers)
    k       : int   — Exact number of assets in the portfolio (k <= N).
    alpha   : float — CVaR confidence level (default: 0.95).
                      Standard in risk management; 95% captures the average
                      of the worst 5% of daily losses.
    gamma   : float — Maximum allowable CVaR (risk budget).
                      Default: 0.005 (i.e., the average worst-5% daily loss
                      cannot exceed 0.5% of portfolio value). This is a
                      conservative daily risk limit appropriate for short-vol
                      strategies which have fat-tailed loss distributions.

==============================================================================
Usage
==============================================================================

    # Basic usage with defaults (N=25, k=5, alpha=0.95, gamma=0.005):
    python solver.py --N 25 --k 5

    # Vary universe size and portfolio size for runtime analysis:
    python solver.py --N 10 --k 3
    python solver.py --N 25 --k 10 --alpha 0.99 --gamma 0.01

    # Use a specific processed data file:
    python solver.py --N 15 --k 5 --data data/processed_data/short-vol-returns-21window-15premium.csv

    # Specify solver (default: GLPK_MI):
    python solver.py --N 25 --k 5 --solver GLPK_MI

==============================================================================
Outputs (saved to results/ directory)
==============================================================================

    results/solution_N{N}_k{k}_alpha{alpha}_gamma{gamma}.json
        Contains:
        - status          : Solver status (optimal, infeasible, etc.)
        - objective_value : Maximized expected portfolio return
        - solve_time_sec  : Wall-clock time for solving
        - parameters      : Dict of all input parameters (N, k, alpha, gamma)
        - selected_assets : List of ticker symbols chosen (z_i = 1)
        - weights         : Dict mapping ticker -> optimal weight
        - portfolio_mu    : Expected daily return of the optimal portfolio
        - portfolio_cvar  : Realized CVaR of the optimal portfolio
        - var_estimate    : Optimal VaR auxiliary variable value

    results/solution_N{N}_k{k}_alpha{alpha}_gamma{gamma}_details.csv
        Full per-asset detail: ticker, z_i, w_i, mu_i for all N assets.

==============================================================================
Assumptions & Design Decisions
==============================================================================

    1. The short-vol return data is treated as an empirical scenario set where
       each historical day is an equally-likely scenario (uniform probability
       1/S for each of the S days).

    2. alpha = 0.95 is the industry standard for CVaR. It balances sensitivity
       to tail events against having enough tail observations (~62 days in the
       worst 5% of 1232 scenarios) for a statistically meaningful estimate.

    3. gamma = 0.005 was chosen as a conservative daily CVaR limit. Short-vol
       strategies are exposed to sudden variance spikes (e.g., VIX blowups),
       so a tight risk budget prevents catastrophic drawdowns. This means the
       average loss on the worst 5% of days must not exceed 50 bps.

    4. We use GLPK_MI (GNU Linear Programming Kit, Mixed Integer) as the
       default solver because it is free, widely available, and implements
       Branch-and-Bound natively. For large N, commercial solvers like Gurobi
       or MOSEK would be significantly faster.

    5. The Big-M constant is 1 (since max weight = 1 due to the budget
       constraint). This is tight and does not cause numerical issues.

    6. VaR is an unrestricted continuous variable (can be negative, meaning
       the portfolio could profit even in the "worst" quantile).
"""

import argparse
import json
import os
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, "data", "processed_data")
_RESULTS_DIR = os.path.join(_SCRIPT_DIR, "results")

# Default short-vol returns file (matches process_data.py output naming)
_DEFAULT_DATA_FILE = os.path.join(
    _DATA_DIR, "short-vol-returns-21window-15premium.csv"
)

# Import ASSET_UNIVERSE for ticker name resolution.
# We extract it without executing the full gather_data module to avoid
# importing yfinance (heavy dependency not needed for solving).
def _load_asset_universe() -> list[str]:
    """Parse ASSET_UNIVERSE from gather_data.py without importing yfinance."""
    import ast
    gather_path = os.path.join(_SCRIPT_DIR, "utils", "gather_data.py")
    with open(gather_path, "r") as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and hasattr(node.target, "id"):
            if node.target.id == "ASSET_UNIVERSE":
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if hasattr(target, "id") and target.id == "ASSET_UNIVERSE":
                    return ast.literal_eval(node.value)
    raise RuntimeError("Could not find ASSET_UNIVERSE in gather_data.py")


ASSET_UNIVERSE = _load_asset_universe()

# ---------------------------------------------------------------------------
# Default Parameters
# ---------------------------------------------------------------------------
DEFAULT_ALPHA = 0.95
DEFAULT_GAMMA = 0.005
DEFAULT_SOLVER = "GLPK_MI"


def load_returns(data_path: str, N: int) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load the short-vol returns matrix and slice to the first N assets.

    Parameters
    ----------
    data_path : str
        Path to the short-vol returns CSV.
    N : int
        Number of assets to include (first N from ASSET_UNIVERSE).

    Returns
    -------
    df : pd.DataFrame
        The sliced returns DataFrame (S x N).
    R : np.ndarray
        Returns matrix of shape (S, N).
    tickers : list[str]
        The N ticker symbols used.
    """
    df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")

    tickers = ASSET_UNIVERSE[:N]
    missing = [t for t in tickers if t not in df.columns]
    if missing:
        raise ValueError(f"Tickers not found in data: {missing}")

    df = df[tickers].dropna()
    R = df.values  # shape (S, N)

    return df, R, tickers


def solve_portfolio(
    R: np.ndarray,
    tickers: list[str],
    k: int,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    solver: str = DEFAULT_SOLVER,
    verbose: bool = True,
) -> dict:
    """
    Solve the k-asset CVaR-constrained short-vol portfolio MILP.

    Parameters
    ----------
    R : np.ndarray
        Returns matrix of shape (S, N). Each row is a scenario (day),
        each column is an asset's short-vol return on that day.
    tickers : list[str]
        Asset ticker symbols corresponding to columns of R.
    k : int
        Exact number of assets to include in the portfolio.
    alpha : float
        CVaR confidence level (e.g., 0.95).
    gamma : float
        Maximum allowable CVaR (daily risk budget).
    solver : str
        CVXPY solver name (e.g., "GLPK_MI", "ECOS_BB", "SCIP", "GUROBI").
    verbose : bool
        If True, print progress information.

    Returns
    -------
    dict
        Solution dictionary containing status, weights, selected assets,
        objective value, and timing information.
    """
    S, N = R.shape

    if k > N:
        raise ValueError(f"k={k} cannot exceed N={N}.")
    if k < 1:
        raise ValueError(f"k must be >= 1, got k={k}.")

    # Expected daily return for each asset (mean across all scenarios)
    mu = R.mean(axis=0)  # shape (N,)

    if verbose:
        print(f"Setting up MILP: N={N}, k={k}, S={S}, alpha={alpha}, gamma={gamma}")
        print(f"Solver: {solver}")
        print(f"Asset expected returns (annualized approx, *252):")
        for i, t in enumerate(tickers):
            print(f"    {t:6s}: {mu[i]*252:.6f}")

    # ------------------------------------------------------------------
    # Decision Variables
    # ------------------------------------------------------------------
    w = cp.Variable(N, name="weights")          # continuous portfolio weights
    z = cp.Variable(N, boolean=True, name="selection")  # binary selection
    VaR = cp.Variable(name="VaR")               # Value-at-Risk auxiliary
    u = cp.Variable(S, name="shortfall")        # per-scenario excess loss

    # ------------------------------------------------------------------
    # Objective: maximize expected portfolio return
    # ------------------------------------------------------------------
    objective = cp.Maximize(mu @ w)

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------
    constraints = []

    # C1: Budget constraint (fully invested)
    constraints.append(cp.sum(w) == 1)

    # C2: Cardinality constraint (exactly k assets)
    constraints.append(cp.sum(z) == k)

    # C3: Big-M linking (w_i <= z_i, with M=1)
    constraints.append(w <= z)

    # C4: Non-negativity of weights
    constraints.append(w >= 0)

    # C5: Loss exceeding VaR captured by u_s
    # Portfolio return on day s: R[s, :] @ w
    # Loss = -(R[s, :] @ w)
    # u_s >= loss - VaR = -(R[s,:] @ w) - VaR
    constraints.append(u >= -R @ w - VaR)

    # C6: Non-negativity of shortfall
    constraints.append(u >= 0)

    # C7: CVaR risk limit
    # CVaR = VaR + (1 / (S*(1-alpha))) * sum(u_s) <= gamma
    cvar_expr = VaR + (1.0 / (S * (1.0 - alpha))) * cp.sum(u)
    constraints.append(cvar_expr <= gamma)

    # C8: Binary restriction is handled by boolean=True in variable definition

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    problem = cp.Problem(objective, constraints)

    if verbose:
        print(f"\nProblem size: {problem.size_metrics}")
        print(f"Solving...")

    start_time = time.time()
    problem.solve(solver=solver, verbose=verbose)
    solve_time = time.time() - start_time

    if verbose:
        print(f"\nSolve time: {solve_time:.3f} seconds")
        print(f"Status: {problem.status}")

    # ------------------------------------------------------------------
    # Extract Results
    # ------------------------------------------------------------------
    result = {
        "status": problem.status,
        "objective_value": None,
        "solve_time_sec": solve_time,
        "parameters": {
            "N": N,
            "k": k,
            "S": S,
            "alpha": alpha,
            "gamma": gamma,
            "solver": solver,
        },
        "selected_assets": [],
        "weights": {},
        "portfolio_mu": None,
        "portfolio_cvar": None,
        "var_estimate": None,
        "all_assets": [],
    }

    if problem.status in ("optimal", "optimal_inaccurate"):
        w_val = w.value
        z_val = z.value
        VaR_val = VaR.value
        u_val = u.value

        # Round binary variables to clean 0/1 (solver may return 0.9999)
        z_rounded = np.round(z_val).astype(int)

        # Build results
        selected_mask = z_rounded == 1
        selected_tickers = [tickers[i] for i in range(N) if selected_mask[i]]
        weights_dict = {
            tickers[i]: float(w_val[i]) for i in range(N) if w_val[i] > 1e-8
        }

        portfolio_return = float(mu @ w_val)
        realized_cvar = float(VaR_val + (1.0 / (S * (1 - alpha))) * np.sum(u_val))

        result["objective_value"] = float(problem.value)
        result["selected_assets"] = selected_tickers
        result["weights"] = weights_dict
        result["portfolio_mu"] = portfolio_return
        result["portfolio_cvar"] = realized_cvar
        result["var_estimate"] = float(VaR_val)

        # Per-asset detail for the CSV output
        all_assets = []
        for i in range(N):
            all_assets.append({
                "ticker": tickers[i],
                "selected": int(z_rounded[i]),
                "weight": float(w_val[i]),
                "expected_return": float(mu[i]),
            })
        result["all_assets"] = all_assets

        if verbose:
            print(f"\n{'='*60}")
            print(f"  OPTIMAL SOLUTION FOUND")
            print(f"{'='*60}")
            print(f"  Objective (expected daily return): {portfolio_return:.8f}")
            print(f"  Annualized (approx, *252):         {portfolio_return*252:.6f}")
            print(f"  Portfolio CVaR:                    {realized_cvar:.8f}")
            print(f"  VaR estimate:                     {VaR_val:.8f}")
            print(f"  Selected assets ({k}):")
            for t in selected_tickers:
                print(f"      {t:6s}  w={weights_dict.get(t, 0.0):.6f}")
            print(f"{'='*60}")
    else:
        if verbose:
            print(f"\nSolver did not find an optimal solution.")
            print(f"Status: {problem.status}")
            if problem.status == "infeasible":
                print("The problem is infeasible. Consider relaxing gamma "
                      "(increase risk budget) or reducing k.")

    return result


def save_results(result: dict, output_dir: str = _RESULTS_DIR) -> tuple[str, str]:
    """
    Save solver results to JSON and CSV files.

    Parameters
    ----------
    result : dict
        The solution dictionary from solve_portfolio().
    output_dir : str
        Directory to save results into.

    Returns
    -------
    tuple[str, str]
        Paths to the saved JSON and CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    params = result["parameters"]
    base_name = (
        f"solution_N{params['N']}_k{params['k']}"
        f"_alpha{params['alpha']}_gamma{params['gamma']}"
    )

    # Save JSON (full solution details)
    json_path = os.path.join(output_dir, f"{base_name}.json")
    json_output = {k: v for k, v in result.items() if k != "all_assets"}
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=2)

    # Save CSV (per-asset detail)
    csv_path = os.path.join(output_dir, f"{base_name}_details.csv")
    if result["all_assets"]:
        df_detail = pd.DataFrame(result["all_assets"])
        df_detail.to_csv(csv_path, index=False)
    else:
        pd.DataFrame(
            columns=["ticker", "selected", "weight", "expected_return"]
        ).to_csv(csv_path, index=False)

    print(f"\nResults saved:")
    print(f"    JSON: {os.path.abspath(json_path)}")
    print(f"    CSV:  {os.path.abspath(csv_path)}")

    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve the k-asset CVaR-constrained short-vol portfolio MILP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--N", type=int, default=len(ASSET_UNIVERSE),
        help=f"Number of assets from ASSET_UNIVERSE to consider (default: {len(ASSET_UNIVERSE)}).",
    )
    parser.add_argument(
        "--k", type=int, required=True,
        help="Exact number of assets in the portfolio.",
    )
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"CVaR confidence level (default: {DEFAULT_ALPHA}).",
    )
    parser.add_argument(
        "--gamma", type=float, default=DEFAULT_GAMMA,
        help=f"Maximum allowable CVaR / risk budget (default: {DEFAULT_GAMMA}).",
    )
    parser.add_argument(
        "--data", type=str, default=_DEFAULT_DATA_FILE,
        help="Path to the short-vol returns CSV file.",
    )
    parser.add_argument(
        "--solver", type=str, default=DEFAULT_SOLVER,
        help=f"CVXPY MILP solver (default: {DEFAULT_SOLVER}). Options: GLPK_MI, ECOS_BB, SCIP, GUROBI.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output.",
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading short-vol returns from: {args.data}")
    df, R, tickers = load_returns(args.data, args.N)

    # Solve
    result = solve_portfolio(
        R=R,
        tickers=tickers,
        k=args.k,
        alpha=args.alpha,
        gamma=args.gamma,
        solver=args.solver,
        verbose=not args.quiet,
    )

    # Save
    save_results(result)


if __name__ == "__main__":
    main()
