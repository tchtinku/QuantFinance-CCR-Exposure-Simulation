# CCR_Exposure_Simulation.py
# Monte Carlo Simulation for Counterparty Credit Risk (CCR)
# Author: Tinku Choudhary

import argparse
from typing import Iterable, List, Tuple
import os
import numpy as np


def simulate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon_years: float,
    num_steps: int,
    num_paths: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Simulate asset price paths using Geometric Brownian Motion (GBM).

    SDE (under real measure): dS_t = mu * S_t dt + sigma * S_t dW_t
    Discretization (Euler-Maruyama in log space):
      S_t = S_{t-1} * exp((mu - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z_t)

    Returns
    -------
    np.ndarray
        Array of shape (num_steps, num_paths) of simulated prices.
    """
    if initial_price <= 0:
        raise ValueError("initial_price must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if time_horizon_years <= 0 or num_steps < 2:
        raise ValueError("time_horizon_years must be > 0 and num_steps >= 2")
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1")

    dt = time_horizon_years / num_steps
    rng = np.random.default_rng(random_seed)
    standard_normals = rng.standard_normal(size=(num_steps, num_paths))

    prices = np.zeros_like(standard_normals, dtype=float)
    prices[0] = initial_price

    drift_term = (drift - 0.5 * volatility ** 2) * dt
    diffusion_scale = volatility * np.sqrt(dt)

    for t in range(1, num_steps):
        prices[t] = prices[t - 1] * np.exp(drift_term + diffusion_scale * standard_normals[t])

    return prices


def compute_exposure_metrics(
    prices: np.ndarray,
    strike: float,
    risk_free_rate: float,
    time_horizon_years: float,
    pfe_percentile: float = 95.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute EE, PFE, and EPE for a European call option exposure.

    Exposure at time t per path: max(S_t - K, 0)
    EE(t): mean exposure across paths.
    PFE(t): percentile exposure across paths.
    EPE: discounted time-average of EE(t), using exp(-r * t).
    """
    if strike <= 0:
        raise ValueError("strike must be positive")
    if not (0.0 < pfe_percentile < 100.0):
        raise ValueError("pfe_percentile must be in (0, 100)")

    exposure = np.maximum(prices - strike, 0.0)

    # Sanity checks: exposure non-negative
    if np.any(exposure < -1e-12):
        raise AssertionError("Exposure contains negative values, which is invalid for a call option")

    expected_exposure = np.mean(exposure, axis=1)
    pfe = np.percentile(exposure, pfe_percentile, axis=1)

    # Sanity: PFE should not exceed the per-time-step max exposure by more than tolerance
    max_exposure = np.max(exposure, axis=1)
    if np.any(pfe - max_exposure > 1e-9):
        raise AssertionError("Computed PFE exceeds max exposure at some time steps")

    steps = prices.shape[0]
    times = np.linspace(0.0, time_horizon_years, steps)
    discount_factors = np.exp(-risk_free_rate * times)
    epe = float(np.sum(expected_exposure * discount_factors) / steps)

    return expected_exposure, pfe, epe


def compute_running_discounted_epe(
    expected_exposure: np.ndarray,
    risk_free_rate: float,
    time_horizon_years: float,
) -> np.ndarray:
    """Compute running discounted EPE curve over time for visualization."""
    steps = expected_exposure.shape[0]
    times = np.linspace(0.0, time_horizon_years, steps)
    discount_factors = np.exp(-risk_free_rate * times)
    cumulative = np.cumsum(expected_exposure * discount_factors)
    # avoid divide by zero for first element; start from 1
    denom = np.arange(1, steps + 1)
    return cumulative / denom


def plot_exposure(expected_exposure: np.ndarray, pfe: np.ndarray, title_suffix: str = "", running_epe: np.ndarray | None = None) -> None:
    """Plot the EE and PFE profiles over time, with optional running discounted EPE."""
    import matplotlib.pyplot as plt  # local import after backend is set
    plt.figure(figsize=(10, 5))
    plt.plot(expected_exposure, label="Expected Exposure (EE)")
    plt.plot(pfe, "--", label="Potential Future Exposure (PFE)")
    if running_epe is not None:
        plt.plot(running_epe, ":", label="Running discounted EPE")
    plt.title(f"Counterparty Credit Risk Exposure Simulation{title_suffix}")
    plt.xlabel("Time Steps (Days)")
    plt.ylabel("Exposure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def run_sensitivity(
    sweep_param: str,
    sweep_values: Iterable[float],
    base_args: argparse.Namespace,
) -> None:
    """Run sensitivity analysis over a sweep of a selected parameter.

    For each value, compute EPE and terminal-time PFE and plot summary vs the swept value.
    """
    import matplotlib.pyplot as plt  # local import after backend is set
    x_values: List[float] = []
    epe_values: List[float] = []
    pfe_terminal_values: List[float] = []

    for value in sweep_values:
        args = base_args
        if sweep_param == "sigma":
            args_sigma = float(value)
            if args_sigma < 0:
                continue
            sigma = args_sigma
            mu = args.mu
            T = args.T
        elif sweep_param == "mu":
            sigma = args.sigma
            mu = float(value)
            T = args.T
        elif sweep_param == "T":
            sigma = args.sigma
            mu = args.mu
            T = float(value)
            if T <= 0:
                continue
        else:
            raise ValueError("sweep_param must be one of: sigma, mu, T")

        prices = simulate_gbm_paths(
            initial_price=args.S0,
            drift=mu,
            volatility=sigma,
            time_horizon_years=T,
            num_steps=args.steps,
            num_paths=args.paths,
            random_seed=args.seed,
        )
        _, pfe, epe = compute_exposure_metrics(
            prices=prices,
            strike=args.K,
            risk_free_rate=args.r,
            time_horizon_years=T,
            pfe_percentile=args.pfe,
        )
        x_values.append(float(value))
        epe_values.append(epe)
        pfe_terminal_values.append(float(pfe[-1]))

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, epe_values, marker="o", label="EPE")
    plt.plot(x_values, pfe_terminal_values, marker="s", label="PFE at maturity")
    plt.title(f"Sensitivity of EPE and PFE vs {sweep_param}")
    plt.xlabel(sweep_param)
    plt.ylabel("Exposure")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo CCR Exposure Simulation")
    parser.add_argument("--S0", type=float, default=100.0, help="Initial asset price")
    parser.add_argument("--mu", type=float, default=0.05, help="Drift (annual)")
    parser.add_argument("--sigma", type=float, default=0.2, help="Volatility (annual)")
    parser.add_argument("--T", type=float, default=1.0, help="Time horizon in years")
    parser.add_argument("--steps", type=int, default=252, help="Number of time steps")
    parser.add_argument("--paths", type=int, default=1000, help="Number of Monte Carlo paths")
    parser.add_argument("--K", type=float, default=100.0, help="Option strike price")
    parser.add_argument("--r", type=float, default=0.03, help="Risk-free rate (annual)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pfe", type=float, default=95.0, help="PFE percentile (e.g., 95)")
    parser.add_argument("--confidence", type=float, default=None, help="Confidence level (e.g., 0.99) to map to PFE percentile")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--save-fig", type=str, default="", help="Path to save plot image (e.g., exposure.png)")
    parser.add_argument("--save_plot", action="store_true", help="Alias: save plot to exposure_plot.png")
    parser.add_argument("--plot-epe", action="store_true", help="Also plot running discounted EPE curve")

    # Sensitivity analysis options
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis instead of a single run")
    parser.add_argument("--sweep-param", type=str, default="sigma", help="Parameter to sweep: sigma, mu, or T")
    parser.add_argument(
        "--sweep-values",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated values for sweep (e.g., '0.1,0.2,0.3')",
    )
    parser.add_argument(
        "--save-fig-sensitivity",
        type=str,
        default="",
        help="Path to save sensitivity plot (e.g., sensitivity.png)",
    )

    return parser.parse_args()


def select_backend(no_show: bool, saving: bool) -> None:
    """Select a non-GUI backend when running headless or saving figures.

    This avoids GUI requirements and common PIL backend issues on some systems.
    """
    if no_show or saving or os.environ.get("CI"):
        os.environ["MPLBACKEND"] = "Agg"


def main() -> None:
    args = parse_args()

    # Map confidence (0-1) to percentile if provided
    if args.confidence is not None:
        if not (0.0 < args.confidence < 1.0):
            raise ValueError("--confidence must be in (0,1)")
        args.pfe = 100.0 * float(args.confidence)

    # Map --save_plot to a default save path if not provided
    if args.save_plot and not args.save_fig:
        args.save_fig = "exposure_plot.png"
    if args.save_plot:
        args.no_show = True

    # Choose backend before importing pyplot
    select_backend(no_show=args.no_show, saving=bool(args.save_fig or args.save_fig_sensitivity))
    import matplotlib.pyplot as plt  # after backend selection

    if args.sensitivity:
        values = [float(x.strip()) for x in args.sweep_values.split(",") if x.strip()]
        run_sensitivity(args.sweep_param, values, args)
        if args.save_fig_sensitivity:
            plt.savefig(args.save_fig_sensitivity, dpi=200)
        if not args.no_show:
            plt.show()
        else:
            plt.close()
        print("Sensitivity analysis complete ✅")
        return

    # Single-run mode
    prices = simulate_gbm_paths(
        initial_price=args.S0,
        drift=args.mu,
        volatility=args.sigma,
        time_horizon_years=args.T,
        num_steps=args.steps,
        num_paths=args.paths,
        random_seed=args.seed,
    )

    expected_exposure, pfe, epe = compute_exposure_metrics(
        prices=prices,
        strike=args.K,
        risk_free_rate=args.r,
        time_horizon_years=args.T,
        pfe_percentile=args.pfe,
    )

    running_epe = compute_running_discounted_epe(expected_exposure, risk_free_rate=args.r, time_horizon_years=args.T) if args.plot_epe else None

    title_suffix = f" (PFE {args.pfe:.0f}th)"
    plot_exposure(expected_exposure, pfe, title_suffix=title_suffix, running_epe=running_epe)

    if args.save_fig:
        plt.savefig(args.save_fig, dpi=200)

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    print(f"Expected Positive Exposure (EPE): {epe:.2f}")
    if len(pfe) > 0:
        print(f"PFE at maturity ({args.pfe:.0f}th): {pfe[-1]:.2f}")
    print("Simulation Complete ✅")


if __name__ == "__main__":
    main()
