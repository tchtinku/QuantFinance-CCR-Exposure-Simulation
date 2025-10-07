# CCR_Exposure_Simulation.py
# Monte Carlo Simulation for Counterparty Credit Risk (CCR)
# Author: Tinku Choudhary

import argparse
import numpy as np
import matplotlib.pyplot as plt


def simulate_gbm_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    time_horizon_years: float,
    num_steps: int,
    num_paths: int,
    random_seed: int = 42,
) -> np.ndarray:
    """Simulate asset price paths using Geometric Brownian Motion.

    Returns an array of shape (num_steps, num_paths).
    """
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
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute EE, PFE, and EPE for a call option payoff.

    - EE: Expected Exposure over time
    - PFE: percentile exposure over time (default 95%)
    - EPE: Discounted average over time of EE
    """
    exposure = np.maximum(prices - strike, 0.0)

    expected_exposure = np.mean(exposure, axis=1)
    pfe = np.percentile(exposure, pfe_percentile, axis=1)

    steps = prices.shape[0]
    times = np.linspace(0.0, time_horizon_years, steps)
    discount_factors = np.exp(-risk_free_rate * times)
    epe = float(np.sum(expected_exposure * discount_factors) / steps)

    return expected_exposure, pfe, epe


def plot_exposure(expected_exposure: np.ndarray, pfe: np.ndarray, title_suffix: str = "") -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(expected_exposure, label="Expected Exposure (EE)")
    plt.plot(pfe, "--", label="Potential Future Exposure (PFE)")
    plt.title(f"Counterparty Credit Risk Exposure Simulation{title_suffix}")
    plt.xlabel("Time Steps (Days)")
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
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    parser.add_argument("--save-fig", type=str, default="", help="Path to save plot image (e.g., exposure.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    title_suffix = f" (PFE {args.pfe:.0f}th)"
    plot_exposure(expected_exposure, pfe, title_suffix=title_suffix)

    if args.save_fig:
        plt.savefig(args.save_fig, dpi=200)

    if not args.no_show:
        plt.show()
    else:
        plt.close()

    print(f"Expected Positive Exposure (EPE): {epe:.2f}")
    print("Simulation Complete ✅")


if __name__ == "__main__":
    main()
