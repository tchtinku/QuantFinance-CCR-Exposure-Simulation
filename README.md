# Monte Carlo Simulation for Counterparty Credit Risk (CCR)

[![CI](https://github.com/tchtinku/QuantFinance-CCR-Exposure-Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/tchtinku/QuantFinance-CCR-Exposure-Simulation/actions/workflows/ci.yml)

This Python project simulates Expected Exposure (EE), Potential Future Exposure (PFE), and Expected Positive Exposure (EPE) for a simple derivative (European call option) using Geometric Brownian Motion (GBM) and Monte Carlo methods.

## Why this project
- Demonstrates understanding of stochastic processes and Brownian motion
- Implements Monte Carlo simulation (core IMM concept)
- Calculates and visualizes CCR exposure metrics used by risk teams (EE, PFE, EPE)

## Background (brief)
- Underlying dynamics (GBM): \( dS_t = \mu S_t dt + \sigma S_t dW_t \). Discretized as \( S_t = S_{t-1} \cdot \exp((\mu - 0.5\sigma^2)dt + \sigma\sqrt{dt}Z_t) \).
- Exposure for a long European call: \( E_t = \max(S_t - K, 0) \) (cannot be negative).
- Expected Exposure (EE): average exposure across paths at each time.
- Potential Future Exposure (PFE): a percentile (e.g., 95th) of exposure distribution across paths at each time.
- Expected Positive Exposure (EPE): discounted time-average of EE: \( \text{EPE} = \frac{1}{N}\sum_t EE(t) e^{-rt_t} \).

## Project structure
- `CCR_Exposure_Simulation.py` — main script to simulate price paths, compute exposures, and plot results
- `requirements.txt` — minimal dependencies
- `.github/workflows/ci.yml` — CI runs the simulation headlessly and uploads the plot as an artifact
- `.github/workflows/release.yml` — creates a Release on tags and attaches the plot

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Alternatively, install directly:
```bash
pip install numpy matplotlib pillow
```

## Run (defaults)
```bash
python CCR_Exposure_Simulation.py
```

You should see a plot of Expected Exposure (EE) and Potential Future Exposure (PFE), and a printed EPE value like:
```
Expected Positive Exposure (EPE): 8.47
Simulation Complete ✅
```

## CLI usage
You can override parameters and control plotting via flags:
```bash
python CCR_Exposure_Simulation.py \
  --S0 100 --mu 0.05 --sigma 0.2 --T 1.0 --steps 252 --paths 1000 \
  --K 100 --r 0.03 --seed 42 --pfe 95
```

- `--no-show`: run headless (no GUI window)
- `--save-fig exposure.png`: save the plot to a file (PNG recommended)

Example for GitHub-friendly artifact without opening a window:
```bash
python CCR_Exposure_Simulation.py --no-show --save-fig exposure.png
```

## Sensitivity analysis
Run with `--sensitivity` to sweep a parameter and plot summary EPE and terminal PFE:
```bash
# Sweep volatility
python CCR_Exposure_Simulation.py --sensitivity --sweep-param sigma --sweep-values 0.1,0.2,0.3,0.4,0.5 \
  --no-show --save-fig-sensitivity sensitivity_sigma.png

# Sweep drift
python CCR_Exposure_Simulation.py --sensitivity --sweep-param mu --sweep-values 0.00,0.02,0.04,0.06 \
  --no-show --save-fig-sensitivity sensitivity_mu.png

# Sweep horizon
python CCR_Exposure_Simulation.py --sensitivity --sweep-param T --sweep-values 0.5,1,2 \
  --no-show --save-fig-sensitivity sensitivity_T.png
```

## CI and artifacts
- On every push/PR, CI runs the simulation headlessly and uploads `exposure.png` as a workflow artifact.
- Download it from the Actions run → Artifacts.
- To create a release with the plot attached, tag a version like `v1.0.0` and push the tag; the Release workflow will publish `exposure.png`.

## Validation and sanity checks
- Input validation for parameters (e.g., positive `S0`, non-negative `sigma`, `0 < pfe < 100`).
- Output checks ensure exposure is non-negative and PFE does not exceed max exposure at any time step.

## Limitations / future work
- Single-factor GBM; no rate curves, credit spreads, or stochastic volatility.
- Single derivative (European call); no netting sets or collateral/CSA.
- No wrong-way risk modeling or counterparty default dynamics.
- Future extensions: multi-factor models, correlated risk factors, CSA/netting, MVA/CVA/DVA.

## Resume/email snippet
- Resume: "Monte Carlo Simulation for CCR Exposure — Built a Python-based model to simulate stochastic price paths and compute Expected Exposure (EE), Potential Future Exposure (PFE), and Expected Positive Exposure (EPE)."
- Email: "I’ve included a small Python project demonstrating Monte Carlo-based CCR exposure modeling (EE, PFE, EPE)."

## Discoverability (GitHub topics)
Add repository topics such as `quantitative-finance`, `counterparty-risk`, `monte-carlo`, `risk-management` in the GitHub UI (Settings → Topics) to improve discoverability.

## Notes
- Defaults: 1,000 paths, 252 steps (1y), \(\mu=5\%\), \(\sigma=20\%\), \(r=3\%\), \(K=100\)
- Adjust parameters via CLI flags for quick sensitivity analysis.
