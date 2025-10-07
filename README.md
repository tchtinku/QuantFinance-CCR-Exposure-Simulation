# Monte Carlo Simulation for Counterparty Credit Risk (CCR)

This Python project simulates Expected Exposure (EE), Potential Future Exposure (PFE), and Expected Positive Exposure (EPE) for a simple derivative (European call option) using Geometric Brownian Motion (GBM) and Monte Carlo methods.

## Why this project
- Demonstrates understanding of stochastic processes and Brownian motion
- Implements Monte Carlo simulation (core IMM concept)
- Calculates and visualizes CCR exposure metrics used by risk teams (EE, PFE, EPE)

## Project structure
- `CCR_Exposure_Simulation.py` — main script to simulate price paths, compute exposures, and plot results
- `requirements.txt` — minimal dependencies

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Alternatively, install directly:
```bash
pip install numpy matplotlib
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

## How it works (brief)
1. Simulate GBM price paths for an underlying asset:
   - \( dS_t = \mu S_t dt + \sigma S_t dW_t \)
2. Compute option exposure per path and time: \( \max(S_t - K, 0) \)
3. Aggregate across paths:
   - EE(t): mean exposure over paths
   - PFE(t): percentile (default 95th) exposure over paths
   - EPE: discounted average of EE(t) over time

## Resume/email snippet
- Resume: "Monte Carlo Simulation for CCR Exposure — Built a Python-based model to simulate stochastic price paths and compute Expected Exposure (EE), Potential Future Exposure (PFE), and Expected Positive Exposure (EPE)."
- Email: "I’ve included a small Python project demonstrating Monte Carlo-based CCR exposure modeling (EE, PFE, EPE)."

## Notes
- Defaults: 1,000 paths, 252 steps (1y), \(\mu=5\%\), \(\sigma=20\%\), \(r=3\%\), \(K=100\)
- Adjust parameters via CLI flags for quick sensitivity analysis.
