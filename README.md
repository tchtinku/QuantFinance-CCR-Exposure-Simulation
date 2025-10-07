# QuantFinance – CCR Exposure Simulation

A Python-based quantitative finance project that simulates **Expected Exposure (EE)**, **Potential Future Exposure (PFE)**, and **Expected Positive Exposure (EPE)** using a **Monte Carlo approach** under a simplified **Geometric Brownian Motion (GBM)** model.

---

## 📘 Background

**Counterparty Credit Risk (CCR)** measures the risk that a counterparty to a financial transaction could default before the final settlement of cash flows.  
This simulation estimates:

- **EE (Expected Exposure)** – Average exposure over time  
- **PFE (Potential Future Exposure)** – Exposure at a given confidence level (e.g., 95th percentile)  
- **EPE (Expected Positive Exposure)** – Time-averaged EE  

These metrics are foundational to **Internal Model Method (IMM)** frameworks used by banks to quantify CCR.

---

## ⚙️ Features

- Monte Carlo simulation for asset price evolution  
- Exposure profile computation (EE, PFE, EPE)  
- Adjustable model parameters (μ, σ, time horizon, paths)  
- Visualization of simulated exposures  
- Lightweight and runs locally (no paid APIs)

---

## 🚀 Usage

```bash
# Clone the repo
git clone https://github.com/tchtinku/QuantFinance-CCR-Exposure-Simulation.git
cd QuantFinance-CCR-Exposure-Simulation

# Run simulation with default parameters
python CCR_Exposure_Simulation.py

# Or customize parameters
python CCR_Exposure_Simulation.py --mu 0.02 --sigma 0.15 --paths 5000 --confidence 0.99
```

📊 Example Output

| Metric | Example Value |
|-------:|---------------:|
| EE (mean) | 2.47 |
| PFE (95%) | 4.12 |
| EPE | 2.36 |

Sample Exposure Profile (Generated via matplotlib):

![Exposure Profile](exposure_plot.png)

---

## 🧠 Future Enhancements

- Correlated risk factors  
- Netting set aggregation  
- Credit Valuation Adjustment (CVA) estimation  
- Integration with financial datasets

---

## 👨‍💻 Author

Tinku Choudhary  
Data Engineer | Aspiring Quant Developer  
LinkedIn | Email

---
