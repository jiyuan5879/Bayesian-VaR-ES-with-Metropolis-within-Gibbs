# Bayesian-VaR-ES-with-Metropolis-within-Gibbs


This repository implements a full workflow for the STATS 211 final project:

- clean and export S&P 500 return data,
- fit a Bayesian Student-t return model,
- estimate posterior parameters using **Metropolis-within-Gibbs (MWG)**,
- evaluate convergence diagnostics,
- estimate **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**.

---

## Project Overview

Financial daily returns are modeled as Student-t with unknown parameters:

- location: `mu`
- scale: `sigma^2`
- degrees of freedom: `nu` (tail heaviness)

A hierarchical representation is used:

- `y_i | lambda_i ~ Normal(mu, sigma^2 / lambda_i)`
- `lambda_i ~ Gamma(nu/2, nu/2)`

Posterior sampling is done via Gibbs updates for `(lambda, mu, sigma^2)` and an MH step for `nu`.

---

## Repository Structure

- `scripts/01_export_clean_returns.py`  
  Export clean data from `.numbers` to CSV.
- `scripts/02_mwg_sampler.py`  
  Run MWG sampler (supports multi-chain MCMC).
- `scripts/03_convergence_diagnostics.py`  
  Generate diagnostics tables + diagnostic plots (trace/hist+smooth curve/ACF).
- `scripts/04_report_tables.py`  
  Build report-ready summary tables.
- `data/clean_returns.csv`  
  Cleaned daily return dataset (generated).
- `outputs_multichain/`  
  Final multi-chain posterior/risk outputs.
- `outputs/diagnostics/`  
  Diagnostic summaries and figures.
- `report/Final_Report_Text.md`  
  Full academic text report.

---

## Environment Setup

Recommended: use a local Python virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas numbers-parser
