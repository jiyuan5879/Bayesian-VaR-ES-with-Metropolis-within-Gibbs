# STATS 211 Final Project Report (Text Version)

## Title
Estimation of Value-at-Risk and Expected Shortfall for S&P 500 Log Returns via Bayesian Student-t Modeling and Metropolis-within-Gibbs Sampling

## Abstract
This report implements the STATS 211 project requirement of modeling financial return data with a Student-t distribution under a Bayesian framework, estimating parameters via Metropolis-within-Gibbs (MWG), and computing Value-at-Risk (VaR) and Expected Shortfall (ES). Using daily S&P 500 data over approximately three years, we construct cleaned log returns and fit a hierarchical t model with latent scale variables. To improve inferential reliability, we run four independent MCMC chains and perform convergence diagnostics based on effective sample size (ESS), Geweke statistics, and Gelman-Rubin R-hat. The final posterior estimates indicate heavy tails (posterior mean of nu approximately 3.91), supporting the use of a t distribution over a Gaussian alternative. Estimated risk measures on the loss scale are VaR_0.95 = 0.01268, ES_0.95 = 0.01984, VaR_0.99 = 0.02329, and ES_0.99 = 0.03366. Overall, the model and sampler satisfy the guidance requirements and produce stable risk estimates with quantified uncertainty.

## 1. Project Objective and Scope
Following the project guidance, the objective is to:

1. Use financial data (here: S&P 500 daily closing prices, about three years).
2. Transform prices into daily returns and log returns.
3. Model returns by a Student-t distribution with unknown parameters.
4. Estimate the posterior distribution of parameters via Metropolis-within-Gibbs.
5. Compute risk measures VaR and ES using posterior output.
6. Provide convergence diagnostics and uncertainty-aware inference.

The analysis focuses on one-factor univariate market risk for daily horizon returns.

## 2. Data and Preprocessing

### 2.1 Data source and cleaned file
- Raw price series: `cleaned_returns.numbers` (internally validated and exported).
- Clean analysis file: `data/clean_returns.csv`.

### 2.2 Final dataset used for Bayesian estimation
- Number of observations: 751 daily observations.
- Date range: 2023-04-03 to 2026-03-31.
- Variables used:
  - `Close`: daily close price.
  - `SimpleReturn`: P_t / P_{t-1} - 1.
  - `LogReturn`: log(P_t / P_{t-1}).

### 2.3 Descriptive statistics (log returns)
- Mean: 0.0006164
- Standard deviation: 0.0093460
- Annualized mean (252 days): 0.1553
- Annualized volatility (252 days): 0.1484
- Min / Max: -0.06161 / 0.09089
- Skewness: 0.3715
- Excess kurtosis: 15.1956

The very high excess kurtosis strongly suggests fat tails, which motivates Student-t modeling.

## 3. Bayesian Model Specification

### 3.1 Hierarchical representation of Student-t
Let y_i denote the daily log return. We use:

- y_i | mu, sigma^2, lambda_i ~ N(mu, sigma^2 / lambda_i)
- lambda_i | nu ~ Gamma(nu/2, nu/2)

Integrating out lambda_i yields y_i ~ t_nu(mu, sigma^2), i.e., a Student-t distribution.

### 3.2 Priors
The implemented priors are weakly informative:

- mu | sigma^2 ~ N(mu_0, sigma^2 / kappa_0), with mu_0 = sample mean of y, kappa_0 = 1.
- sigma^2 ~ Inv-Gamma(a_0, b_0), with a_0 = 2, b_0 = sample variance of y.
- nu - 2 ~ Exponential(rate_nu), rate_nu = 0.1, with support nu > 2.

These priors are standard and ensure positivity constraints while allowing heavy tails.

## 4. Posterior Sampling via Metropolis-within-Gibbs

### 4.1 Algorithmic structure
At each iteration:
1. Gibbs update for latent scales lambda_i.
2. Gibbs update for location parameter mu.
3. Gibbs update for scale parameter sigma^2.
4. MH update for degrees of freedom nu via eta = log(nu - 2).

### 4.2 MCMC settings (final multi-chain run)
- Script: `scripts/02_mwg_sampler.py`
- Chains: 4 independent chains.
- Iterations per chain: 120000
- Burn-in per chain: 20000
- Thinning: 20
- Saved draws per chain: 5000
- Total posterior draws: 20000
- Proposal SD for nu update: 0.25

### 4.3 Acceptance behavior
From `outputs_multichain/sampler_meta_by_chain.csv`:
- nu acceptance rates by chain: 0.4344, 0.4337, 0.4322, 0.4359
- Mean acceptance rate: 0.4341

This acceptance range is generally healthy for random-walk MH in this setting.

## 5. Convergence Diagnostics

Diagnostics files:
- `outputs/diagnostics/diagnostics_summary_new_chain.csv`
- `outputs/diagnostics/mu_diagnostics_new_chain.png`
- `outputs/diagnostics/sigma2_diagnostics_new_chain.png`
- `outputs/diagnostics/nu_diagnostics_new_chain.png`

### 5.1 Quantitative diagnostics

For mu:
- ESS = 18770.99 (ESS/draw = 0.9385)
- R-hat = 0.99999
- Geweke z = 0.6117

For sigma^2:
- ESS = 9947.50 (ESS/draw = 0.4974)
- R-hat = 1.00010
- Geweke z = -0.6748

For nu:
- ESS = 7009.62 (ESS/draw = 0.3505)
- R-hat = 1.00034
- Geweke z = 1.0418

Interpretation:
- R-hat values are effectively 1, indicating satisfactory between-chain convergence.
- ESS values are sufficiently large for stable posterior summaries.
- nu remains the slowest-mixing parameter (expected in heavy-tail models), but current ESS is adequate.

### 5.2 Visual diagnostics
- Trace plots show no sustained drift and overlap across chains.
- Histograms plus smoothed density curves indicate unimodal posterior behavior.
- ACF is higher for nu than for mu, but materially improved versus early single-chain runs.

Overall, the final four-chain configuration is diagnostically acceptable and suitable for inference.

## 6. Posterior Parameter Estimates

From `outputs_multichain/posterior_summary.csv`:

- mu:
  - Posterior mean = 0.0009690
  - 95% credible interval = [0.0004263, 0.0015076]

- sigma^2:
  - Posterior mean = 3.9888e-05
  - 95% credible interval = [3.3158e-05, 4.7532e-05]

- nu:
  - Posterior mean = 3.9062
  - 95% credible interval = [2.9322, 5.2295]

The posterior mass for nu is concentrated at low values, confirming substantial tail heaviness and validating the Student-t choice.

## 7. Risk Measures: VaR and ES

Risk summary from `outputs_multichain/var_es_summary.csv` (loss scale):

- At confidence level 95%:
  - VaR_0.95 = 0.0126788
  - ES_0.95 = 0.0198361

- At confidence level 99%:
  - VaR_0.99 = 0.0232871
  - ES_0.99 = 0.0336584

Interpretation:
- On a typical day, the model estimates a 5% probability that loss exceeds about 1.27%.
- In the most adverse 1% tail, expected loss is about 3.37%, materially larger than the corresponding VaR threshold, as expected for fat-tailed data.

## 8. Discussion

### 8.1 Why this model is appropriate
The empirical return distribution exhibits strong excess kurtosis. A Gaussian model would understate tail risk. The Bayesian Student-t framework directly addresses this issue through the nu parameter and provides full posterior uncertainty quantification.

### 8.2 What improved over initial runs
- Transition from single-chain to four-chain estimation.
- Better MH tuning and thinning, with improved mixing for nu.
- Stronger convergence evidence (true multi-chain R-hat and high ESS).
- Cleaner visualization design for communication quality.

### 8.3 Practical risk implication
The difference between VaR and ES is substantial at both confidence levels, emphasizing that tail-conditional risk (ES) provides additional and important information beyond quantile risk (VaR).

## 9. Limitations and Extensions

Current limitations:
1. The model assumes conditionally i.i.d. returns and does not explicitly model volatility clustering (e.g., GARCH-type dynamics).
2. VaR/ES are estimated in-sample; formal out-of-sample backtesting is not yet included.
3. Prior sensitivity for nu could be investigated more systematically.

Recommended extensions:
1. Add VaR backtesting (e.g., Kupiec and Christoffersen tests).
2. Perform prior sensitivity analysis for the nu prior rate.
3. Compare with Gaussian and/or GARCH-t benchmark models.

## 10. Reproducibility and File Map

Core scripts:
- `scripts/01_export_clean_returns.py`
- `scripts/02_mwg_sampler.py`
- `scripts/03_convergence_diagnostics.py`
- `scripts/04_report_tables.py`

Main outputs used in this report:
- `outputs_multichain/posterior_summary.csv`
- `outputs_multichain/var_es_summary.csv`
- `outputs_multichain/sampler_meta.csv`
- `outputs_multichain/sampler_meta_by_chain.csv`
- `outputs/diagnostics/diagnostics_summary_new_chain.csv`
- `outputs/diagnostics/*_diagnostics_new_chain.png`

Suggested command sequence:
1. Export cleaned returns:
   - `./.venv/bin/python scripts/01_export_clean_returns.py --input cleaned_returns.numbers --output data/clean_returns.csv`
2. Run multi-chain MWG:
   - `./.venv/bin/python scripts/02_mwg_sampler.py --input data/clean_returns.csv --output-dir outputs_multichain --n-chains 4`
3. Generate diagnostics:
   - `python3 scripts/03_convergence_diagnostics.py --input outputs_multichain/posterior_samples.csv --output-dir outputs/diagnostics --file-suffix _new_chain`
4. Build report tables:
   - `./.venv/bin/python scripts/04_report_tables.py --returns data/clean_returns.csv --posterior-summary outputs_multichain/posterior_summary.csv --risk outputs_multichain/var_es_summary.csv --diagnostics outputs/diagnostics/diagnostics_summary_new_chain.csv --output-dir outputs/report`

## 11. Conclusion
This project successfully satisfies the guidance requirements: data preparation, Bayesian Student-t modeling, Metropolis-within-Gibbs posterior sampling, convergence diagnostics, and VaR/ES estimation. The final four-chain MCMC run shows strong convergence and effective sampling efficiency. Estimated risk metrics are numerically stable and economically interpretable, while posterior results provide clear evidence of heavy-tailed market behavior. The produced pipeline is reproducible and can be extended to backtesting and alternative risk models for further enhancement.

