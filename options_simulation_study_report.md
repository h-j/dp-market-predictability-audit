# Options Strategy Simulation Study & Gate G-OPT Report (Workstream 2)

## Executive Summary

Evaluated synthetic options trading strategies on NIFTY weekly expiry using Model C volatility forecasts vs India VIX. Tested unconditional variance risk premium harvesting against model-filtered policies across naked and defined-risk structures.

### 🔴 GATE G-OPT VERDICT: FAILED

No model-filtered policy beat the unconditional short-strangle baseline on both Sortino ratio and Max Drawdown simultaneously.

**Research Conclusion**: Harvesting the unconditional variance risk premium yields positive return, but model-filtering based on daily bar features does not produce a statistically superior risk-adjusted profile over unconditional premium selling.


## Short Strangle Strategy Results (Naked Structure)

| Policy | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **always_sell** | +14.30% | -0.08 | -0.11 | 3.26% | 2.14% | 0.2815 | FAILED |
| **sell_when_calm_k1.0** | +12.27% | -0.23 | -0.29 | 3.30% | 2.14% | 0.2517 | FAILED |
| **sell_when_calm_k0.9** | +10.01% | -0.38 | -0.50 | 3.37% | 2.14% | 0.2155 | FAILED |
| **sell_when_calm_k0.8** | +8.06% | -0.49 | -0.73 | 3.52% | 2.02% | 0.2089 | FAILED |
| **buy_when_storm** | +3.67% | -1.10 | -1.27 | 2.11% | 0.04% | 0.0000 | FAILED |
| **combined_regime_k1.0** | +10.42% | -0.35 | -0.45 | 3.36% | 2.24% | 0.2137 | FAILED |

## Iron Condor Strategy Results (Defined-Risk Structure)

| Policy | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **always_sell** | +10.29% | -0.61 | -0.60 | 2.86% | 1.58% | 0.2391 | FAILED |
| **sell_when_calm_k1.0** | +8.67% | -0.80 | -0.80 | 2.90% | 1.58% | 0.2094 | FAILED |
| **sell_when_calm_k0.9** | +6.70% | -1.03 | -1.04 | 2.96% | 1.58% | 0.1703 | FAILED |
| **sell_when_calm_k0.8** | +5.98% | -1.12 | -1.24 | 2.93% | 1.46% | 0.1834 | FAILED |
| **buy_when_storm** | +3.67% | -1.10 | -1.27 | 2.11% | 0.04% | 0.0000 | FAILED |
| **combined_regime_k1.0** | +6.82% | -0.96 | -0.96 | 2.95% | 1.67% | 0.1647 | FAILED |

---

## Verbatim Honesty Caveats

> Synthetic pricing assumes BS with VIX as ATM IV; real chains have skew, smile, and liquidity effects not modeled. Results are upper bounds on realism until replaced with actual option chain data. This simulation does not constitute a profitable-strategy claim.
