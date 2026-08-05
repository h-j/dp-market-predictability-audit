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
| **sell_when_calm_k1.0** | +3.28% | -0.51 | -2.13 | 2.84% | 0.03% | 0.3739 | FAILED |
| **sell_when_calm_k0.9** | +3.28% | -0.51 | -2.13 | 2.84% | 0.03% | 0.3739 | FAILED |
| **sell_when_calm_k0.8** | +3.28% | -0.51 | -2.13 | 2.84% | 0.03% | 0.3739 | FAILED |
| **buy_when_storm** | -13.19% | -3.03 | -1.49 | 16.99% | 1.98% | 0.0000 | FAILED |
| **combined_regime_k1.0** | -9.91% | -2.45 | -1.26 | 14.93% | 2.18% | -1.1282 | FAILED |

## Iron Condor Strategy Results (Defined-Risk Structure)

| Policy | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **always_sell** | +10.29% | -0.61 | -0.60 | 2.86% | 1.58% | 0.2391 | FAILED |
| **sell_when_calm_k1.0** | +3.14% | -1.43 | -3.66 | 1.33% | 0.02% | 0.4799 | FAILED |
| **sell_when_calm_k0.9** | +3.14% | -1.43 | -3.66 | 1.33% | 0.02% | 0.4799 | FAILED |
| **sell_when_calm_k0.8** | +3.14% | -1.43 | -3.66 | 1.33% | 0.02% | 0.4799 | FAILED |
| **buy_when_storm** | -13.19% | -3.03 | -1.49 | 16.99% | 1.98% | 0.0000 | FAILED |
| **combined_regime_k1.0** | -10.05% | -2.70 | -1.30 | 14.77% | 1.98% | -1.5348 | FAILED |

---

## Verbatim Honesty Caveats

> Synthetic pricing assumes BS with VIX as ATM IV; real chains have skew, smile, and liquidity effects not modeled. Results are upper bounds on realism until replaced with actual option chain data. This simulation does not constitute a profitable-strategy claim.
