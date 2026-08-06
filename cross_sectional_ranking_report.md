# Cross-Sectional Ranking Study Report (Phase 2b)

**Date**: 2026-08-05  
**Substrate**: `dp-market-predictability-audit`  
**Evaluation Window**: 35 Rebalance Months  
**Final Track Verdict**: **`CROSS-SECTIONAL NULL`**

## Pre-Registered Gate Summary

| Gate / Metric | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :---: | :---: |
| **G-XS1 (Signal Existence)** | Composite Mean IC > 0 & CI > 0 & > Null P97.5 | Mean IC=-0.0063, 95% CI=[-0.0536, +0.0407], Null P97.5=+0.0330 | 🔴 **FAILED** |
| **G-XS2 (Retail Implementable)** | Top Quintile Sharpe > NIFTY Sharpe (after 0.25% cost) | Top-Quintile Sharpe=0.45 vs NIFTY Sharpe=0.47 | 🔴 **FAILED** |

### Final Track Decision: **`CROSS-SECTIONAL NULL`**

Composite cross-sectional ranking contains no statistically significant edge over random permutations. The information-value research program's empirical phase closes with a definitive CROSS-SECTIONAL NULL.

## Signal Rank Information Coefficient (IC) Summary Table

| Signal | Mean IC | Std IC | t-statistic | Bootstrap 95% CI | Null P97.5 | G-XS1 Status | Q5-Q1 Spread (Ann %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| mom_6m1m | +0.0222 | 0.1998 | +0.66 | [-0.0425, +0.0858] | +0.0332 | FAIL | +7.69% |
| rev_1m | +0.0048 | 0.2029 | +0.14 | [-0.0604, +0.0672] | +0.0353 | FAIL | +0.54% |
| rs_nifty_3m | -0.0194 | 0.1983 | -0.58 | [-0.0820, +0.0432] | +0.0329 | FAIL | -1.04% |
| vol_3m_inv | -0.0577 | 0.2673 | -1.28 | [-0.1408, +0.0334] | +0.0349 | FAIL | -18.52% |
| vol_trend | +0.0275 | 0.1361 | +1.20 | [-0.0165, +0.0699] | +0.0344 | FAIL | +3.44% |
| **composite** | -0.0063 | 0.1512 | -0.25 | [-0.0536, +0.0407] | +0.0330 | FAIL | -6.35% |

## Long-Only Equal-Weighted Top Quintile (Q5) Portfolio Performance (0.25% Costs)

| Portfolio Signal | CAGR (%) | Sharpe Ratio | Max Drawdown (%) | Mean Monthly Turnover | Total Costs (%) | Active Return vs NIFTY (%) | Win Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Q5 (mom_6m1m) | +24.33% | 0.87 | 16.29% | 0.67 | 5.88% | +18.33% | 65.71% |
| Q5 (rev_1m) | +20.66% | 0.81 | 20.66% | 1.56 | 13.64% | +14.66% | 57.14% |
| Q5 (rs_nifty_3m) | +25.10% | 0.92 | 18.89% | 0.86 | 7.54% | +19.10% | 74.29% |
| Q5 (vol_3m_inv) | +13.17% | 0.49 | 17.07% | 0.54 | 4.75% | +7.17% | 57.14% |
| Q5 (vol_trend) | +27.20% | 1.11 | 20.87% | 1.37 | 12.01% | +21.20% | 71.43% |
| **Q5 Top Quintile (composite)** | +12.44% | 0.45 | 19.06% | 1.21 | 10.62% | +6.44% | 62.86% |
| **NIFTY 100 Equal-Weight Benchmark** | +18.96% | 0.72 | 18.06% | 0.00 | 0.00% | +6.88% | 68.57% |
| **NIFTY Benchmark (B&H)** | +12.08% | 0.47 | 13.85% | 0.00 | 0.00% | 0.00% | 62.86% |

---

## Mandatory Survivorship Bias Caveat

> Backfilling current NIFTY 100 constituent tickers introduces survivorship bias, inflating long-side returns. Any passing verdict must be treated as upper-bound candidate edge pending historical constituent verification.
