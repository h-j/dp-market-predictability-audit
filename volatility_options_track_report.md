# Consolidated Volatility & Options-Flow Research Track Report

**Date**: 2026-08-05  
**Repository Substrate**: `dp-core-phase1-substrate-v3`  
**Research Track Phase**: Phase 3 — Volatility & Options-Flow Research Track  
**Git Commit**: `fix(options): correct forecast annualization units, add regression tests, rerun Gate G-OPT`

---

## Executive Summary & Final Verdicts

This consolidated report synthesizes the empirical findings of the **Volatility & Options-Flow Research Track** across three completed workstreams, including the **corrected forecast annualization unit fix** for Workstream 2.

> [!CAUTION]
> **Prior Audit Note**: The Gate G-OPT verdict in commit `46fa982` was **INVALIDATED - units defect (commit 46fa982)** due to an un-annualized daily volatility conversion bug. Below are the corrected, certified results following explicit annualization ($\hat{\sigma}_{ann} = \hat{\sigma}_{daily} \times \sqrt{252}$).

### Overall Track Status & Pre-Registered Gate Summary

| Gate / Hypothesis | Subject | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **Vol Forecast Edge** | Realized Vol Forecasting | Model $R^2_{vs\_pers} > 0.20$ out-of-sample | $R^2_{vs\_pers} = +0.26$ to $+0.40$ | 🟢 **CERTIFIED** |
| **H-V1** | NIFTY Index Volatility | Model A (VIX-only) $\ge$ Model B (No VIX) | $R^2_A = +0.35$ vs $R^2_B = +0.27$ | 🟢 **CONFIRMED** |
| **H-V2** | Single-Name Dual Channel | Model C > A and C > B for single names | $R^2_C < R^2_A$ on RELIANCE/TCS | 🔴 **REJECTED** |
| **H-V3** | Monetizable Vol Incremental Edge | Model B $\beta_B > 0$ with 95% CI $> 0$ for NIFTY | $\beta_B = 0.0410$, 95% CI $[-0.2277, +0.3504]$ | 🟢 **CONFIRMED (No Edge)** |
| **Gate G-OPT (Prior)** | Options Filter Profitability | Commit 46fa982 verdict | Units defect in conversion heuristic | ⚠️ **INVALIDATED - units defect (commit 46fa982)** |
| **Gate G-OPT (Corrected)** | Options Filter Profitability | Model-filtered policy beats unconditional baseline on Sortino AND MaxDD simultaneously | Sortino $-0.23$ vs $-0.08$, MaxDD $3.30\%$ vs $3.26\%$ | 🔴 **FAILED** |

---

## 1. Workstream 3: Volatility Harness & Positive-Control Study

Out-of-sample 5-day realized volatility forecasting was evaluated across `RELIANCE`, `NIFTY`, and `TCS` using expanding-window cross-validation (initial train size=250, test fold size=100) with non-overlapping **stride-5** test evaluation.

### Multi-Asset Volatility Model Comparison

| Asset | Folds | HAR-RV R² (vs Pers) | HAR-RV QLIKE | HAR Spearman | HAR vol-rise MCC | GBM R² (vs Pers) | GBM QLIKE | GBM Spearman | GBM vol-rise MCC | Pers QLIKE | Pers Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | 7 | **+0.4054** | 0.0991 | -0.0861 | +0.4426 | **+0.2953** | 0.1268 | -0.0852 | +0.5738 | 0.2523 | -0.0626 |
| **NIFTY** | 6 | **+0.3999** | 0.1165 | -0.0852 | +0.4189 | **+0.2483** | 0.1450 | +0.0702 | +0.4644 | 0.2610 | -0.0682 |
| **TCS** | 7 | **+0.2877** | 0.1188 | +0.0497 | +0.4718 | **+0.2626** | 0.1444 | -0.0098 | +0.4500 | 0.2884 | -0.0706 |

---

## 2. Workstream 1: India VIX Ingestion & Mincer-Zarnowitz Encompassing Study

741 trading days of India VIX history (2023-08-07 to 2026-08-04) were ingested, validated, and merged as point-in-time features (`vix_close`, `vix_change_5d`, `vix_vs_20d_ma`, `vix_percentile_252d`).

### Mincer-Zarnowitz Joint Regression Results: $RealizedVol_{5d} = \alpha + \beta_A \hat{y}_A + \beta_B \hat{y}_B + \epsilon$

| Asset | Folds | Model A R² (VIX Only) | Model B R² (No VIX) | Model C R² (Model B+VIX) | Beta A (VIX) | Beta B (Model B) | 95% Block Bootstrap CI (Beta B) | Beta B > 0? | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NIFTY** | 6 | **+0.3497** | +0.2697 | +0.2565 | 0.1952 | 0.0410 | [-0.2277, +0.3504] | NO | NO-EDGE |
| **RELIANCE** | 7 | +0.2416 | +0.2711 | +0.2394 | -0.4383 | 0.2377 | [-0.2571, +0.7315] | NO | NO-EDGE |
| **TCS** | 7 | **+0.4038** | +0.3162 | +0.3114 | 0.6092 | 0.2304 | [-0.1528, +0.5689] | NO | NO-EDGE |

---

## 3. Workstream 2: Corrected Options Strategy Simulation & Gate G-OPT

Corrected weekly options trading strategy simulations on NIFTY index (119 test weeks) using Black-Scholes synthetic option pricing and full friction costs (brokerage ₹20/order, STT 0.0625%, exchange charges/GST, 0.5% premium slippage per leg). Forecasts annualized explicitly via $\hat{\sigma}_{ann} = \hat{\sigma}_{daily} \times \sqrt{252}$.

### Corrected Options Strategy Performance Comparison

| Strategy Structure | Policy Name | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short Strangle (Naked)** | **always_sell (BASELINE)** | +14.30% | -0.08 | -0.11 | 3.26% | 2.14% | 0.2815 | BASELINE |
| Short Strangle | sell_when_calm_k1.0 | +12.27% | -0.23 | -0.29 | 3.30% | 2.14% | 0.2517 | FAILED |
| Short Strangle | sell_when_calm_k0.9 | +10.01% | -0.38 | -0.50 | 3.37% | 2.14% | 0.2155 | FAILED |
| Short Strangle | sell_when_calm_k0.8 | +8.06% | -0.49 | -0.73 | 3.52% | 2.02% | 0.2089 | FAILED |
| Short Strangle | buy_when_storm | +3.67% | -1.10 | -1.27 | 2.11% | 0.04% | 0.0000 | FAILED |
| Short Strangle | combined_regime_k1.0 | +10.42% | -0.35 | -0.45 | 3.36% | 2.24% | 0.2137 | FAILED |
| **Iron Condor (Defined Risk)** | **always_sell** | +10.29% | -0.61 | -0.60 | 2.86% | 1.58% | 0.2391 | FAILED |
| Iron Condor | sell_when_calm_k1.0 | +8.67% | -0.80 | -0.80 | 2.90% | 1.58% | 0.2094 | FAILED |
| Iron Condor | sell_when_calm_k0.9 | +6.70% | -1.03 | -1.04 | 2.96% | 1.58% | 0.1703 | FAILED |
| Iron Condor | sell_when_calm_k0.8 | +5.98% | -1.12 | -1.24 | 2.93% | 1.46% | 0.1834 | FAILED |
| Iron Condor | buy_when_storm | +3.67% | -1.10 | -1.27 | 2.11% | 0.04% | 0.0000 | FAILED |
| Iron Condor | combined_regime_k1.0 | +6.82% | -0.96 | -0.96 | 2.95% | 1.67% | 0.1647 | FAILED |

### Gate G-OPT Final Verdict: 🔴 FAILED (Re-Verified)

**Corrected Result**:
- Prior verdict in commit `46fa982`: **INVALIDATED - units defect (commit 46fa982)**.
- New corrected verdict: **FAILED**.
- Reason: Unconditional short-strangle premium selling yields **+14.30%** return (Sortino `-0.08`, MaxDD `3.26%`). Model-filtered policies (e.g. `sell_when_calm` at $k=1.0$) yield **+12.27%** (Sortino `-0.23`, MaxDD `3.30%`), failing to beat the unconditional baseline on Sortino and Max Drawdown simultaneously.

---

## 4. Verbatim Honesty Caveats

> Synthetic pricing assumes BS with VIX as ATM IV; real chains have skew, smile, and liquidity effects not modeled. Results are upper bounds on realism until replaced with actual option chain data. This simulation does not constitute a profitable-strategy claim.

---

## 5. What Would Change These Conclusions?

To overturn the current NO-GO / FAILED verdicts, future substrate research must introduce structural changes to the data inputs and modeling domain:

1. **Intraday Options Chain Data**: Replacing synthetic Black-Scholes pricing with tick-level or minute-bar option chain data containing real volatility skew, smile, and bid-ask spreads.
2. **Order Flow & Microstructure Signals**: Incorporating real-time options open interest (OI) buildup, Put-Call Ratio (PCR) skew, and institutional order-flow imbalances rather than daily OHLCV bars.
3. **Volatility Clustering & GARCH Filters**: Testing short-term GARCH(1,1) or high-frequency realized volatility (5-minute intraday bars) for intra-week gamma exposure management.
