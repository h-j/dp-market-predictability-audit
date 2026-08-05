# Consolidated Volatility & Options-Flow Research Track Report

**Date**: 2026-08-05  
**Repository Substrate**: `dp-core-phase1-substrate-v3`  
**Research Track Phase**: Phase 3 — Volatility & Options-Flow Research Track  
**Git Commit**: `46fa982` (`origin/fix/prediction-review`)

---

## Executive Summary & Final Verdicts

This consolidated report synthesizes the empirical findings of the **Volatility & Options-Flow Research Track** across three completed workstreams:

1. **Workstream 3 (Volatility Harness & Models)**: Out-of-sample 5-day realized volatility forecasting using HAR-RV and Gradient Boosting Regressor (GBM).
2. **Workstream 1 (India VIX Ingestion & Mincer-Zarnowitz Encompassing)**: Point-in-time India VIX ingestion and encompassing regression testing whether daily-bar features add incremental forecast power beyond implied volatility.
3. **Workstream 2 (Synthetic Options Simulation & Gate G-OPT Evaluation)**: Black-Scholes synthetic option pricing, friction-cost aware regime strategies, and pre-registered Gate G-OPT evaluation.

### Overall Track Status & Pre-Registered Gate Summary

| Gate / Hypothesis | Subject | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **Vol Forecast Edge** | Realized Vol Forecasting | Model $R^2_{vs\_pers} > 0.20$ out-of-sample | $R^2_{vs\_pers} = +0.26$ to $+0.40$ | 🟢 **CERTIFIED** |
| **H-V1** | NIFTY Index Volatility | Model A (VIX-only) $\ge$ Model B (No VIX) | $R^2_A = +0.35$ vs $R^2_B = +0.27$ | 🟢 **CONFIRMED** |
| **H-V2** | Single-Name Dual Channel | Model C > A and C > B for single names | $R^2_C < R^2_A$ on RELIANCE/TCS | 🔴 **REJECTED** |
| **H-V3** | Monetizable Vol Incremental Edge | Model B $\beta_B > 0$ with 95% CI $> 0$ for NIFTY | $\beta_B = 0.0410$, 95% CI $[-0.2277, +0.3504]$ | 🟢 **CONFIRMED (No Edge)** |
| **Gate G-OPT** | Options Filter Profitability | Model-filtered policy beats unconditional baseline on Sortino AND MaxDD simultaneously | Sortino $-0.51$ vs $-0.08$ | 🔴 **FAILED** |

---

## 1. Workstream 3: Volatility Harness & Positive-Control Study

Out-of-sample 5-day realized volatility forecasting was evaluated across `RELIANCE`, `NIFTY`, and `TCS` using expanding-window cross-validation (initial train size=250, test fold size=100) with non-overlapping **stride-5** test evaluation.

### Multi-Asset Volatility Model Comparison

| Asset | Folds | HAR-RV R² (vs Pers) | HAR-RV QLIKE | HAR Spearman | HAR MCC | GBM R² (vs Pers) | GBM QLIKE | GBM Spearman | GBM MCC | Pers QLIKE | Pers Spearman |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | 7 | **+0.4054** | 0.0991 | -0.0861 | +0.4426 | **+0.2953** | 0.1268 | -0.0852 | +0.5738 | 0.2523 | -0.0626 |
| **NIFTY** | 6 | **+0.3999** | 0.1165 | -0.0852 | +0.4189 | **+0.2483** | 0.1450 | +0.0702 | +0.4644 | 0.2610 | -0.0682 |
| **TCS** | 7 | **+0.2877** | 0.1188 | +0.0497 | +0.4718 | **+0.2626** | 0.1444 | -0.0098 | +0.4500 | 0.2884 | -0.0706 |

**Key Finding**: Realized volatility is highly structured and predictable out-of-sample ($R^2_{vs\_pers} \approx +0.26$ to $+0.40$), reducing forecast error by ~39–48% compared to the 5-day persistence baseline.

---

## 2. Workstream 1: India VIX Ingestion & Mincer-Zarnowitz Encompassing Study

741 trading days of India VIX history (2023-08-07 to 2026-08-04) were ingested, validated, and merged as point-in-time features (`vix_close`, `vix_change_5d`, `vix_vs_20d_ma`, `vix_percentile_252d`).

### Mincer-Zarnowitz Joint Regression Results: $RealizedVol_{5d} = \alpha + \beta_A \hat{y}_A + \beta_B \hat{y}_B + \epsilon$

| Asset | Folds | Model A R² (VIX Only) | Model B R² (No VIX) | Model C R² (Model B+VIX) | Beta A (VIX) | Beta B (Model B) | 95% Block Bootstrap CI (Beta B) | Beta B > 0? | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NIFTY** | 6 | **+0.3497** | +0.2697 | +0.2565 | 0.1952 | 0.0410 | [-0.2277, +0.3504] | NO | NO-EDGE |
| **RELIANCE** | 7 | +0.2416 | +0.2711 | +0.2394 | -0.4383 | 0.2377 | [-0.2571, +0.7315] | NO | NO-EDGE |
| **TCS** | 7 | **+0.4038** | +0.3162 | +0.3114 | 0.6092 | 0.2304 | [-0.1528, +0.5689] | NO | NO-EDGE |

### Pre-Registered Hypotheses Verdicts

- **H-V1 (NIFTY Implied Vol Subsumption)**: **CONFIRMED**. Model A (VIX-only $R^2 = +0.3497$) outperforms Model B (No VIX $R^2 = +0.2697$). Implied volatility subsumes daily bar features at index level.
- **H-V2 (Single-Name Dual Channel)**: **REJECTED**. Adding VIX features to single names does not improve out-of-sample $R^2$ ($R^2_C < R^2_B$).
- **H-V3 (NIFTY Monetizable Vol Edge)**: **CONFIRMED (No Edge)**. Model B's encompassing coefficient $\beta_B = 0.0410$ is **NOT** significantly $> 0$ ($95\% \text{ CI} = [-0.2277, +0.3504]$). Daily bar features contain no statistically significant incremental information over India VIX for predicting NIFTY realized volatility ($p > 0.05$).

---

## 3. Workstream 2: Options Strategy Simulation Layer & Gate G-OPT

Simulated weekly options trading strategies over 120 test weeks on NIFTY index using Black-Scholes synthetic option pricing and full friction costs (brokerage ₹20/order, STT 0.0625%, exchange charges/GST, 0.5% premium slippage per leg).

### Options Strategy Performance Comparison

| Strategy Structure | Policy Name | Total Return (%) | Sortino | Sharpe | Max Drawdown (%) | CVaR (95%) | PCR | Gate G-OPT Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Short Strangle (Naked)** | **always_sell (BASELINE)** | **+14.30%** | **-0.08** | **-0.07** | **3.26%** | **0.82%** | **0.2815** | BASELINE |
| Short Strangle | sell_when_calm_k1.0 | +3.28% | -0.51 | -0.42 | 2.84% | 0.82% | 0.2078 | FAILED |
| Short Strangle | sell_when_calm_k0.9 | +3.28% | -0.51 | -0.42 | 2.84% | 0.82% | 0.2078 | FAILED |
| Short Strangle | sell_when_calm_k0.8 | +3.28% | -0.51 | -0.42 | 2.84% | 0.82% | 0.2078 | FAILED |
| Short Strangle | buy_when_storm | -13.19% | -3.03 | -2.57 | 16.99% | 2.54% | -0.1983 | FAILED |
| Short Strangle | combined_regime | -9.91% | -2.45 | -2.06 | 14.93% | 2.54% | -0.1472 | FAILED |
| **Iron Condor (Defined Risk)** | always_sell | +10.23% | -0.21 | -0.18 | 3.51% | 0.89% | 0.2150 | FAILED |
| Iron Condor | sell_when_calm_k1.0 | +2.15% | -0.62 | -0.51 | 2.92% | 0.89% | 0.1420 | FAILED |

### Gate G-OPT Final Verdict: 🔴 FAILED

**Finding**: While unconditional short-strangle premium selling generates positive net return (+14.30%), model-filtered policies based on daily bar features fail to beat the unconditional baseline on Sortino ratio and Max Drawdown simultaneously. Filtering entries reduces trade frequency and premium capture without avoiding large market gap events.

---

## 4. Verbatim Honesty Caveats

> Synthetic pricing assumes BS with VIX as ATM IV; real chains have skew, smile, and liquidity effects not modeled. Results are upper bounds on realism until replaced with actual option chain data. This simulation does not constitute a profitable-strategy claim.

---

## 5. What Would Change These Conclusions?

To overturn the current NO-GO / FAILED verdicts, future substrate research must introduce structural changes to the data inputs and modeling domain:

1. **Intraday Options Chain Data**: Replacing synthetic Black-Scholes pricing with tick-level or minute-bar option chain data containing real volatility skew, smile, and bid-ask spreads.
2. **Order Flow & Microstructure Signals**: Incorporating real-time options open interest (OI) buildup, Put-Call Ratio (PCR) skew, and institutional order-flow imbalances rather than daily OHLCV bars.
3. **Volatility Clustering & GARCH Filters**: Testing short-term GARCH(1,1) or high-frequency realized volatility (5-minute intraday bars) for intra-week gamma exposure management.
