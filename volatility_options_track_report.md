# Consolidated Volatility & Options-Flow Research Track Report

**Date**: 2026-08-05  
**Repository Substrate**: `dp-core-phase1-substrate-v3`  
**Research Program**: Consolidated Substrate Empirical Research Program (Phases 1B – 4)  
**Git Tag**: `v1.0-research-complete`

---

## Executive Summary & Final Verdicts

This consolidated report synthesizes the empirical findings of the **Reflective Cognition Substrate Research Program** across all seven completed research tracks.

> [!CAUTION]
> **Prior Audit Note**: The initial Gate G-OPT verdict in commit `46fa982` was **INVALIDATED due to a unit conversion defect** ($\hat{\sigma}_{daily}$ vs $\hat{\sigma}_{ann}$). All results below reflect the corrected, certified forecast annualization ($\hat{\sigma}_{ann} = \hat{\sigma}_{daily} \times \sqrt{252}$).

---

### Master Program Status & Pre-Registered Gate Summary (All 7 Research Tracks)

| # | Research Track / Study | Pre-Registered Gate | Pre-Registered Condition | Empirical Result | Final Verdict | Study Report Link |
| :-: | :--- | :--- | :--- | :---: | :---: | :--- |
| **1** | **Daily/3-Day Single-Name Direction** | Directional Predictability | MCC $> 0.10$ out-of-sample | Mean $MCC \approx 0.00$ on RELIANCE/NIFTY/TCS | 🔴 **`DIRECTION IS DEAD`** | [RESEARCH_FINDINGS.md](file:///Users/hemantj/Proj/dp_core/dp-core-phase1-substrate-v3/RESEARCH_FINDINGS.md#track-1-single-name--index-daily3-day-direction) |
| **2** | **5-Day Realized Volatility Forecasting** | Realized Vol Edge | HAR-RV / EWMA / GB $R^2_{\text{vs\_pers}} > 0.20$ | $R^2_{\text{vs\_pers}} = +0.26$ to $+0.40$ across assets | 🟢 **`VOLATILITY IS ALIVE`** | [Section 1](#1-workstream-3-volatility-harness--positive-control-study) |
| **3** | **India VIX Encompassing Test** | Gate G-VIX / H-V3 | Daily-bar model adds incremental edge over VIX ($\beta_B > 0, 95\% \text{ CI} > 0$) | $\beta_B = 0.0410$, $95\% \text{ CI} [-0.2277, +0.3504]$ | 🟢 **`H-V3 CONFIRMED (NO EDGE BEYOND VIX)`** | [Section 2](#2-workstream-1-india-vix-ingestion--mincer-zarnowitz-encompassing-study) |
| **4** | **Options Volatility Regime Strategy** | Gate G-OPT (Corrected) | Model-filtered policy beats unconditional baseline on Sortino AND MaxDD | Sortino $-0.23$ vs $-0.08$, MaxDD $3.30\%$ vs $3.26\%$ | 🔴 **`G-OPT FAILED`** | [Section 3](#3-workstream-2-corrected-options-strategy-simulation--gate-g-opt) |
| **5** | **Cross-Sectional Ranking Study** | Gate G-XS1 & G-XS2 | Composite IC $> 0$ & Top Quintile Sharpe $>$ NIFTY/Equal-Weight Benchmark | Mean IC = $-0.0063$, Q5 Sharpe = $0.45$ vs Equal-Weight $0.72$ | 🔴 **`CROSS-SECTIONAL NULL`** | [cross_sectional_ranking_report.md](file:///Users/hemantj/Proj/dp_core/dp-core-phase1-substrate-v3/cross_sectional_ranking_report.md) |
| **6** | **LLM Hypothesis Generation (Initial Track)** | Gate G-LLM1 & G-LLM2 | Out-of-sample LLM survival rate > random grammar baseline | Fisher $p=1.0000$, survival rate 0% | ⚠️ **`SUPERSEDED BY PHASE 3 REBUILD`** | [llm_hypothesis_value_prereg.md](file:///Users/hemantj/Proj/dp_core/dp-core-phase1-substrate-v3/experiments/preregistration/llm_hypothesis_value_prereg.md) |
| **7** | **Substrate LLM Hypothesis Survival (Phase 3 Rebuild)** | Gate G-P3 | Pooled one-sided Mann-Whitney $p < 0.05$ on survivor OOS edge | LLM Survivor Edge = $+0.1948$ vs Random = $+0.2493$ ($p = 0.6876$) | 🔴 **`LLM REASONING NULL`** | [llm_hypothesis_survival_report.md](file:///Users/hemantj/Proj/dp_core/dp-core-phase1-substrate-v3/llm_hypothesis_survival_report.md) |

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

## 4. Workstream 4: Cross-Sectional Ranking Study

Evaluated point-in-time cross-sectional signals (`mom_6m1m`, `rev_1m`, `rs_nifty_3m`, `vol_3m_inv`, `vol_trend`) across NIFTY 100 constituents over 35 rebalance months.

### Long-Only Top Quintile (Q5) Portfolio Performance vs Benchmarks (0.25% Transaction Costs)

| Portfolio Signal / Benchmark | CAGR (%) | Sharpe Ratio | Max Drawdown (%) | Mean Monthly Turnover | Total Costs (%) | Active Return vs NIFTY (%) | Win Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Q5 (mom_6m1m) | +24.33% | 0.87 | 16.29% | 0.67 | 5.88% | +18.33% | 65.71% |
| Q5 (rev_1m) | +20.66% | 0.81 | 20.66% | 1.56 | 13.64% | +14.66% | 57.14% |
| Q5 (rs_nifty_3m) | +25.10% | 0.92 | 18.89% | 0.86 | 7.54% | +19.10% | 74.29% |
| Q5 (vol_3m_inv) | +13.17% | 0.49 | 17.07% | 0.54 | 4.75% | +7.17% | 57.14% |
| Q5 (vol_trend) | +27.20% | 1.11 | 20.87% | 1.37 | 12.01% | +21.20% | 71.43% |
| **Q5 Top Quintile (composite)** | +12.44% | 0.45 | 19.06% | 1.21 | 10.62% | +6.44% | 62.86% |
| **NIFTY 100 Equal-Weight Benchmark** | **+18.96%** | **0.72** | **18.06%** | **0.00** | **0.00%** | **+6.88%** | **68.57%** |
| **NIFTY Benchmark (B&H)** | **+12.08%** | **0.47** | **13.85%** | **0.00** | **0.00%** | **0.00%** | **62.86%** |

---

## 5. Phase 3 Rebuild: Substrate LLM Reasoning Survival Study

Evaluated 150 Substrate LLM hypotheses ($H_{\text{LLM}}$) vs 150 Control Random hypotheses ($H_{\text{Random}}$) over 60/40 In-Sample/Out-of-Sample temporal splits across `NIFTY`, `RELIANCE`, `TCS`.

| Gate / Hypothesis | Subject | Pre-Registered Condition | Result | Verdict |
| :--- | :--- | :--- | :---: | :---: |
| **Gate G-P3** | Survivor OOS Edge Superiority | Pooled One-Sided Mann-Whitney p < 0.05 | LLM Survivor Edge=+0.1948 vs Random=+0.2493 (MW p=0.6876) | 🔴 **FAILED** |

> Substrate LLM hypothesis generation on daily-bar technical and volatility features demonstrates no statistically significant out-of-sample edge over random grammar synthesis (Mann-Whitney p = 0.6876). While both arms yield a small set of in-sample surviving hypotheses with positive out-of-sample edge (+19.48% vs +24.93%), LLM-guided prompt synthesis fails to generate superior out-of-sample performance compared to random structural sampling.

---

## 6. Verbatim Honesty & Methodological Caveats

1. **Option Pricing Realism**: Synthetic pricing assumes Black-Scholes with India VIX as ATM IV; real option chains contain skew, smile, and liquidity effects not modeled in daily bars.
2. **Survivorship Bias**: Backfilling current NIFTY 100 constituent tickers introduces survivorship bias, inflating long-side returns on unconstrained universe backtests.
3. **Daily-Bar Granularity Limit**: Daily OHLCV bars do not capture intraday market microstructure, options open interest (OI) dynamics, or order flow imbalances.

---

## 7. What Would Change These Conclusions?

To overturn the current NO-GO / FAILED verdicts, future substrate research must transition from daily technical bars to high-frequency options infrastructure:

1. **Intraday Options Chain Data**: Replacing synthetic Black-Scholes pricing with tick-level or minute-bar option chain data containing real volatility skew, smile, and bid-ask spreads.
2. **Order Flow & Microstructure Signals**: Incorporating real-time options open interest (OI) buildup, Put-Call Ratio (PCR) skew, and institutional order-flow imbalances rather than daily OHLCV bars.
3. **High-Frequency Volatility Filters**: Testing 5-minute intraday realized volatility and intraday GARCH(1,1) filters for intra-week gamma exposure management.
